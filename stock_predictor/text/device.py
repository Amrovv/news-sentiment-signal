"""Where the transformer models in this package run: GPU when one is usable,
CPU otherwise.

The judge (`coref_judge`) picks its own device through llama.cpp's own
`n_gpu_layers` setting, since it is a GGUF model rather than a torch one. This
module covers the three torch models the pipeline loads -- FinBERT
(`sentiment`), DeBERTa ABSA (`absa`) and fastcoref (`coref`) -- so the choice is
made once and identically for all of them.

Fails soft, in both directions. `resolve_device()` returns "cpu" when no CUDA
device is visible, and `to_device()` leaves a model on the CPU if the move
raises (a busy or too-small card, a driver mismatch). CPU-only has to stay a
working configuration, not just a theoretical fallback: it is what this project
was built around, and a machine without a GPU must still complete a run rather
than fail one.

Device is resolved once per process and memoised, so a run cannot end up with
one model on the GPU and the next on the CPU because the card filled up in
between -- that split would be silent and confusing, and the memoised answer
plus the per-model fallback keeps the failure legible in the log instead.
"""

from loguru import logger

from stock_predictor.config import TEXT_DEVICE

_RESOLVED: str | None = None


def resolve_device() -> str:
    """The torch device string the text models should load onto.

    Honours `TEXT_DEVICE` when it names a device explicitly; on the default
    "auto" it reports "cuda" only when torch can actually see a CUDA device.
    Any failure to answer that question is treated as "no GPU", never as an
    error: this decides where a model runs, and being wrong must cost speed
    rather than the run.
    """
    global _RESOLVED
    if _RESOLVED is not None:
        return _RESOLVED

    configured = (TEXT_DEVICE or "auto").strip().lower()
    if configured != "auto":
        _RESOLVED = configured
        logger.info(f"Text models pinned to '{_RESOLVED}' by TEXT_DEVICE")
        return _RESOLVED

    try:
        import torch

        if torch.cuda.is_available():
            _RESOLVED = "cuda"
            logger.info(f"CUDA available: text models will run on {torch.cuda.get_device_name(0)}")
        else:
            _RESOLVED = "cpu"
            logger.info("No CUDA device visible; text models will run on CPU")
    except Exception as exc:  # noqa: BLE001 - any failure means "assume no GPU"
        _RESOLVED = "cpu"
        logger.warning(
            f"Could not query CUDA ({type(exc).__name__}: {exc}); text models will run on CPU"
        )
    return _RESOLVED


def to_device(model, label: str):
    """Move a loaded torch model onto `resolve_device()`, or leave it on the CPU.

    Returns the model either way. `label` names the model in the log so a
    fallback says which one stayed behind.
    """
    device = resolve_device()
    if device == "cpu":
        return model
    try:
        moved = model.to(device)
        logger.info(f"{label} running on {device}")
        return moved
    except Exception as exc:  # noqa: BLE001 - a failed move must not fail the run
        logger.warning(
            f"{label} could not be moved to {device} ({type(exc).__name__}: {exc}); "
            "continuing on CPU"
        )
        return model


def release_gpu() -> None:
    """Drop the memoised torch models and hand their GPU memory back.

    The pipeline's phases are sequential but its models are process-lifetime
    globals, so without this the coref, FinBERT and ABSA weights sit on the card
    long after their phase is done -- and the judge, which wants ~4.4GB of GGUF
    through llama.cpp, then has nothing left to load into. On a 24GB card that
    is merely wasteful; on an 8GB one it is the difference between a working
    judge and one that fails closed and marks every row `unsure`.

    Safe to call when nothing was ever loaded, when torch is absent, or when
    running CPU-only: it clears the caches either way and skips what does not
    apply.
    """
    import gc

    from stock_predictor.text import absa, coref

    coref._MODEL = None
    coref._DEVICE = None
    absa._MODEL = None
    absa._TOKENIZER = None
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Released GPU memory held by the text models")
    except Exception as exc:  # noqa: BLE001 - freeing memory must never fail a run
        logger.warning(f"Could not empty the CUDA cache ({type(exc).__name__}: {exc})")


def inputs_to(inputs, model):
    """Move a tokenizer's batch onto whatever device `model` actually sits on.

    Read from the model rather than from `resolve_device()` so an injected or
    deliberately CPU-pinned model still gets matching inputs, and so a model
    that fell back to the CPU above is not handed GPU tensors.
    """
    device = getattr(model, "device", None)
    if device is None:
        return inputs
    return {k: v.to(device) for k, v in inputs.items()}
