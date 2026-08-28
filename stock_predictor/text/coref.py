"""Neural coreference over article bodies. Backend for entity_filter.

Roughly a third of target sentences refer back to the company instead of naming
it, and a coref model reads the mention chain that word-order rules cannot.

Spans are character offsets into the string passed in, so callers must pass the
same string they will index later: in the pipeline that is the
_fix_missing_space()-cleaned body. absa.py reuses the spans to substitute the name.

Best-effort throughout: if fastcoref is missing or inference raises, is_available()
is False and resolve_documents() returns empty clusters, leaving entity_filter to
tag on explicit names alone.

Cache: keyed by sentiment.hash_text() of the cleaned body, clusters JSON-encoded
into one column. Saves MERGE with the cache loaded at call start.

    is_available()       whether the backend imported and loaded
    load_cache()         the on-disk cluster cache (empty frame on any error)
    save_cache()         persist, deduped on text_hash (keep last)
    resolve_documents()  per document, a list of clusters of (start, end)
"""

import json
import logging

from loguru import logger
import pandas as pd

from stock_predictor.config import (
    COREF_BATCH_SIZE,
    COREF_BATCH_SIZE_GPU,
    COREF_CACHE_PATH,
    COREF_MODEL,
)
from stock_predictor.text.sentiment import hash_text

# Loaded once per process. _LOAD_FAILED latches so a broken backend warns once.
_MODEL = None
_LOAD_FAILED = False
# The device the model actually loaded onto, which is not necessarily the one
# asked for: _load_model() falls back to the CPU. Read by resolve_documents()
# to size its batches, so a fallback gets CPU-sized batches rather than the
# small GPU ones.
_DEVICE: str | None = None

# Cluster spans are (char_start, char_end) into the document text.
Span = tuple[int, int]
Cluster = list[Span]

CACHE_COLUMNS = ["text_hash", "clusters_json"]


def _load_model():
    """Load (once, then memoise) the fastcoref model. Returns None on failure.

    Tries the GPU first when one is available, falling back to CPU; see the
    comment on the loop below for why the fallback matters here specifically.

    Wrapped rather than imported bare for two compatibility reasons: fastcoref 2.1.x
    predates transformers 5.x and its FCorefModel lacks `all_tied_weights_keys`,
    which from_pretrained() reads unconditionally, so the shim below supplies the
    empty mapping; and fastcoref logs at INFO through stdlib logging, which would
    swamp loguru output.
    """
    global _MODEL, _LOAD_FAILED, _DEVICE
    if _MODEL is not None or _LOAD_FAILED:
        return _MODEL

    logging.getLogger("fastcoref").setLevel(logging.WARNING)

    # GPU first, CPU second. Inference here is the pipeline's other long silent
    # stage (FCoref.predict runs a transformer per document with no progress of
    # its own), so the device matters; a GPU-specific failure must still leave a
    # working CPU run rather than turning coref off altogether, which would
    # silently drop every coref-resolved sentence from the corpus.
    from stock_predictor.text.device import resolve_device

    devices = ["cpu"] if resolve_device() == "cpu" else [resolve_device(), "cpu"]
    for i, device in enumerate(devices):
        try:
            from fastcoref import FCoref
            from fastcoref.modeling import FCorefModel

            # fastcoref 2.1.x predates transformers 5.x and its FCorefModel lacks
            # `all_tied_weights_keys`, which from_pretrained() reads
            # unconditionally, so supply the empty mapping.
            if not hasattr(FCorefModel, "all_tied_weights_keys"):
                FCorefModel.all_tied_weights_keys = {}

            logger.info(f"Loading coreference model '{COREF_MODEL}' ({device})")
            _MODEL = FCoref(
                model_name_or_path=COREF_MODEL, device=device, enable_progress_bar=False
            )
            _DEVICE = device
            logger.info(f"Coreference model loaded ({device})")
            return _MODEL
        except Exception as exc:  # noqa: BLE001 - any failure means "fall back"
            if i < len(devices) - 1:
                logger.warning(
                    f"Coreference model failed to load on {device} "
                    f"({type(exc).__name__}: {exc}); falling back to CPU"
                )
                continue
            _LOAD_FAILED = True
            logger.warning(
                f"Coreference backend unavailable ({type(exc).__name__}: {exc}); "
                "tagging will use explicit company names only"
            )
            _MODEL = None
    return _MODEL


def is_available() -> bool:
    """True when the backend imported and the model loaded.

    The enable-gate: a False must mean "tag on explicit names alone", never "raise".
    """
    return _load_model() is not None


def _serialize_clusters(clusters: list[Cluster]) -> str:
    """Clusters -> JSON string, the on-disk form of the cache's value column."""
    return json.dumps([[[int(s), int(e)] for s, e in cluster] for cluster in clusters])


def _deserialize_clusters(payload) -> list[Cluster]:
    """JSON string -> clusters, tolerating a garbage value by returning [].

    A single unparseable row must not take down a whole run: the caller treats
    an empty cluster list as "coref produced nothing for this document", which
    is exactly the graceful-degradation behaviour the rest of the module has.
    """
    try:
        raw = json.loads(payload)
        return [[(int(s), int(e)) for s, e in cluster] for cluster in raw]
    except Exception:  # noqa: BLE001 - a corrupt row degrades, never raises
        return []


def load_cache(path=COREF_CACHE_PATH) -> pd.DataFrame:
    """Load the on-disk cluster cache, or an empty frame when absent.

    Never raises: a corrupt or schema-drifted file logs one warning and reads as
    empty, costing a cold pass. Laxer than sentiment.load_cache() on purpose, since
    this cache affects speed only and never output values.
    """
    if not path.exists():
        return pd.DataFrame(columns=CACHE_COLUMNS)
    try:
        df = pd.read_parquet(path)
        missing = set(CACHE_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"missing columns: {missing}")
        return df[CACHE_COLUMNS]
    except Exception as exc:  # noqa: BLE001 - unreadable cache == no cache
        logger.warning(
            f"Coreference cache at {path} is unreadable "
            f"({type(exc).__name__}: {exc}); treating it as empty"
        )
        return pd.DataFrame(columns=CACHE_COLUMNS)


def save_cache(cache_df: pd.DataFrame, path=COREF_CACHE_PATH) -> None:
    """Persist the cache, deduped on text_hash (keep last).

    Callers must pass the merge of the loaded cache and the new rows, never the new
    rows alone.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    deduped = cache_df.drop_duplicates(subset="text_hash", keep="last")[CACHE_COLUMNS]
    deduped.to_parquet(path, index=False)
    logger.info(f"Saved coreference cache with {len(deduped)} entries to {path}")


def resolve_documents(
    texts: list[str],
    batch_size: int | None = None,
    cache_path=COREF_CACHE_PATH,
    use_cache: bool = True,
) -> list[list[Cluster]]:
    """Resolve coreference over a batch of documents, with an on-disk cache.

    Returns one list of clusters per input document, in input order; each cluster is a
    list of (char_start, char_end) spans into that document's text.

    Only cache misses reach the model, duplicate texts collapse to one inference, and
    new rows are merged with the cache loaded at call start. Pass use_cache=False for
    a cold run.

    Batching uses fastcoref's predict(), which packs documents up to `batch_size`
    subword tokens per pass. Left as None it follows the device the model actually
    loaded onto -- COREF_BATCH_SIZE_GPU on a GPU, COREF_BATCH_SIZE on the CPU --
    since the batch a GPU can hold is far smaller than the one system RAM can.

    Never raises: on backend failure every document gets an empty cluster list.
    """
    if not texts:
        return []

    out: list[list[Cluster]] = [[] for _ in texts]

    # fastcoref chokes on empty/non-string inputs; keep the positional mapping
    # so the returned list still lines up 1:1 with `texts`.
    keep_idx = [i for i, t in enumerate(texts) if isinstance(t, str) and t.strip()]
    if not keep_idx:
        return out

    cache_df = load_cache(cache_path) if use_cache else pd.DataFrame(columns=CACHE_COLUMNS)
    cache_lookup: dict[str, str] = {}
    if len(cache_df):
        cache_lookup = {
            row.text_hash: row.clusters_json for row in cache_df.itertuples(index=False)
        }

    hashes = {i: hash_text(texts[i]) for i in keep_idx}
    # hash -> one representative text; duplicate documents are inferred once.
    to_run: dict[str, str] = {}
    n_hits = 0
    for i in keep_idx:
        h = hashes[i]
        if h in cache_lookup:
            out[i] = _deserialize_clusters(cache_lookup[h])
            n_hits += 1
        else:
            to_run.setdefault(h, texts[i])

    logger.info(
        f"resolve_documents: {len(keep_idx)} documents "
        f"({n_hits} cache hits, {len(to_run)} unique texts to resolve)"
    )

    if not to_run:
        return out

    model = _load_model()
    if model is None:
        return out

    run_hashes = list(to_run)
    payload = [to_run[h] for h in run_hashes]

    # Sized after the load, not before, so a GPU that fell back to the CPU gets
    # CPU-sized batches rather than needlessly small ones.
    if batch_size is None:
        batch_size = COREF_BATCH_SIZE_GPU if _DEVICE == "cuda" else COREF_BATCH_SIZE

    logger.info(
        f"Resolving coreference over {len(payload)} documents "
        f"(max {batch_size} tokens per batch, {_DEVICE or 'cpu'})"
    )
    try:
        preds = model.predict(texts=payload, max_tokens_in_batch=batch_size)
    except Exception as exc:  # noqa: BLE001 - inference failure must degrade
        logger.warning(
            f"Coreference inference failed ({type(exc).__name__}: {exc}); "
            "tagging will use explicit company names only"
        )
        return out

    resolved: dict[str, list[Cluster]] = {}
    n_clusters = 0
    for h, pred in zip(run_hashes, preds):
        clusters = [
            [(int(s), int(e)) for s, e in cluster]
            for cluster in pred.get_clusters(as_strings=False)
        ]
        resolved[h] = clusters
        n_clusters += len(clusters)

    for i in keep_idx:
        h = hashes[i]
        if h in resolved:
            out[i] = resolved[h]

    logger.info(f"Coreference produced {n_clusters} clusters across {len(payload)} documents")

    if use_cache and resolved:
        new_rows = pd.DataFrame(
            [
                {"text_hash": h, "clusters_json": _serialize_clusters(c)}
                for h, c in resolved.items()
            ],
            columns=CACHE_COLUMNS,
        )
        # Merge, never replace: `resolved` covers only this call's misses, so
        # saving it alone would drop every previously cached document.
        save_cache(pd.concat([cache_df, new_rows], ignore_index=True), path=cache_path)

    return out
