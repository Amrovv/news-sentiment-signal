"""Load Maverick on this environment (torch 2.13 cpu, transformers 5.x, PL 2.6).

Two shims are needed:
  * torch>=2.6 defaults weights_only=True and PL passes it explicitly, so the
    omegaconf-bearing checkpoint refuses to unpickle. We patch torch.load inside
    lightning_fabric.utilities.cloud_io to force weights_only=False. The
    checkpoint comes from the SapienzaNLP HF org, i.e. a trusted source.
"""
import functools

import torch
import lightning_fabric.utilities.cloud_io as _cio

_orig_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)


_cio.torch.load = _patched_load


def load_maverick(device="cpu", hf="sapienzanlp/maverick-mes-ontonotes"):
    from maverick import Maverick

    mv = Maverick(hf_name_or_path=hf, device=device)
    # The published checkpoint mixes fp16 encoder weights with fp32 heads, which
    # raises "mat1 and mat2 must have the same dtype" on a CPU forward pass
    # (no autocast on CPU). Cast the whole thing to fp32 — CPU has no fp16
    # kernels worth having anyway.
    mv.model = mv.model.float()
    return mv


if __name__ == "__main__":
    m = load_maverick()
    print("loaded ok")
    txt = (
        "Tesla reported record deliveries. The company said it expects growth. "
        "BYD also grew, and it overtook rivals in Europe."
    )
    r = m.predict(txt)
    print("token clusters:", r["clusters_token_text"])
    print("char offsets:", r["clusters_char_offsets"])
    for cl in r["clusters_char_offsets"] or []:
        print([txt[s : e + 1] for s, e in cl])
