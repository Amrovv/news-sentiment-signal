"""Stage 4b part C: stratum weights + bounds on Maverick's true backend precision."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\ML\stock-predictor")
sys.path.insert(0, str(ROOT))


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


sw = pd.read_parquet(ROOT / "data/interim/stage4b_backend_swap_sentences.parquet")
h2 = pd.read_parquet(ROOT / "data/interim/stage4b_backend_headtohead.parquet")

sw["grp"] = np.select(
    [sw["fc_coref"] & sw["mv_coref"], sw["fc_coref"] & ~sw["mv_coref"],
     ~sw["fc_coref"] & sw["mv_coref"]],
    ["both", "fc_only", "mv_only"], default="neither",
)
print("=== population structure over the 200 eval articles (measured) ===")
tab = sw[sw["grp"] != "neither"].groupby(["grp"]).agg(
    n=("sent_idx", "size"),
    fc_span=("fc_has_span", "sum"),
    mv_span=("mv_has_span", "sum"),
)
print(tab.to_string())

fc_pop = sw[sw["fc_coref"]]
mv_pop = sw[sw["mv_coref"]]
print(f"\nfastcoref coref population: {len(fc_pop)}  "
      f"(span {int(fc_pop['fc_has_span'].sum())}, no-span {int((~fc_pop['fc_has_span']).sum())})")
print(f"maverick  coref population: {len(mv_pop)}  "
      f"(span {int(mv_pop['mv_has_span'].sum())}, no-span {int((~mv_pop['mv_has_span']).sum())})")

# ---- stratum precisions measured on the labelled rows ----
# fastcoref strata = its own span/no-span
fc_span_k = int(((h2["has_span"]) & h2["human_target"]).sum())
fc_span_n = int(h2["has_span"].sum())
fc_ns_k = int(((~h2["has_span"]) & h2["human_target"]).sum())
fc_ns_n = int((~h2["has_span"]).sum())
# maverick strata, restricted to rows it also claims (the SHARED population)
mv = h2[h2["mv_claims_target"]]
mv_span_k = int((mv["has_span"] & mv["human_target"]).sum())
mv_span_n = int(mv["has_span"].sum())
mv_ns_k = int(((~mv["has_span"]) & mv["human_target"]).sum())
mv_ns_n = int((~mv["has_span"]).sum())

print("\n=== stratum precisions (measured on hand labels) ===")
for lbl, k, n in [("fastcoref span", fc_span_k, fc_span_n),
                  ("fastcoref no-span", fc_ns_k, fc_ns_n),
                  ("maverick span (shared rows)", mv_span_k, mv_span_n),
                  ("maverick no-span (shared rows)", mv_ns_k, mv_ns_n)]:
    lo, hi = wilson_ci(k, n)
    print(f"  {lbl:32s} {k:3d}/{n:3d} = {k/n:6.1%}  [{lo:.1%}, {hi:.1%}]")

# ---- blended over the 200-article population ----
w_fc_span = int(fc_pop["fc_has_span"].sum()) / len(fc_pop)
fc_blend = w_fc_span * fc_span_k / fc_span_n + (1 - w_fc_span) * fc_ns_k / fc_ns_n
print(f"\nfastcoref blended precision over its {len(fc_pop)} rows "
      f"(w_span={w_fc_span:.3f}): {fc_blend:.1%}  [measured strata, weighted]")

shared = sw[sw["grp"] == "both"]
w_sh_span = int(shared["fc_has_span"].sum()) / len(shared)
mv_shared_prec = w_sh_span * mv_span_k / mv_span_n + (1 - w_sh_span) * mv_ns_k / mv_ns_n
w_shared = len(shared) / len(mv_pop)
lo = w_shared * mv_shared_prec
hi = w_shared * mv_shared_prec + (1 - w_shared)
print(f"maverick precision on the SHARED {len(shared)} rows (w_span={w_sh_span:.3f}): "
      f"{mv_shared_prec:.1%}  [measured strata, weighted]")
print(f"maverick's UNLABELLED share of its own output: {1-w_shared:.1%} "
      f"({len(mv_pop)-len(shared)} of {len(mv_pop)} rows)")
print(f"=> Maverick's true backend precision is BOUNDED to [{lo:.1%}, {hi:.1%}] "
      f"(worst case: every Maverick-only row wrong; best case: all right)")
print(f"   fastcoref's comparable figure: {fc_blend:.1%}")
print(f"   break-even: Maverick-only rows must be >= "
      f"{(fc_blend - lo) / (1 - w_shared):.1%} correct for Maverick to match fastcoref")

# ---- same comparison, but weighted to the CORPUS span/no-span mix (2428/1189) ----
W = 2428 / 3617
fc_corpus = W * fc_span_k / fc_span_n + (1 - W) * fc_ns_k / fc_ns_n
mv_corpus_shared = W * mv_span_k / mv_span_n + (1 - W) * mv_ns_k / mv_ns_n
lo2 = w_shared * mv_corpus_shared
hi2 = lo2 + (1 - w_shared)
print("\n=== same head-to-head, weighted to the CORPUS mix (span share 67.1%) ===")
print(f"  fastcoref blended precision (= handoff's 78.3%): {fc_corpus:.1%}")
print(f"  maverick on shared rows, corpus-weighted:       {mv_corpus_shared:.1%}")
print(f"  maverick bounded overall:                       [{lo2:.1%}, {hi2:.1%}]")
print(f"  break-even: Maverick-only rows must be >= {(fc_corpus - lo2) / (1 - w_shared):.1%} correct")

# corpus projection -- scaled by the RATIO to fastcoref's coref population, NOT by
# article count: the 200 eval articles are coref-enriched (they hold 1369 of the
# corpus's 3617 coref rows, 37.8%, from 9.4% of articles), so per-article scaling
# would overstate everything ~4x.
print("\n=== corpus projection (ESTIMATED, ratio-scaled off fastcoref's own population) ===")
for lbl, n_here, fc_here, corpus_n in [
    ("maverick-only, span", int(((sw['grp'] == 'mv_only') & sw['mv_has_span']).sum()),
     int(fc_pop['fc_has_span'].sum()), 2428),
    ("maverick-only, no-span", int(((sw['grp'] == 'mv_only') & ~sw['mv_has_span']).sum()),
     int((~fc_pop['fc_has_span']).sum()), 1189),
    ("fastcoref-only (lost), span", int(((sw['grp'] == 'fc_only') & sw['fc_has_span']).sum()),
     int(fc_pop['fc_has_span'].sum()), 2428),
    ("fastcoref-only (lost), no-span", int(((sw['grp'] == 'fc_only') & ~sw['fc_has_span']).sum()),
     int((~fc_pop['fc_has_span']).sum()), 1189),
]:
    r = n_here / fc_here
    print(f"  {lbl:32s} {n_here:4d}/{fc_here:4d} = {r:5.1%} -> ~{round(r * corpus_n):5d} corpus-wide")
