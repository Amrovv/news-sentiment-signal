"""Collapse the Stage 4b bound now that the 166-row blind spot has a labelled sample.

Everything is computed on the SAME denominator -- the 200 eval articles -- so the two
backends are compared on one population rather than mixing corpus-level and
article-level figures.
"""
import numpy as np
import pandas as pd


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


swap = pd.read_parquet("data/interim/stage4b_backend_swap_sentences.parquet")
h2h = pd.read_parquet("data/interim/stage4b_backend_headtohead.parquet")
mvo = pd.read_parquet("data/eval/mvonly_eval_labelled.parquet")

# --- population structure of each backend's output over the 200 articles ---
fc = swap[swap["fc_coref"]]
mv = swap[swap["mv_coref"]]
shared = swap[swap["fc_coref"] & swap["mv_coref"]]
mv_only = swap[~swap["fc_coref"] & swap["mv_coref"]]

print("POPULATIONS (200 eval articles)")
print(f"  fastcoref output : {len(fc):5d}  ({int(fc['fc_has_span'].sum())} span / {int((~fc['fc_has_span']).sum())} no-span)")
print(f"  Maverick output  : {len(mv):5d}  ({int(mv['mv_has_span'].sum())} span / {int((~mv['mv_has_span']).sum())} no-span)")
print(f"    shared         : {len(shared):5d}")
print(f"    Maverick-only  : {len(mv_only):5d}  ({int(mv_only['mv_has_span'].sum())} span / {int((~mv_only['mv_has_span']).sum())} no-span)")
print()

# --- accuracy rates, each measured on its own labelled sample ---
print("MEASURED ACCURACY RATES")
rates = {}
for name, sub, mask in [
    ("fastcoref, span", h2h[h2h["has_span"]], None),
    ("fastcoref, no-span", h2h[~h2h["has_span"]], None),
]:
    k, n = int((sub["verdict"] == "target").sum()), len(sub)
    lo, hi = wilson_ci(k, n)
    rates[name] = k / n
    print(f"  {name:24s} {k:3d}/{n:3d} = {k/n:6.1%}  95% CI [{lo:.1%}, {hi:.1%}]")

# Maverick on the SHARED rows: eval rows where Maverick also claims target.
for label, sub in [("span", h2h[h2h["has_span"]]), ("no-span", h2h[~h2h["has_span"]])]:
    claimed = sub[sub["mv_claims_target"]]
    k, n = int(claimed["human_target"].sum()), len(claimed)
    lo, hi = wilson_ci(k, n)
    rates[f"maverick shared, {label}"] = k / n
    print(f"  {'maverick shared, ' + label:24s} {k:3d}/{n:3d} = {k/n:6.1%}  95% CI [{lo:.1%}, {hi:.1%}]")

# Maverick-only rows: NEWLY LABELLED, this is what collapses the bound.
for label, has_span in [("span", True), ("no-span", False)]:
    sub = mvo[mvo["has_span"] == has_span]
    k, n = int((sub["verdict"] == "target").sum()), len(sub)
    lo, hi = wilson_ci(k, n)
    rates[f"maverick-only, {label}"] = k / n
    print(f"  {'maverick-only, ' + label:24s} {k:3d}/{n:3d} = {k/n:6.1%}  95% CI [{lo:.1%}, {hi:.1%}]  <- NEW")
k, n = int((mvo["verdict"] == "target").sum()), len(mvo)
lo, hi = wilson_ci(k, n)
print(f"  {'maverick-only, pooled':24s} {k:3d}/{n:3d} = {k/n:6.1%}  95% CI [{lo:.1%}, {hi:.1%}]  <- NEW (unweighted)")
print()


def weighted(pairs):
    num = sum(w * r for w, r in pairs)
    return num / sum(w for w, _ in pairs)


# --- backend-level precision, population-weighted over the 200 articles ---
fc_span_n = int(fc["fc_has_span"].sum())
fc_nospan_n = len(fc) - fc_span_n
fc_prec = weighted([(fc_span_n, rates["fastcoref, span"]),
                    (fc_nospan_n, rates["fastcoref, no-span"])])

sh_span_n = int(shared["mv_has_span"].sum())
sh_nospan_n = len(shared) - sh_span_n
mvo_span_n = int(mv_only["mv_has_span"].sum())
mvo_nospan_n = len(mv_only) - mvo_span_n

mv_prec = weighted([
    (sh_span_n, rates["maverick shared, span"]),
    (sh_nospan_n, rates["maverick shared, no-span"]),
    (mvo_span_n, rates["maverick-only, span"]),
    (mvo_nospan_n, rates["maverick-only, no-span"]),
])

print("BACKEND-LEVEL PRECISION, population-weighted over the 200 eval articles")
print(f"  fastcoref : {fc_prec:.1%}   (n={len(fc)})")
print(f"  Maverick  : {mv_prec:.1%}   (n={len(mv)})")
print(f"  difference: {(mv_prec - fc_prec) * 100:+.1f}pp")
print()

# What the bound looked like BEFORE these labels, vs now.
mv_known = weighted([(sh_span_n, rates["maverick shared, span"]),
                     (sh_nospan_n, rates["maverick shared, no-span"])])
share_known = len(shared) / len(mv)
share_blind = len(mv_only) / len(mv)
print("THE BOUND, BEFORE AND AFTER")
print(f"  known part : {share_known:.1%} of Maverick's output at {mv_known:.1%}")
print(f"  blind spot : {share_blind:.1%} of Maverick's output")
print(f"  before     : [{share_known * mv_known:.1%}, {share_known * mv_known + share_blind:.1%}]  (blind spot 0%..100%)")
print(f"  after      : {mv_prec:.1%}  (blind spot measured at {rates['maverick-only, span'] * mvo_span_n / len(mv_only) + rates['maverick-only, no-span'] * mvo_nospan_n / len(mv_only):.1%} population-weighted)")
break_even = (fc_prec - share_known * mv_known) / share_blind
print(f"  break-even : blind spot needed >= {break_even:.1%} for the swap to win")
print()

# Net corpus effect of a swap, on the 200 articles.
fc_only = swap[swap["fc_coref"] & ~swap["mv_coref"]]
print("NET EFFECT OF A SWAP (200 articles)")
print(f"  sentences imported : {len(mv_only):4d} at {mvo['verdict'].eq('target').mean():.1%} correct (measured)")
print(f"  sentences dropped  : {len(fc_only):4d}")
print(f"  net channel size   : {len(fc):4d} -> {len(mv):4d}  ({len(mv) - len(fc):+d})")
