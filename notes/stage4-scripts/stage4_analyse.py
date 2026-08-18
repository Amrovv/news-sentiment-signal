"""Stage 4 analysis: does fastcoref/Maverick disagreement predict the labelled errors?"""
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\ML\stock-predictor")
df = pd.read_parquet(ROOT / "data/interim/maverick_agreement.parquet")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def fmt(k, n):
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100*k/n:.1f}% [{100*lo:.1f}, {100*hi:.1f}]"


def report(sub, name, flag_col):
    err = sub["verdict"] == "other"
    flag = sub[flag_col].astype(bool)
    tp = int((flag & err).sum())
    fp = int((flag & ~err).sum())
    fn = int((~flag & err).sum())
    tn = int((~flag & ~err).sum())
    n_err, n_ok = int(err.sum()), int((~err).sum())
    print(f"\n--- {name}  (n={len(sub)}, errors={n_err}, correct={n_ok}) ---")
    print(f"  flagged (discarded):        {tp+fp}")
    print(f"  recall on errors:           {fmt(tp, n_err)}")
    print(f"  cost: correct rows lost:    {fmt(fp, n_ok)}")
    print(f"  precision of the flag:      {fmt(tp, tp+fp)}")
    if tn + fn:
        print(f"  purity of what SURVIVES:    {fmt(tn, tn + fn)}  (baseline {fmt(n_ok, len(sub))})")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn)


print("=" * 78)
print("STAGE 4 -- fastcoref vs Maverick disagreement as an error predictor")
print("=" * 78)
print("\nmv_state distribution:")
print(pd.crosstab(df["mv_state"], [df["has_span"], df["verdict"]]))

span = df[df["has_span"]].copy()
nospan = df[~df["has_span"]].copy()

span["flag"] = ~span["agree_span"].astype(bool)
nospan["flag"] = ~nospan["agree_sentence"].astype(bool)

print("\n\n########## PRIMARY CRITERION ##########")
report(span, "SPAN rows (criterion: Maverick puts the rewritten anaphor in a target chain)", "flag")
report(nospan, "NO-SPAN rows (criterion: Maverick puts >=1 target-chain mention in the sentence)", "flag")

both = pd.concat([span, nospan])
report(both, "ALL rows (row-type-appropriate criterion)", "flag")

print("\n\n########## STRICTER: no_mention vs other_chain ##########")
for nm, sub in [("SPAN", span), ("NO-SPAN", nospan)]:
    s = sub.copy()
    s["flag"] = s["mv_state"] == "other_chain"
    report(s, f"{nm} -- flag only 'Maverick assigns it to a NON-target chain'", "flag")
    s2 = sub.copy()
    s2["flag"] = s2["mv_state"] == "no_mention"
    report(s2, f"{nm} -- flag only 'Maverick finds no mention there at all'", "flag")

print("\n\n########## THRESHOLD SWEEP on mv_support_frac ##########")
print("(flag row when mv_support_frac < t; t=0 flags nothing, t>1 flags everything)")
for nm, sub in [("SPAN", span), ("NO-SPAN", nospan), ("ALL", both)]:
    print(f"\n{nm}:")
    print(f"  {'t':>5} {'flagged':>8} {'err caught':>12} {'recall':>8} {'correct lost':>13} {'cost%':>7} {'prec':>7}")
    n_err = int((sub["verdict"] == "other").sum())
    n_ok = int((sub["verdict"] != "other").sum())
    for t in [0.0, 0.001, 0.25, 0.5, 0.75, 1.0, 1.001]:
        f = sub["mv_support_frac"].fillna(0.0) < t
        tp = int((f & (sub["verdict"] == "other")).sum())
        fp = int((f & (sub["verdict"] != "other")).sum())
        prec = f"{100*tp/(tp+fp):.1f}%" if tp + fp else "-"
        print(f"  {t:>5} {tp+fp:>8} {tp:>12} {100*tp/n_err:>7.1f}% {fp:>13} {100*fp/n_ok:>6.1f}% {prec:>7}")

print("\n\n########## BASELINES ##########")
for nm, sub in [("SPAN", span), ("NO-SPAN", nospan), ("ALL", both)]:
    n_err = int((sub["verdict"] == "other").sum())
    print(f"  {nm:8} accept-everything: keeps {len(sub)} rows, {n_err} of them wrong "
          f"-> purity {fmt(len(sub)-n_err, len(sub))}")
