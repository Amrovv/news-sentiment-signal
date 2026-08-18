"""Stage 4 sensitivity: chunking, borderline rows, and statistical significance."""
import math
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact

ROOT = Path(r"D:\ML\stock-predictor")
df = pd.read_parquet(ROOT / "data/interim/maverick_agreement.parquet")
df["flag"] = ~df.apply(
    lambda r: bool(r["agree_span"]) if r["has_span"] else bool(r["agree_sentence"]), axis=1
)
df["err"] = df["verdict"] == "other"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def line(sub, name):
    n = len(sub)
    ne, no = int(sub["err"].sum()), int((~sub["err"]).sum())
    tp = int((sub["flag"] & sub["err"]).sum())
    fp = int((sub["flag"] & ~sub["err"]).sum())
    tn = int((~sub["flag"] & ~sub["err"]).sum())
    fn = int((~sub["flag"] & sub["err"]).sum())
    if ne == 0 or no == 0:
        print(f"{name:<46} n={n} (degenerate)")
        return
    p = fisher_exact([[tp, fp], [fn, tn]], alternative="greater")[1]
    lo, hi = wilson(tn, tn + fn)
    print(
        f"{name:<46} n={n:>3} err={ne:>3} | recall {100*tp/ne:>5.1f}% "
        f"cost {100*fp/no:>5.1f}% | purity {100*(no)/n:>5.1f}% -> "
        f"{100*tn/(tn+fn):>5.1f}% [{100*lo:.1f},{100*hi:.1f}] | Fisher p={p:.4f}"
    )


print("kept-population purity is the number that matters; baseline is 'accept everything'\n")
for nm, m in [
    ("full set", pd.Series(True, index=df.index)),
    ("excl. chunked articles", ~df["mv_chunked"]),
    ("chunked articles only", df["mv_chunked"]),
    ("excl. borderline rows", ~df["borderline"]),
    ("borderline counted as ERRORS", pd.Series(True, index=df.index)),
]:
    sub = df[m].copy()
    if nm == "borderline counted as ERRORS":
        sub["err"] = sub["err"] | sub["borderline"]
    print(f"--- {nm} ---")
    line(sub[sub["has_span"]], "  SPAN")
    line(sub[~sub["has_span"]], "  NO-SPAN")
    line(sub, "  ALL")
    print()

print("chunked articles:", int(df.groupby('article_id')['mv_chunked'].first().sum()),
      "of", df['article_id'].nunique(), "articles;", int(df['mv_chunked'].sum()), "of",
      len(df), "eval rows")

print("\n--- flag rate by article length decile (is the signal just length?) ---")
df["decile"] = pd.qcut(df["article_tokens"], 5, labels=False, duplicates="drop")
print(df.groupby("decile").agg(
    n=("flag", "size"), flag_rate=("flag", "mean"), err_rate=("err", "mean"),
    med_tokens=("article_tokens", "median")).round(3).to_string())
