"""Whole-corpus evaluation of the judge-gated pipeline.

Answers the question the branch exists to answer: across all 2,124 articles, what
is the referent error rate before and after the judge, where does the remaining
error sit, and what does gating do to article-level sentiment.

TWO KINDS OF NUMBER, KEPT SEPARATE THROUGHOUT, because conflating them is the
easiest way to overstate this result:

  * MEASURED corpus-wide -- accept/reject counts, retention, sentiment shifts.
    These are facts about all 2,124 articles.
  * PROJECTED -- error rates. Precision can only be known where there are hand
    labels, so corpus error is the measured accept counts multiplied by the
    per-channel precision measured on 270 labelled rows. Every projected figure
    is marked and carries the caveat.

THE VALIDITY CHECK THAT MATTERS. Accept RATE is measurable everywhere, unlike
precision. If the judge accepts span rows at 62% on the labelled sample and at
some very different rate corpus-wide, the labelled sample is not representative
and the projected error rates do not transfer. That comparison is the closest
thing to out-of-sample validation available without labelling more rows, so it
is reported first and prominently.
"""
import numpy as np
import pandas as pd

from stock_predictor.text.coref_eval import accept_only, load_eval_set, load_judge_cache, wilson_ci
from stock_predictor.text.fusion import provenance_channel

MODEL_ID, PROMPT = "qwen2.5-7b-instruct-q4km", "v3"
CHANNELS = ["surface", "coref_span", "coref_nospan"]

sent = pd.read_parquet("data/interim/full_run/sentences_scored.parquet")
sent["channel"] = provenance_channel(sent)
pop = sent[
    sent["mentions_target"].fillna(False) & ~sent["is_boilerplate"].fillna(False)
].copy()

cache = load_judge_cache()
mine = cache[(cache["model_id"] == MODEL_ID) & (cache["prompt_version"] == PROMPT)]
pop = pop.merge(
    mine[["article_id", "sent_idx", "answer"]], on=["article_id", "sent_idx"], how="left"
)
is_coref = pop["resolved_by_coref"].fillna(False)
pop["accepted"] = (~is_coref) | pop["answer"].map(lambda a: accept_only(a) if pd.notna(a) else False)

coref_pop = pop[is_coref]
unjudged = int(coref_pop["answer"].isna().sum())
print("=" * 78)
print("WHOLE-CORPUS RESULT -- 2,124 articles")
print("=" * 78)
print(f"target sentences (non-boilerplate) : {len(pop):,}")
print(f"coref sentences judged             : {len(coref_pop) - unjudged:,} of {len(coref_pop):,}"
      + (f"   ({unjudged} UNJUDGED -> dropped, fail-closed)" if unjudged else ""))
print(f"judge answers                      : {coref_pop['answer'].value_counts().to_dict()}")

# --- 1. validity check: do corpus accept rates match the labelled sample? ----
ev = load_eval_set()
ev = ev.merge(mine[["article_id", "sent_idx", "answer"]], on=["article_id", "sent_idx"], how="left")
ev["accepted"] = ev["answer"].map(lambda a: accept_only(a) if pd.notna(a) else False)

print("\n" + "-" * 78)
print("1. VALIDITY CHECK -- accept rate, labelled sample vs whole corpus [MEASURED]")
print("-" * 78)
print(f"{'channel':16s}{'labelled':>22s}{'corpus':>24s}{'delta':>10s}")
rates = {}
for ch, has_span in [("coref_span", True), ("coref_nospan", False)]:
    lab = ev[ev["has_span"] == has_span]
    cor = coref_pop[coref_pop["channel"] == ch]
    cor = cor[cor["answer"].notna()]
    lab_rate, cor_rate = lab["accepted"].mean(), cor["accepted"].mean()
    lo, hi = wilson_ci(int(lab["accepted"].sum()), len(lab))
    rates[ch] = {"lab": lab_rate, "cor": cor_rate}
    inside = "within CI" if lo <= cor_rate <= hi else "OUTSIDE CI"
    print(f"{ch:16s}{lab_rate:>10.1%} [{lo:.1%},{hi:.1%}]{cor_rate:>14.1%} (n={len(cor):,})"
          f"{(cor_rate - lab_rate) * 100:>+8.1f}pp  {inside}")

# --- 2. retention -----------------------------------------------------------
print("\n" + "-" * 78)
print("2. WHAT SURVIVES THE GATE [MEASURED]")
print("-" * 78)
print(f"{'channel':16s}{'before':>10s}{'after':>10s}{'retained':>11s}")
kept_n = {}
for ch in CHANNELS:
    sub = pop[pop["channel"] == ch]
    kept = int(sub["accepted"].sum())
    kept_n[ch] = kept
    print(f"{ch:16s}{len(sub):10,d}{kept:10,d}{kept/max(len(sub),1):11.1%}")
print(f"{'TOTAL':16s}{len(pop):10,d}{int(pop['accepted'].sum()):10,d}"
      f"{pop['accepted'].mean():11.1%}")

# --- 3. error, projected from the labelled precision ------------------------
print("\n" + "-" * 78)
print("3. REFERENT ERROR [PROJECTED from 270 labelled rows -- NOT measured corpus-wide]")
print("-" * 78)
prec = {}
for ch, has_span in [("coref_span", True), ("coref_nospan", False)]:
    lab = ev[ev["has_span"] == has_span]
    before = (lab["verdict"] == "other").mean()
    acc = lab[lab["accepted"]]
    after = (acc["verdict"] == "other").mean()
    lo, hi = wilson_ci(int((acc["verdict"] == "other").sum()), len(acc))
    prec[ch] = {"before": before, "after": after, "lo": lo, "hi": hi}
    print(f"{ch:16s} error {before:6.1%} -> {after:6.1%}   95% CI [{lo:.1%}, {hi:.1%}]")

wrong_before = sum(len(pop[pop["channel"] == ch]) * prec[ch]["before"] for ch in prec)
wrong_after = {ch: kept_n[ch] * prec[ch]["after"] for ch in prec}
W = sum(wrong_after.values())
total_after = int(pop["accepted"].sum())
print(f"\n  whole target set BEFORE : {wrong_before:7.0f} wrong of {len(pop):,} = {wrong_before/len(pop):.2%}")
print(f"  whole target set AFTER  : {W:7.0f} wrong of {total_after:,} = {W/total_after:.2%}")
print("\n  remaining error by channel:")
for ch in wrong_after:
    print(f"    {ch:14s}{wrong_after[ch]:7.0f}  ({wrong_after[ch]/W:5.1%})")

n_surface = kept_n["surface"]
print(f"\n  surface is UNMEASURED: {n_surface:,} rows = {n_surface/total_after:.0%} of survivors")
for e in [0.005, 0.01, 0.02]:
    sw = n_surface * e
    print(f"    if surface error = {e:4.1%} -> {sw:5.0f} wrong = {sw/(sw+W):4.0%} of ALL error, "
          f"overall {(sw+W)/total_after:.2%}")

# --- 4. article-level impact ------------------------------------------------
print("\n" + "-" * 78)
print("4. ARTICLE-LEVEL IMPACT [MEASURED]")
print("-" * 78)
ung = pd.read_parquet("data/interim/full_run/articles_ungated.parquet")
gat = pd.read_parquet("data/interim/full_run/articles_judge_gated.parquet")
col = "fus_conf_graft_mean"
j = ung[["article_id", col]].merge(gat[["article_id", col]], on="article_id",
                                   suffixes=("_ungated", "_gated"))
both = j.dropna()
flips = np.sign(both[f"{col}_ungated"]) != np.sign(both[f"{col}_gated"])
lost = j[j[f"{col}_ungated"].notna() & j[f"{col}_gated"].isna()]
print(f"  articles with a sentiment score, ungated : {j[f'{col}_ungated'].notna().sum():,}")
print(f"  articles with a sentiment score, gated   : {j[f'{col}_gated'].notna().sum():,}")
print(f"  articles that LOSE their score entirely  : {len(lost):,}")
print(f"  correlation ungated vs gated             : {both[f'{col}_ungated'].corr(both[f'{col}_gated']):.4f}")
print(f"  articles whose sentiment SIGN flips      : {int(flips.sum()):,} ({flips.mean():.1%})")
print(f"  mean |shift|                             : "
      f"{(both[f'{col}_gated'] - both[f'{col}_ungated']).abs().mean():.4f}")
