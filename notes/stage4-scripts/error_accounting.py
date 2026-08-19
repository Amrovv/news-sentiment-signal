"""Full error accounting over every Tesla-tagged sentence in the corpus.

Answers, for all 14,605 non-boilerplate target sentences across 2,124 articles:
how many are correctly tagged, how many are wrong, where the wrong ones are, and
how confident we can be in each figure.

THREE TIERS OF EVIDENCE, never blended:

  MEASURED     -- counted directly. The judge produced a verdict for every coref
                  sentence, so "how many did the judge reject" is a fact about
                  all 3,553, not an extrapolation.
  CALIBRATED   -- a measured count corrected by an accuracy rate learned from the
                  270 hand-labelled rows. The judge's rejections are not all
                  correct, so the true error count is the rejection count times
                  the rate at which rejections are right, plus the errors that
                  survived acceptance.
  UNMEASURED   -- the surface channel. Never judged, never audited. Reported as
                  a hole, with what it would cost to close.
"""
import numpy as np
import pandas as pd

from stock_predictor.text.coref_eval import accept_only, load_eval_set, load_judge_cache, wilson_ci
from stock_predictor.text.fusion import provenance_channel

MODEL_ID, PROMPT = "qwen2.5-7b-instruct-q4km", "v3"

sent = pd.read_parquet("data/interim/full_run/sentences_scored.parquet")
sent["channel"] = provenance_channel(sent)
pop = sent[sent["mentions_target"].fillna(False) & ~sent["is_boilerplate"].fillna(False)].copy()

cache = load_judge_cache()
mine = cache[(cache["model_id"] == MODEL_ID) & (cache["prompt_version"] == PROMPT)]
pop = pop.merge(mine[["article_id", "sent_idx", "answer"]], on=["article_id", "sent_idx"], how="left")

W = 78
print("=" * W)
print("ERROR ACCOUNTING -- every Tesla-tagged sentence, 2,124 articles")
print("=" * W)
print(f"\nnon-boilerplate target sentences : {len(pop):,}")
print(f"  judged by the LLM (coref)      : {int(pop['answer'].notna().sum()):,}")
print(f"  NOT judged (surface)           : {int(pop['answer'].isna().sum()):,}")

# ---------------------------------------------------------------- MEASURED --
print("\n" + "=" * W)
print("A. WHAT THE JUDGE FOUND -- MEASURED over all 3,553 coref sentences")
print("=" * W)
judged = pop[pop["answer"].notna()]
print(f"\n{'channel':16s}{'n':>7s}{'yes':>8s}{'no':>8s}{'unsure':>8s}{'flagged wrong':>15s}")
for ch in ["coref_span", "coref_nospan"]:
    sub = judged[judged["channel"] == ch]
    vc = sub["answer"].value_counts()
    flagged = len(sub) - int(vc.get("yes", 0))
    print(f"{ch:16s}{len(sub):7,d}{int(vc.get('yes',0)):8,d}{int(vc.get('no',0)):8,d}"
          f"{int(vc.get('unsure',0)):8,d}{flagged:10,d} ({flagged/len(sub):.1%})")
vc = judged["answer"].value_counts()
flagged_total = len(judged) - int(vc.get("yes", 0))
print(f"{'TOTAL':16s}{len(judged):7,d}{int(vc.get('yes',0)):8,d}{int(vc.get('no',0)):8,d}"
      f"{int(vc.get('unsure',0)):8,d}{flagged_total:10,d} ({flagged_total/len(judged):.1%})")

# -------------------------------------------------------------- CALIBRATED --
print("\n" + "=" * W)
print("B. HOW OFTEN THE JUDGE IS RIGHT -- from the 270 hand-labelled rows")
print("=" * W)
ev = load_eval_set().merge(mine[["article_id", "sent_idx", "answer"]],
                           on=["article_id", "sent_idx"], how="left")
ev["accepted"] = ev["answer"].map(lambda a: accept_only(a) if pd.notna(a) else False)
ev["truly_wrong"] = ev["verdict"] == "other"

cal = {}
print(f"\n{'channel':16s}{'reject precision':>20s}{'accept error':>16s}")
for ch, has_span in [("coref_span", True), ("coref_nospan", False)]:
    sub = ev[ev["has_span"] == has_span]
    rej, acc = sub[~sub["accepted"]], sub[sub["accepted"]]
    rp = rej["truly_wrong"].mean()          # of rows judge rejected, how many really were errors
    ae = acc["truly_wrong"].mean()          # of rows judge accepted, how many are still errors
    rp_lo, rp_hi = wilson_ci(int(rej["truly_wrong"].sum()), len(rej))
    ae_lo, ae_hi = wilson_ci(int(acc["truly_wrong"].sum()), len(acc))
    cal[ch] = {"rp": rp, "ae": ae, "rp_lo": rp_lo, "rp_hi": rp_hi, "ae_lo": ae_lo, "ae_hi": ae_hi}
    print(f"{ch:16s}{rp:>13.1%} (n={len(rej):3d}){ae:>11.1%} (n={len(acc):3d})")
print("\n  reject precision = of the sentences the judge threw out, how many were genuinely")
print("  mis-tagged. The rest are correct sentences lost -- the cost of the gate.")

# --------------------------------------------------- TOTAL ERROR ACCOUNTING --
print("\n" + "=" * W)
print("C. TOTAL ERROR -- CALIBRATED (measured counts x labelled accuracy rates)")
print("=" * W)
rows, tot_before, tot_after, tot_lost = [], 0.0, 0.0, 0.0
for ch in ["coref_span", "coref_nospan"]:
    sub = judged[judged["channel"] == ch]
    n = len(sub)
    n_rej = n - int((sub["answer"] == "yes").sum())
    n_acc = n - n_rej
    c = cal[ch]
    true_wrong_rejected = n_rej * c["rp"]          # errors correctly removed
    correct_lost = n_rej * (1 - c["rp"])           # correct sentences wrongly removed
    wrong_remaining = n_acc * c["ae"]              # errors that survived
    wrong_before = true_wrong_rejected + wrong_remaining
    rows.append((ch, n, wrong_before, n_acc, wrong_remaining, correct_lost))
    tot_before += wrong_before
    tot_after += wrong_remaining
    tot_lost += correct_lost

print(f"\n{'channel':16s}{'n':>7s}{'wrong BEFORE':>16s}{'kept':>8s}{'wrong AFTER':>15s}")
for ch, n, wb, na, wr, cl in rows:
    print(f"{ch:16s}{n:7,d}{wb:9.0f} ({wb/n:5.1%}){na:8,d}{wr:8.0f} ({wr/na:5.1%})")
n_coref = len(judged)
n_kept = sum(r[3] for r in rows)
print(f"{'CORAF TOTAL':16s}{n_coref:7,d}{tot_before:9.0f} ({tot_before/n_coref:5.1%})"
      f"{n_kept:8,d}{tot_after:8.0f} ({tot_after/n_kept:5.1%})")
print(f"\n  correct sentences lost to the gate: {tot_lost:.0f} "
      f"({tot_lost/n_coref:.1%} of the coref channel)")

n_surface = int((pop["channel"] == "surface").sum())
total_all = len(pop)
total_kept = n_surface + n_kept
print("\n" + "-" * W)
print("WHOLE TARGET SET (the number you asked for)")
print("-" * W)
print(f"  sentences before gate : {total_all:,}")
print(f"  sentences after gate  : {total_kept:,}  ({total_kept/total_all:.1%} retained)")
print(f"\n  KNOWN wrong before    : {tot_before:6.0f}  = {tot_before/total_all:.2%} of all target sentences")
print(f"  KNOWN wrong after     : {tot_after:6.0f}  = {tot_after/total_kept:.2%} of survivors")
print(f"  error reduction       : {1 - (tot_after/total_kept)/(tot_before/total_all):.0%}")
print(f"\n  ...but {n_surface:,} surface sentences ({n_surface/total_kept:.0%} of survivors) were")
print("  NEVER CHECKED, so the true totals above are lower bounds.")

# ------------------------------------------------------------------- WHERE --
print("\n" + "=" * W)
print("D. WHERE THE ERRORS ARE -- MEASURED (judge rejections, all 3,553)")
print("=" * W)
judged = judged.copy()
judged["rejected"] = judged["answer"] != "yes"
per_art = judged.groupby("article_id").agg(n=("rejected", "size"), bad=("rejected", "sum"))
print(f"\n  articles containing coref sentences      : {len(per_art):,} of {pop['article_id'].nunique():,}")
print(f"  articles with >=1 rejected sentence       : {int((per_art['bad'] > 0).sum()):,}")
print(f"  articles where EVERY coref sentence fails : {int((per_art['bad'] == per_art['n']).sum()):,}")
print(f"  median rejected per affected article      : {per_art[per_art['bad']>0]['bad'].median():.0f}")

print("\n  worst 10 articles by rejected-sentence count:")
worst = per_art.sort_values("bad", ascending=False).head(10)
heads = sent.drop_duplicates("article_id").set_index("article_id")
for aid, r in worst.iterrows():
    h = str(heads.loc[aid]["text"])[:60] if aid in heads.index else ""
    print(f"    {aid}  {int(r['bad']):3d}/{int(r['n']):3d} rejected   {h}")

print("\n  error CONCENTRATION (are errors clustered or spread?):")
bad_sorted = per_art["bad"].sort_values(ascending=False)
tot_bad = bad_sorted.sum()
for frac in [0.05, 0.10, 0.25]:
    k = max(1, int(len(bad_sorted) * frac))
    print(f"    top {frac:.0%} of articles ({k:,}) hold {bad_sorted.head(k).sum()/tot_bad:.0%} of all rejections")

print("\n  referent classes among KNOWN errors (from the 270 labelled rows only):")
lab_err = ev[ev["truly_wrong"]]
print(f"    {len(lab_err)} labelled errors, {lab_err['referent'].nunique()} distinct referents")
for ref, n in lab_err["referent"].value_counts().head(8).items():
    print(f"      {n:2d}  {ref}")
