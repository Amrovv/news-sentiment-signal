"""Reconstruct notebook 2.4's 300-row blind-labelling sample and freeze it.

WHY THIS EXISTS. 2.4's sample is defined by strata that include
`disagree_comparative`, built from `is_comparative` -- a column deleted from the
sentence schema in commit b1fcc6f. The 900 hand labels in references/fusion-labels*.csv
are keyed by `sample_id` ALONE, so without the sample frame they are orphaned: the
labels survive but nothing says which sentence each one refers to.

PROVENANCE, established by search rather than assumed. 2.4 read data/sentences.parquet
when it still had 24 columns. The current file has 20. data/interim/sentences_before_2.3
.parquet also has is_comparative but is NOT the right snapshot -- it reproduces the
population size (13,947) yet gives 3,413 sign disagreements against 2.4's recorded
3,390, which perturbs the seeded permutation and yields a sample joining only 72/300.
The correct source is the git blob at commit 94228ec ("Regenerate the feature tables
from the Wave 4-6 pipeline"), which reproduces all three of 2.4 cell [5]'s printed
figures exactly: pop 13,947, disagree 3,390, |absa|<0.10 6,106.

    git show 94228ec:data/sentences.parquet > <path>

Replicates cells [5]-[7] verbatim. Asserts all four label joins (300/300/300/30) before
writing, so a wrong reconstruction fails loudly instead of silently producing a
plausible-looking sample. Output: references/fusion-sample-frozen.csv, committed.

DO NOT REDRAW THIS SAMPLE against the current schema. It would silently invalidate all
900 labels.
"""
import sys

import numpy as np
import pandas as pd

from stock_predictor.text.fusion import signed

SEED = 20260816
N_PER_STRATUM = 50
MAX_PER_ARTICLE = 2

CORPUS = sys.argv[1] if len(sys.argv) > 1 else "sentences_at_94228ec.parquet"
sentences = pd.read_parquet(CORPUS)

pop = sentences[
    sentences["mentions_target"]
    & ~sentences["is_boilerplate"]
    & sentences["pos"].notna()
    & sentences["absa_pos"].notna()
].copy()

pop["fin_raw"] = signed(pop["pos"], pop["neg"])
pop["absa_raw"] = signed(pop["absa_pos"], pop["absa_neg"])
pop["is_resolved"] = pop["resolved_by_coref"] | pop["resolved_by_anaphora"]

same_sign = np.sign(pop["fin_raw"]) == np.sign(pop["absa_raw"])
absa_flat = pop["absa_raw"].abs() < 0.10

print(f"population: {len(pop)}")
print(f"  sign disagreement : {int((~same_sign).sum())} ({100*(~same_sign).mean():.1f}%)")
print(f"  |absa| < 0.10     : {int(absa_flat.sum())} ({100*absa_flat.mean():.1f}%)")

stratum = pd.Series("", index=pop.index, dtype=object)


def assign(name, mask):
    global stratum
    stratum = stratum.where(~((stratum == "") & mask), name)


assign("disagree_comparative", (~same_sign) & pop["is_comparative"])
assign("disagree_plain", ~same_sign)
assign("absa_flat", absa_flat)
assign("resolved", pop["is_resolved"])
assign("agree_strong", same_sign & (pop["fin_raw"].abs() >= 0.5))
assign("agree_weak", same_sign)
pop["stratum"] = stratum

STRATA = [
    "disagree_comparative",
    "disagree_plain",
    "absa_flat",
    "resolved",
    "agree_strong",
    "agree_weak",
]

rng = np.random.default_rng(SEED)
picked = []
for name in STRATA:
    block = pop[pop["stratum"] == name]
    order = rng.permutation(len(block))
    per_article, chosen = {}, []
    for pos_i in order:
        if len(chosen) >= N_PER_STRATUM:
            break
        aid = block.iloc[pos_i]["article_id"]
        if per_article.get(aid, 0) >= MAX_PER_ARTICLE:
            continue
        per_article[aid] = per_article.get(aid, 0) + 1
        chosen.append(pos_i)
    picked.append(block.iloc[chosen])

sample = pd.concat(picked)
sample["sample_id"] = (
    sample["stratum"]
    + "-"
    + sample["article_id"].astype(str)
    + "-"
    + sample["sent_idx"].astype(str)
)
sample = sample.sort_values("sample_id").reset_index(drop=True)
print(f"\nsample: {len(sample)} rows, {sample['stratum'].nunique()} strata")
print(sample["stratum"].value_counts().to_string())

# Verify against the labels this sample must join to -- the whole point of freezing it.
labels = pd.read_csv("references/fusion-labels.csv")
joined = sample.merge(labels, on="sample_id", how="inner")
assert len(joined) == 300, f"join lost rows: {len(joined)} -- reconstruction is WRONG"
print(f"\njoin against references/fusion-labels.csv: {len(joined)}/300 OK")

for extra in ("fusion-labels-llm-a", "fusion-labels-llm-b"):
    n = len(sample.merge(pd.read_csv(f"references/{extra}.csv"), on="sample_id", how="inner"))
    assert n == 300, f"{extra} join lost rows: {n}"
    print(f"join against references/{extra}.csv: {n}/300 OK")

n_fine = len(sample.merge(pd.read_csv("references/fusion-fine-ranks.csv"), on="sample_id", how="inner"))
assert n_fine == 30, f"fine-ranks join lost rows: {n_fine}"
print(f"join against references/fusion-fine-ranks.csv: {n_fine}/30 OK")

KEEP = [
    "sample_id",
    "stratum",
    "article_id",
    "sent_idx",
    "text",
    "mentions_target",
    "mentions_ceo",
    "is_comparative",
    "target_is_subject",
    "resolved_by_anaphora",
    "resolved_by_coref",
    "is_boilerplate",
    "char_len",
    "anaphor_char_start",
    "anaphor_char_end",
    "pos",
    "neg",
    "neu",
    "absa_text",
    "absa_aspect",
    "absa_pos",
    "absa_neg",
    "absa_neu",
]
out = sample[KEEP]
out.to_csv("references/fusion-sample-frozen.csv", index=False)
print(f"\nwrote references/fusion-sample-frozen.csv  {out.shape}")
