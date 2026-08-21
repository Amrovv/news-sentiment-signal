# `references/`

Label output from the manual evaluation rounds, plus the protocol they were produced under. All of
it is tracked, none of it is regenerable, and none of it is read by the package: these files exist
so the notebooks' figures can be recomputed and checked.

## Reading a `sample_id`

Every label file keys on `sample_id`, and the format is self-describing:

```
disagree_plain-139582672-30
^stratum       ^article_id ^sent_idx
coref-136478880-13
```

So any label rejoins to `sentences_judged.parquet` on `(article_id, sent_idx)` without needing the
sheet it was labelled from. That was verified after the sheets were deleted: all 1,130 rows across
the five files still resolve against the current corpus.

## The fusion labelling round

Behind notebook `2.2`, which chooses the shipped per-sentence score. 300 sentences drawn from six
strata, each scored on a five-point scale for sentiment toward the target: +1.0 clearly good news,
+0.5 mildly good, 0.0 none, -0.5 mildly bad, -1.0 clearly bad.

- **`fusion-labels.csv`** (300 rows): **the authoritative set.** Labelled blind by the first
  annotator, who also designed the variants. Every headline figure in `2.2` sections 5 to 9 is
  measured against this file.
- **`fusion-labels-llm-a.csv`** and **`fusion-labels-llm-b.csv`** (300 rows each): two further
  annotators working from the protocol alone, blind to the model scores, to the labels above, and to
  the fact that scoring variants were being compared. They exist for section 10, which uses them to
  bound how subjective the task is. Both are LLM annotators, so they are not independent human
  validation, and nothing here establishes that any one set is more accurate than another. The
  numbers they support: the two agree with each other 83.3%, and with the authoritative set 69.2%
  even on the rows where they agree with each other.
- **`fusion-fine-ranks.csv`** (30 rows): finer-grained ranks on a subsample, used where the
  five-point scale could not separate two variants.
- **`fusion-labelling-protocol.md`**: the instructions all three annotators worked from. Section 10
  means nothing without it, since the claim being made is that two annotators reproduced the task
  from this document alone.

## The context audit

Behind notebook `2.1` section 3, the second of three attempts to audit what coreference resolved a
mention to.

- **`context-audit-labels-v2.csv`** (200 rows): the referent decisions, carrying `referent` as free
  text, `refers_to_tesla`, `substitution_span_ok` and a `note`. This is the round that scored
  coreference at 74% and was itself discarded once its masking bug was found, which `2.1` section 3
  describes.


