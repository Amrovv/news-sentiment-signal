# Stage 4b — is Maverick *better* than fastcoref, or merely *different*?

**Date:** 2026-08-18 · **Branch:** `text/coref-verification`
**Follow-up to:** `notes/stage4-maverick-findings.md` §5 ("Open question worth one follow-up")
**Per-row outputs:**
- `data/interim/stage4b_backend_headtohead.parquet` — 270 rows, the labelled head-to-head
- `data/interim/stage4b_backend_swap_sentences.parquet` — 8,814 rows, every sentence of the 200 eval articles tagged twice (once per backend)
- `data/interim/stage4b_headtohead_summary.parquet` — the aggregate table below, machine-readable
- `data/interim/stage4b_maverick_only_frame.parquet` — 166 rows, **the unlabelled blind spot**, with a pre-drawn n=60 sample column

**Scripts:** `notes/stage4-scripts/stage4b_headtohead.py`, `notes/stage4-scripts/stage4b_bounds.py` (reuse
`notes/stage4-scripts/` + the cached Maverick clusters; **no new Maverick inference was run**).

Every number below is **measured** unless explicitly marked *(estimated)*.

---

## 1. Hypothesis

Stage 4 measured whether Maverick and fastcoref **agree**, and used disagreement as an error
filter. It did not ask whether Maverick is simply **better**. If it is, the correct move is to
**replace** the pipeline's coref backend — cheaper and structurally simpler than running two models
and ensembling.

**Claim under test.** Tagging sentences from Maverick's clusters alone is more accurate than
tagging them from fastcoref's clusters alone, against the same hand labels.

**Stated before looking at the numbers — what would count as a positive result:**
1. Maverick's precision exceeds fastcoref's on the labelled rows, *and*
2. the discordance is asymmetric in Maverick's favour (it fixes more than it breaks), *and*
3. the sentences Maverick tags that fastcoref does not are few enough, or good enough, that the
   swap does not silently import a new error population.

**(3) is the trap, and it is the whole difficulty of this task.** See §2.1.

---

## 2. Method

### 2.1 The methodological limitation — read this before any number below

The 270 labelled rows are **rows fastcoref selected**. Every one of them is, by construction, a
sentence fastcoref tagged as target. Therefore:

- **Measurable:** *Maverick's precision on fastcoref's picks* — of the sentences fastcoref claimed,
  how often Maverick's own judgement lines up with the human verdict.
- **NOT measurable from this sample:** *Maverick's recall* — there is no label for any sentence
  fastcoref never tagged, and Maverick tags some of those.

A bare "Maverick is 84% accurate, fastcoref is 78%" comparison on this sample is **not a
like-for-like backend comparison**. It is a comparison on a population one of the two contestants
chose. Reporting it as a backend benchmark would be exactly the error `HANDOFF-coref-verification.md`
§9 warns about ("do not benchmark a method against the thing you are testing") and that notebook 2.6
documents twice.

§4.3 does not merely disclaim this — it **quantifies** the blind spot and turns the comparison into
a **bound**.

### 2.2 Definition of "Maverick's own resolution" for a row

The decision Maverick would have made *as the pipeline's backend*, mirroring
`stage4-maverick-findings.md` §2.4 exactly. `M_mv` = every mention of every Maverick cluster that the
pipeline's own `entity_filter.map_coref_clusters()` keyed to `TARGET` (same alias patterns, same
PERSON spans, same parse — nothing ticker-specific added).

- **Span rows (`has_span == True`, n=100)** — the rows the pipeline physically rewrites.
  > **Maverick claims TARGET** ⇔ ∃ m ∈ `M_mv` overlapping the anaphor span `A`.

  i.e. *Maverick would license the same rewrite over the same characters*. This is the strict test,
  and it is the right one, because the rewrite is span-level.

  A **looser** reading is also reported (§4.1): Maverick tags the *sentence*, possibly keying a
  different anaphor. It moves nothing material (94.6% vs 94.5%).

- **No-span rows (`has_span == False`, n=170)** — the pipeline's claim here is only "some
  target-keyed coref mention lies inside this sentence", so the test matches that claim.
  > **Maverick claims TARGET** ⇔ ∃ m ∈ `M_mv` fully contained in the sentence window.

The two tests are **never pooled silently**; every table is split by `has_span`.

`fastcoref claims TARGET` is **True for all 270 rows** by construction — that is what makes them
eval rows, and it is the source of the limitation in §2.1.

### 2.3 The backend-swap measurement (the blind-spot quantifier)

To size what a swap actually changes, the 200 eval articles were re-tagged **twice through the
pipeline's own `tag_sentences()`** — once with fastcoref's mentions, once with Maverick's — over the
identical spaCy parse. Nothing was reimplemented; the same function the corpus was built with was
called with a different `coref_mentions` argument.

**Validation that this is faithful:** the fastcoref pass reproduces the shipped
`data/sentences.parquet` for these articles **exactly** — 8,814 sentences, 3,437 `mentions_target`,
1,369 `resolved_by_coref`, 712 of them with a span. All four match the shipped frame.

### 2.4 Statistics

`wilson_ci` is the exact definition from `notebooks/text/2.7-aw-coref-verification.ipynb` cell 6,
copied verbatim. All CIs are Wilson 95%. Discordance is tested with a **two-sided exact McNemar**
on the discordant pair — the correct test for two methods scored on the same rows.

---

## 3. What we did

1. Read `data/interim/maverick_agreement.parquet` (Stage 4's per-row output) and
   `data/eval/coref_eval_labelled.parquet`. Derived `mv_claims_target` per §2.2.
2. Computed precision for both backends on the 270 rows, split by `has_span` and with/without
   `borderline`, with Wilson CIs; computed the 2×2 discordance and exact McNemar.
3. Re-tagged all 8,814 sentences of the 200 articles under each backend via `tag_sentences()`,
   reusing the cached Maverick clusters from Stage 4 (`scratchpad/maverick_clusters.pkl`) — **no
   re-inference**, so no 5.5 s/article cost was paid.
4. Counted the three regions: both-tag, fastcoref-only, **Maverick-only**; split each by span vs
   no-span; turned the Maverick-only region into a bound on Maverick's true precision.
5. Persisted the Maverick-only region as a **sample frame** with a pre-drawn stratified n=60 sample
   (`sample_n60`, seed `20260818`, 30 span + 30 no-span, `row_id` = `MVONLY-n`). **No labelling was
   done here** — those rows have no ground truth and none was guessed.

**What did not need doing:** nothing was re-run through Maverick, nothing in `stock_predictor/` was
modified, nothing was committed, notebook 2.7 was not touched.

---

## 4. Analysis

### 4.1 Precision on the labelled rows — *fastcoref's picks only, see §2.1*

| population | n | errors | fastcoref precision | Maverick claims | Maverick precision |
|---|---:|---:|---|---:|---|
| **span (rewritten)** | 100 | 11 | 89/100 = **89.0%** [81.4, 93.7] | 91 | 86/91 = **94.5%** [87.8, 97.6] |
| **no-span (tagged only)** | 170 | 74 | 96/170 = **56.5%** [49.0, 63.7] | 121 | 76/121 = **62.8%** [53.9, 70.9] |
| all rows | 270 | 85 | 185/270 = **68.5%** [62.8, 73.8] | 212 | 162/212 = **76.4%** [70.3, 81.6] |
| span, excl. borderline | 99 | 11 | 88/99 = 88.9% [81.2, 93.7] | 91 | 86/91 = 94.5% [87.8, 97.6] |
| no-span, excl. borderline | 134 | 56 | 78/134 = 58.2% [49.7, 66.2] | 96 | 64/96 = 66.7% [56.8, 75.3] |
| all, excl. borderline | 233 | 67 | 166/233 = 71.2% [65.1, 76.7] | 187 | 150/187 = 80.2% [73.9, 85.3] |

Span rows under the **looser** definition (Maverick tags the sentence, span may differ):
87/92 = **94.6%** [87.9, 97.7] — indistinguishable from the strict test. The definition choice is
not load-bearing.

**Note the CIs overlap in every row.** Maverick is ahead everywhere, but nowhere by a margin this
sample resolves.

### 4.2 The head-to-head asymmetry — the actual answer to "is it better"

Same rows, both backends, 2×2. "fc right / mv wrong" = the human said `target` and Maverick would
**not** have tagged it (a recall loss). "fc wrong / mv right" = the human said `other` and Maverick
would **not** have tagged it (an error avoided).

| population | both right | both wrong | **fc right / mv wrong** | **fc wrong / mv right** | McNemar exact p |
|---|---:|---:|---:|---:|---:|
| **span** | 86 | 5 | **3** | **6** | 0.508 |
| **no-span** | 76 | 45 | **20** | **29** | 0.253 |
| all | 162 | 50 | **23** | **35** | **0.148** |
| all, excl. borderline | 150 | 37 | **16** | **30** | **0.054** |

**Errors avoided:** 35/85 = 41.2% [31.3, 51.8] overall; 6/11 = 54.5% [28.0, 78.7] on span rows;
29/74 = 39.2% [28.9, 50.6] on no-span rows.
**Correct rows dropped:** 23/185 = 12.4% [8.4, 18.0] overall; 3/89 = 3.4% [1.2, 9.4] span;
20/96 = 20.8% [13.9, 30.0] no-span.

These are numerically identical to Stage 4's disagreement-filter numbers — **necessarily so**, and
that identity is itself the finding. On a population fastcoref chose, "swap the backend" and "let
Maverick veto fastcoref" are *the same operation*. This sample cannot tell a better backend from a
conservative filter.

Two honest readings of the asymmetry:

- **Statistically:** 35 vs 23 is not significant (p = 0.148; p = 0.054 excluding borderline). Even
  taking the biased population at face value, Maverick's edge is a trend, not a result.
- **Under the project's loss function** ("losing a sentence is acceptable, corrupting one is not"),
  the two cells are not equal-weight. On span rows the trade is 6 corruptions avoided for 3
  sentences lost — good. On no-span rows it is 29 for 20 — with 45 errors surviving anyway.

### 4.3 The recall blind spot — **quantified, not just disclaimed**

Backend swap over the 200 eval articles, all 8,814 sentences (measured):

| region | sentences | span | no-span |
|---|---:|---:|---:|
| both backends coref-tag it (**labelled population**) | **1,101** | 607 | 494 |
| **fastcoref only** (lost on a swap) | **268** | 105 | 163 |
| **Maverick only** (gained on a swap — **no labels exist**) | **166** | 61 | 105 |
| fastcoref coref population (= shipped frame) | 1,369 | 712 | 657 |
| Maverick coref population | 1,267 | 674 | 593 |

**The blind spot is 166 sentences — 13.1% of Maverick's own coref output on these articles.**
Not 5, and not 5,000. This is the single most important number in this document.

Where both backends tag a span, they choose **the same anaphor start in 602 of 604 cases** — the two
models agree about *where* to rewrite almost perfectly; they differ about *whether* to tag at all.

**Corpus-wide projection *(estimated)*.** The 200 eval articles are coref-**enriched** by
construction (they hold 1,369 of the corpus's 3,617 coref rows — 37.8% of the coref population from
9.4% of articles), so per-article scaling would overstate everything ~4×. Scaling instead by the
ratio to fastcoref's own coref population, per stratum:

| | rate here | corpus-wide *(estimated)* |
|---|---|---:|
| Maverick-only, span | 61/712 = 8.6% | ~208 new rewritten sentences |
| Maverick-only, no-span | 105/657 = 16.0% | ~190 new tagged sentences |
| fastcoref-only, span (**lost**) | 105/712 = 14.7% | ~358 rewrites lost |
| fastcoref-only, no-span (**lost**) | 163/657 = 24.8% | ~295 tags lost |

So a backend swap is a **~400-sentence import of unmeasured content and a ~650-sentence deletion**
against a 3,617-row coref channel — roughly an 11% import and an 18% deletion. Material, but not
a rewrite of the corpus.

### 4.4 Turning the blind spot into a bound

Maverick's true precision as a backend is a weighted average of its precision on the **shared**
rows (measured) and on the **Maverick-only** rows (unknown, weight 13.1%). That bounds it:

Weighted to the **corpus** span/no-span mix (67.1% span, the weighting behind the handoff's 78.3%):

| | value |
|---|---|
| fastcoref, blended (reproduces the handoff's headline) | **78.3%** *(measured strata, weighted)* |
| Maverick on the shared rows, same weighting | **84.1%** *(measured strata, weighted)* |
| Maverick overall, **bounded** | **[73.1%, 86.2%]** |
| **break-even** — Maverick-only rows must be this accurate for Maverick to merely *match* fastcoref | **≥ 40.0%** |

Weighted to these 200 articles' own mix (52.0% span) the same computation gives fastcoref 73.4%,
Maverick-shared 80.3%, Maverick bounded **[69.8%, 82.9%]**, break-even **≥ 27.7%**.

**Read that break-even carefully.** It is a low bar — the Maverick-only rows would have to be worse
than the no-span channel's 56.5% and worse than a coin flip for the swap to be a net loss. But
"probably clears a low bar" is not a measurement, and the interval still straddles fastcoref's
78.3%. **The available data cannot settle the question; it can only narrow it to this bound.**

### 4.5 Caveats on the blind-spot number itself

- **57 of the 166 Maverick-only rows sit in the 13 chunked articles.** Chunking loses cross-block
  chains, which can only *reduce* Maverick's mentions — so 166 is a **lower bound** on the true
  Maverick-only count, and the fastcoref-only count (268) is correspondingly inflated.
- Single ticker (TSLA), OntoNotes model, one corpus. Same caveat as Stage 4.
- The 270 labels themselves carry the `borderline` ambiguity; §4.1/§4.2 report both ways.

---

## 5. Findings

**1. On fastcoref's own picks, Maverick is better — but not significantly, and not fairly.**
94.5% vs 89.0% on span rows, 62.8% vs 56.5% on no-span rows, 84.1% vs 78.3% corpus-weighted. Every
CI overlaps, and the head-to-head asymmetry (35 errors avoided vs 23 correct rows dropped) is not
significant (McNemar p = 0.148; p = 0.054 excluding borderline). **And the population was chosen by
fastcoref**, which is a structural advantage for its opponent: on this sample "Maverick as backend"
and "Maverick as veto" are arithmetically the same operation, so the sample cannot distinguish a
better model from a more conservative one.

**2. The recall blind spot is 166 sentences — 13.1% of Maverick's output — not a rounding error and
not a corpus rewrite.** *(Measured on 8,814 sentences.)* Corpus-wide that projects to ~400 imported
sentences and ~650 deleted ones against a 3,617-row channel *(estimated)*. This is the number that
tells you a backend swap is a **medium-sized, reviewable change**, not a leap of faith — and it is
small enough that a bound is computable rather than merely a disclaimer.

**3. That bound is [73.1%, 86.2%] against fastcoref's 78.3%, break-even at 40%.** The swap is
*probably* an improvement — the Maverick-only rows would have to be worse than a coin flip to make
it a loss, and Maverick's measured quality everywhere else is *better* than fastcoref's. But
"probably" is where two previous attempts in this workstream died. **Do not ship on it.**

**4. The two backends agree about *where*, not *whether*.** 602 of 604 shared span rows pick the
identical anaphor start. All the disagreement is in the tag/don't-tag decision. That is a useful
structural fact: a swap cannot change *what gets overwritten* in the shared rows, only *which rows
get overwritten at all* — which is why the rewrite-corruption risk of a swap is confined to the ~208
projected Maverick-only span rows.

**5. Stage 4's conclusion survives, narrowed.** Nothing here overturns "use disagreement as a
discard gate on span rows". What is new is that the *same* evidence is equally consistent with
"Maverick is the better backend", and 60 hand labels would tell them apart.

---

## 6. Recommendation

**Neither ship a swap nor abandon it. Label 60 rows first — the frame is already built.**

Ranked:

1. **Do now, zero new evidence needed:** keep Stage 4's span-row disagreement gate as the shippable
   result, and keep Stage 2 (provenance flag) — both are unaffected by this question.
2. **The one experiment that settles it:** hand-label the pre-drawn n=60 sample in
   `data/interim/stage4b_maverick_only_frame.parquet` (`sample_n60 == True`, 30 span + 30 no-span,
   seed `20260818`, `row_id` = `MVONLY-n`).
   - **What it decides:** whether Maverick-only rows clear the 40% break-even. At n=60 the Wilson
     half-width is ≈ ±12pp, so any observed rate ≥ 55% puts the CI lower bound above break-even and
     **settles the swap in Maverick's favour**; any rate ≤ 30% settles it against. Split 30/30 also
     gives a per-stratum read at ≈ ±17pp, enough to catch the case where the span rows (the
     dangerous ones) are clean but the no-span rows are junk.
   - **What it costs:** no inference (Maverick's clusters are cached), so it is labelling only —
     60 rows against full article context, the same protocol as
     `scratchpad/build_eval_frame_2.py` (seed, context sheet = headline + 4 preceding + target + 1
     following sentence). The n=120 widening in the Task C closeout took one session with two
     parallel Opus subagents plus hand review; **60 rows is roughly half that** *(estimated)*.
   - **Then re-run** `notes/stage4-scripts/stage4b_bounds.py` with the new labels folded in — the bound
     collapses to a point estimate and the recommendation follows mechanically.
3. **If and only if step 2 clears break-even:** swap the backend rather than ensemble. It is the
   cheaper outcome — one model instead of two, no gate to maintain — but it costs ~3.2 h CPU per
   cold corpus pass *(estimated, from Stage 4 §4.5)* versus fastcoref's existing cache, and it
   deletes ~650 currently-kept sentences.
4. **Do not** treat §4.1's precision numbers as a backend benchmark in any downstream document.
   They are precision **on fastcoref's picks**, and that caveat must travel with them.

**What would NOT settle it:** more labelled rows from the existing eval frame (they are all
fastcoref picks — more of them narrows nothing about recall); agreement statistics of any kind
(§4.2 shows they are the same arithmetic); running Maverick over more articles without labelling
(it grows the blind spot rather than measuring it).

---

### Not measured / caveats (consolidated)

- Maverick's recall is **not measured anywhere in this document** and cannot be, from this data.
- Single ticker (TSLA); OntoNotes-trained model; financial newswire is in-domain but nothing here
  generalises across tickers by measurement.
- 13 of 200 articles were chunked in Stage 4's run; 57 of the 166 Maverick-only rows come from them,
  so 166 is a lower bound (§4.5).
- Corpus projections in §4.3 and §6 are **estimated** by ratio-scaling, not measured.
- Nothing was committed, nothing in `stock_predictor/` was modified, notebook 2.7 was not touched,
  torch was not reinstalled (no inference ran at all).
