# Stage 4 — Maverick coref ensemble disagreement

**Date:** 2026-08-18 · **Branch:** `text/coref-verification`
**Spec:** `HANDOFF-coref-verification.md` §8.1
**Per-row output:** `data/interim/maverick_agreement.parquet` (270 rows, keyed `article_id`/`sent_idx`)
**Reproduction scripts:** `notes/stage4-scripts/` (`mvload.py` → `stage4_prep.py` → `stage4_run.py` → `stage4_analyse.py` → `stage4_sens.py`)

Everything below is **measured** on the 270-row hand-labelled eval set unless a
number is explicitly marked *(estimated)*.

---

## 1. Hypothesis

**Claim under test.** The pipeline's coref backend (fastcoref) exposes no
per-link confidence. If a *second, independently-trained* coref model is run
over the same articles, the sentences where the two backends **disagree about
whether the anaphor belongs to the target company's chain** should be enriched
for the resolutions the human auditor marked wrong. Disagreement would then act
as a free, ticker-agnostic confidence proxy: discard on disagreement, keep on
agreement.

**Why an independent backend and not LingMess.** LingMess shares authors and
lineage with fastcoref (fastcoref is distilled from it), so their errors
correlate and disagreement would under-fire. Maverick (Martinelli, Barba &
Navigli, ACL 2024, Sapienza NLP) is a different group, a different architecture
(mention-extraction + span clustering over a DeBERTa-v3-large encoder) and a
different training recipe.

**What would count as failure — stated before looking at the numbers:**

1. **Feasibility failure** — Maverick cannot run on this box (torch 2.13.0+cpu,
   CUDA unavailable, 16 GB RAM) without changing torch, *or* its mention spans
   cannot be aligned to fastcoref's character offsets.
2. **Signal failure** — the flag is not statistically distinguishable from
   flagging at random, i.e. the error rate among flagged rows is no higher than
   the base error rate (Fisher exact p > 0.05, one-sided).
3. **Economic failure** — the flag is significant but the purity of what
   survives does not rise above the accept-everything baseline by more than the
   confidence intervals allow, i.e. the discarded correct rows buy nothing.

Outcome: **(1) did not occur, (2) did not occur, (3) occurred for no-span rows
and did not occur for span rows.** Details in §4–5.

---

## 2. Method

### 2.1 Environment and versions (all measured)

| item | value |
|---|---|
| PyPI package | `maverick-coref` **1.0.7** — name confirmed, sdist-only |
| HF checkpoint | `sapienzanlp/maverick-mes-ontonotes` — confirmed, downloads `weights.ckpt` |
| torch | **2.13.0+cpu, unchanged** — `torch.cuda.is_available() == False` before and after |
| transformers | 5.14.1 (unchanged) · spaCy 3.8.14 (unchanged) · pandas 3.0.5 (unchanged) |
| newly installed | `maverick-coref` 1.0.7, `pytorch-lightning` 2.6.5, `torchmetrics` 1.9.0, `lightning-utilities` 0.15.3, `hydra-core` 1.3.5, `omegaconf` 2.3.1, `nltk` 3.10.3, `antlr4-python3-runtime` 4.9.3 |
| machine | i5-9600K 6-core, 16 GB, CPU-only |
| throughput | **~5.5 s/article** on CPU (measured over 143 articles, 707 s) |

Installed with `--no-deps` for the torch-touching packages (`pytorch-lightning`,
`torchmetrics`, `lightning-utilities`) precisely so pip could not resolve a new
torch. `pip freeze` before/after confirms torch, transformers, numpy and spaCy
are untouched.

### 2.2 Running Maverick

Input is the **same string the pipeline addresses**: `_fix_missing_space(processed_body)`
from `data/processed/processed_articles.parquet`. That string is parsed once with
the pipeline's own spaCy pipeline (`entity_filter._get_nlp()` →
`split_sentences(..., return_spans=True)`), which gives both the sentence
boundaries `sent_idx` refers to and the token char offsets.

Maverick is fed via its **`sentence_tokenized` input path** — a list of lists of
token strings — rather than raw text. This is the whole answer to the alignment
problem (§2.3).

fastcoref's clusters were **not re-inferred**: they are read straight out of the
pipeline's own on-disk cache `data/interim/coref_cache.parquet`, keyed by
`sentiment.hash_text(cleaned)`. All 200 eval articles were cache hits, so the
fastcoref side of the comparison is literally the clusters `data/sentences.parquet`
was built from — not a re-run that might have drifted.

### 2.3 Span alignment — the approach used

Maverick and fastcoref delimit mentions differently, and Maverick's own raw-text
path reconstructs char offsets with a fragile `off += len(sentence) + 1`
accumulation. Both problems were sidestepped rather than patched:

1. Tokenise the cleaned article with **spaCy** (the same parse the pipeline used),
   dropping whitespace tokens. For every token keep `(tok.idx, tok.idx + len(tok.text))`
   — exact char offsets into the cleaned string.
2. Hand Maverick those token strings as `sentence_tokenized`.
3. Maverick returns `clusters_token_offsets` as *inclusive* index pairs into the
   flattened token list. Map `(s, e) → (chars[s][0], chars[e][1])`.

Alignment is therefore **exact by construction** — the two backends' spans are
offsets into the identical string, and no fuzzy matching is involved. Verified on
3 articles by round-tripping every mention's surface form; verified globally by
the fact that all 270 eval rows' `text` matched the reconstructed sentence at
`sent_idx` (0 mismatches).

Maverick's clusters are then passed through **the pipeline's own
`entity_filter.map_coref_clusters()`**, with the same ticker alias patterns and
the same PERSON entity spans from the same parse. So both backends' clusters are
keyed to the target by identical, ticker-agnostic logic. Nothing company-specific
was added anywhere; the target ticker is read from the eval frame's `ticker`
column.

### 2.4 Definition of "disagreement"

Notation, all in absolute char offsets into the cleaned article text:

- `M_fc` = every mention of every fastcoref cluster that `map_coref_clusters`
  keyed to `TARGET`.
- `M_mv` = the same, computed from Maverick's clusters.
- sentence window = `[off, off + len(sent_text))` where `off` is the stripped
  sentence's start char (`span.start_char + leading_whitespace`), exactly as
  `tag_sentences()` computes it.
- overlap(a, b) ⇔ `a[0] < b[1] and b[0] < a[1]`.

**Span rows (`has_span == True`, n=100).** These are the rows the pipeline
physically rewrites, substituting the company name over a specific char span.
The anaphor's absolute span is `A = (off + anaphor_char_start, off + anaphor_char_end)`.

> **AGREE** ⇔ ∃ m ∈ `M_mv` with overlap(A, m).
> **DISAGREE** otherwise.

In words: *Maverick also places the exact text that is about to be overwritten
inside a chain that names the target somewhere.* This is a span-level test
because the dangerous act is span-level.

**No-span rows (`has_span == False`, n=170).** No anaphor span was ever recorded,
so there is nothing span-level to compare — the pipeline's claim for these rows
is only the weaker "some target-keyed coref mention is contained in this
sentence". The criterion is matched to that claim:

> **AGREE** ⇔ ∃ m ∈ `M_mv` fully contained in the sentence window.
> **DISAGREE** otherwise.

These are **different tests and are never pooled silently**; every table below is
split by `has_span`. The span test is strictly stricter than the sentence test.

**Tri-state refinement (`mv_state`), recorded for every row:**

- `support` — agree, per the row-appropriate criterion above;
- `other_chain` — disagree, but Maverick *does* have a mention there, in a chain
  that does not name the target (a positive contradiction);
- `no_mention` — disagree because Maverick found no mention there at all
  (an abstention, not a contradiction).

**Continuous variant (`mv_support_frac`).** For rows where fastcoref has ≥1
target mention contained in the sentence, the fraction of those mentions that
overlap some Maverick target mention. Ranges over [0, 1]; used for the threshold
sweep in §4.3.

### 2.5 Per-row artifact

`data/interim/maverick_agreement.parquet`, one row per eval row, columns:
`row_id, article_id, sent_idx, has_span, verdict, borderline, mv_ok,
article_tokens, mv_chunked, n_mv_clusters, n_mv_target_mentions,
n_fc_target_mentions, n_fc_in_sent, n_mv_target_in_sent, n_mv_any_in_sent,
mv_support_frac, agree_sentence, anaphor_abs_start, anaphor_abs_end,
anaphor_surface, agree_span, mv_covers_anaphor, mv_state`.

---

## 3. What we did — including what did not work

1. **Package name confirmed** (`maverick-coref`, versions 0.1.3 → 1.0.7).
   *Dead end 1:* `pip download` failed outright — the sdist's `setup.py` does
   `open("README.md").read()` with the platform default encoding, which is
   cp1252 on this box, and the README has non-Latin-1 bytes. Fixed by setting
   `PYTHONUTF8=1` for every pip invocation. There are no wheels, so this affects
   anyone installing it on Windows.
2. **Installed without touching torch.** Declared deps include
   `pytorch-lightning`, which would let pip re-resolve torch. Installed with
   `--no-deps` and added the transitive deps by hand; `pip freeze` diff confirms
   torch stayed at 2.13.0+cpu and CUDA is still unavailable.
3. *Dead end 2:* **checkpoint would not unpickle.** torch ≥ 2.6 defaults
   `weights_only=True` and PL passes it explicitly; the checkpoint embeds
   `omegaconf.DictConfig`. `torch.serialization.add_safe_globals` turned into an
   endless allowlist game (`DictConfig` → `typing.Any` → …). Resolved by patching
   `torch.load` **inside `lightning_fabric.utilities.cloud_io` only** to force
   `weights_only=False` (`notes/stage4-scripts/mvload.py`). Scoped to that module
   so nothing else in the process changes behaviour.
4. *Dead end 3:* **`RuntimeError: mat1 and mat2 must have the same dtype, but got
   Half and Float`.** The published checkpoint mixes an fp16 encoder with fp32
   heads; there is no CPU autocast to paper over it. Fixed with
   `mv.model = mv.model.float()`.
5. *Dead end 4:* **Maverick's raw-text path needs `nltk punkt_tab`** and
   reconstructs char offsets with an assumption about inter-sentence whitespace.
   Downloaded punkt for the smoke test, then abandoned that path entirely in
   favour of the `sentence_tokenized` path described in §2.3.
6. **Prototyped alignment on 3 articles** before scaling, per the spec. Clusters
   were legible and correct (`['Tesla ( TSLA +3.80% )', "the company's",
   "Tesla's", 'its', …]` as one chain, `['Rivian Automotive ( RIVN -0.69% )',
   'Rivian', 'it', 'its', …]` as another).
7. *Dead end 5 — the one that cost real time:* **the first full run stalled at
   article 61/200 at 9.3 GB RSS.** Maverick's mention scorer is O(n²) in
   subtokens and the corpus has a long tail — median 755 tokens, p99 8,535, max
   10,380. Killed it. Articles above **1,800 tokens** are now split into
   consecutive, **sentence-aligned** blocks of ≤1,200 tokens and the per-block
   clusters are unioned. That loses cross-block chains, which can only *add*
   disagreement, so every affected article is flagged `mv_chunked` and §4.4
   reports the numbers with and without them. 13 of 200 articles (37 of 270 eval
   rows) were chunked. Cached articles from the pre-chunking partial run that
   exceeded the threshold were discarded and re-run so every long article is
   produced the same way.
8. **Full run: 200 articles, 0 failures, ~5.5 s/article CPU.**

---

## 4. Analysis

Ground truth re-derived from `data/eval/coref_eval_labelled.parquet` as the spec
instructed — the handoff's "35 errors" is stale. Measured: **270 rows, 85 errors**
(74 no-span, 11 span), 200 distinct articles (the handoff's "131" also predates
the widened sample).

### 4.1 Where the disagreement lands

`mv_state` × `has_span` × `verdict` (measured counts):

| mv_state | no-span / error | no-span / correct | span / error | span / correct |
|---|---:|---:|---:|---:|
| `support` (agree) | 45 | 76 | 5 | 86 |
| `other_chain` (contradiction) | 29 | 18 | 5 | 2 |
| `no_mention` (abstention) | 0 | 2 | 1 | 1 |

Note `no_mention` is essentially empty (4 rows total). Disagreement, when it
happens, is almost always a *positive contradiction* — Maverick has a mention
right there and puts it in a chain that never names the target.

### 4.2 Headline numbers (primary criterion, §2.4)

95% CIs are Wilson. Fisher p is one-sided (flag enriched for errors).

| population | n | errors | **recall on errors** | **cost: correct rows lost** | flag precision | purity kept: baseline → filtered | Fisher p |
|---|---:|---:|---|---|---|---|---|
| **span (rewritten)** | 100 | 11 | **6/11 = 54.5%** [28.0, 78.7] | **3/89 = 3.4%** [1.2, 9.4] | 6/9 = 66.7% | 89.0% → **94.5%** [87.8, 97.6] | 0.0000 |
| **no-span (tagged only)** | 170 | 74 | **29/74 = 39.2%** [28.9, 50.6] | **20/96 = 20.8%** [13.9, 30.0] | 29/49 = 59.2% | 56.5% → **62.8%** [53.9, 70.9] | 0.0072 |
| all (row-appropriate) | 270 | 85 | 35/85 = 41.2% [31.3, 51.8] | 23/185 = 12.4% [8.4, 18.0] | 35/58 = 60.3% | 68.5% → **76.4%** [70.3, 81.6] | 0.0000 |

**Accept-everything baseline** (what the pipeline does today): keeps all 270
rows, 85 of them wrong — purity 68.5% [62.8, 73.8]; span-only 89.0% [81.4, 93.7];
no-span-only 56.5% [49.0, 63.7].

Read the two rows very differently:

- **Span rows.** 3.4% of correct rows bought a halving of the residual error
  rate (11/100 = 11.0% → 5/91 = 5.5%). The kept-purity CI [87.8, 97.6] still
  overlaps the baseline CI [81.4, 93.7], because n=11 errors is small — but the
  Fisher test on the 2×2 is decisive (p < 0.0001) and the trade is extremely
  cheap.
- **No-span rows.** Statistically real (p = 0.0072) but economically thin:
  20.8% of correct rows are destroyed to move purity 56.5% → 62.8%, and 45 of
  the 74 errors survive anyway. The kept population is still 37% wrong.

### 4.3 Threshold sweep on `mv_support_frac`

The signal is effectively **binary in practice** — it is flat across the interior
of [0, 1] because almost every row has exactly one fastcoref target mention in
the sentence, so the fraction is 0 or 1. Included for completeness; there is no
knob here worth tuning.

| t (flag if `mv_support_frac` < t) | span: flagged / recall / cost | no-span: flagged / recall / cost |
|---|---|---|
| 0.0 (flag nothing) | 0 / 0.0% / 0.0% | 0 / 0.0% / 0.0% |
| 0.001 | 8 / 54.5% / 2.2% | 50 / 40.5% / 20.8% |
| 0.25 | 8 / 54.5% / 2.2% | 50 / 40.5% / 20.8% |
| 0.50 | 8 / 54.5% / 2.2% | 50 / 40.5% / 20.8% |
| 0.75 | 10 / 54.5% / 4.5% | 52 / 40.5% / 22.9% |
| 1.00 | 10 / 54.5% / 4.5% | 52 / 40.5% / 22.9% |
| >1 (flag everything) | 100 / 100% / 100% | 170 / 100% / 100% |

### 4.4 Sensitivity — is the result an artifact?

| slice | span: recall / cost / purity→ | no-span: recall / cost / purity→ | span p | no-span p |
|---|---|---|---|---|
| full set (n=270) | 54.5% / 3.4% / 89.0→94.5% | 39.2% / 20.8% / 56.5→62.8% | 0.0000 | 0.0072 |
| **excl. chunked articles** (n=233) | 54.5% / 3.7% / 88.2→94.0% | 37.1% / 21.4% / 50.0→55.6% | 0.0000 | 0.0313 |
| chunked articles only (n=37) | degenerate (0 errors) | 75.0% / 19.2% / 86.7→95.5% | – | 0.0475 |
| excl. borderline rows (n=233) | 54.5% / 2.3% / 88.9→94.5% | 42.9% / 17.9% / 58.2→66.7% | 0.0000 | 0.0016 |
| borderline counted as errors | 58.3% / 2.3% / 88.0→94.5% | 38.0% / 17.9% / 45.9→52.9% | 0.0000 | 0.0030 |

The span result is stable to every slice. The no-span result survives all of them
but weakens to p = 0.031 once chunked articles are removed.

**Is the flag just a proxy for article length?** No. Flag rate by article-length
quintile is 0.211 / 0.176 / 0.143 / 0.321 / 0.226 — non-monotone and roughly flat,
against median token counts of 398 / 637 / 796 / 1,158 / 5,214.

### 4.5 Population projection *(estimated, not measured)*

From `HANDOFF-coref-verification.md` §5: 3,617 coref-resolved sentences, 2,428
with a span, 1,189 without. Applying the measured rates:

- Span channel: flag ≈ **9%** of 2,428 ≈ **220 sentences discarded**, of which
  ≈ 145 would have been correct; error count in the rewritten channel drops from
  ≈ 267 to ≈ 122. *(estimated by applying eval-set rates to the population; the
  eval set is a stratified random sample of that population, so this is a
  reasonable extrapolation but carries the CIs of §4.2.)*
- No-span channel: flag ≈ **29%** of 1,189 ≈ **345 sentences discarded**, ≈ 140
  of them correct, ≈ 205 errors removed of ≈ 517.

**Runtime cost *(estimated)*:** 2,124 corpus articles × ~5.5 s ≈ **3.2 hours CPU**
for a cold pass, cacheable exactly like the existing fastcoref cache. Peak RSS
stays under ~3 GB with the 1,800-token chunking guard in place; without it,
long articles exceed 9 GB and stall.

---

## 5. Findings

**1. Stage 4 is technically viable and cost nothing structural.** Maverick 1.0.7
with `sapienzanlp/maverick-mes-ontonotes` runs CPU-only on torch 2.13.0+cpu with
three small shims (UTF-8 pip, `weights_only=False` scoped to lightning's loader,
`.float()` cast). torch, transformers and spaCy are unchanged; FinBERT, ABSA and
fastcoref are untouched. Span alignment turned out **not** to be the hard problem
the handoff feared, because Maverick accepts pre-tokenised input and returns
token-index clusters — feeding it spaCy's tokens makes the mapping exact rather
than approximate.

**2. On span rows — the rows that get physically rewritten — the signal is worth
shipping.** 3.4% of correct rows buys 54.5% of the errors, halving the residual
error rate of the only channel that can turn a sentence into a false assertion.
Under the project's stated posture ("losing a sentence is acceptable, corrupting
one is not"), a 1-in-29 loss rate for a 1-in-2 corruption reduction is a good
trade — arguably the cheapest one available anywhere in this workstream. Caveat
to state plainly: this rests on **11 labelled span errors**. The direction is
solid (p < 0.0001) and stable across every sensitivity slice, but the *magnitude*
(54.5%, CI [28.0, 78.7]) is imprecise, and Stage 1's judge should be measured on
the same rows before either is committed to.

**3. On no-span rows — where 74 of the 85 errors actually live — the signal is
too weak to use on its own.** It is statistically real, and it is not noise: 39%
recall at 21% cost is far better than chance. But it discards one correct
sentence in five to leave a surviving population that is still **37% wrong**.
That does not fix the no-span channel; it makes a bad channel slightly less bad
while throwing away a fifth of its good rows. This is the honest negative half of
Stage 4 and it should not be talked up. **Recommendation: do not gate no-span
rows on Maverick disagreement alone.**

**4. The two backends fail differently, which is the encouraging part.** 34 of
the 85 errors sit in `other_chain` — Maverick actively puts the mention in a
chain that never names the target. Only 4 rows in the whole set are
`no_mention`. So Maverick is not merely abstaining where fastcoref is confident;
it is contradicting it, and the contradictions are enriched for real errors.
That is the ensemble premise holding up, and it is a reason to keep Maverick in
play as a *feature* for Stage 1's judge — e.g. as a cheap pre-filter that routes
only disagreeing rows to the expensive LLM, or as an input the judge sees.

**5. Where this leaves Stage 4's place in the pipeline.** It earns a **narrow**
place: as a discard gate on span rows only, and as a candidate pre-filter/feature
for Stage 1. It does not earn a place as a general coref gate, and it does not
substitute for the referent-verification judge. The unprimed Fable consultation's
framing survives this experiment intact — disagreement is a *confidence proxy*,
not a *referent verifier*, and the no-span numbers are what that distinction
costs.

**Open question worth one follow-up.** We tested whether Maverick *agrees* with
fastcoref. We did not test whether Maverick is simply *better* — i.e. whether
tagging from Maverick's clusters alone beats 78.3% blended. The per-row parquet
has everything needed (`n_mv_target_in_sent`) to answer that against the same 270
labels, and it is a cheap check now that the run is cached.

### Not measured / caveats

- Single ticker (TSLA). "Ticker-agnostic code is not the same claim as
  ticker-uniform accuracy" applies to Maverick too — it is an OntoNotes model,
  and OntoNotes is newswire, so financial-press prose is in-domain, but nothing
  here generalises across tickers by measurement.
- 13 articles were chunked; cross-block chains in those are lost. §4.4 shows the
  conclusions hold with them excluded.
- The eval set's `borderline` rows (8) are counted as `target` by default; §4.4
  shows both alternatives.
- Nothing was committed and nothing in `stock_predictor/` was modified.
