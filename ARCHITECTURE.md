# ARCHITECTURE — what is what in this repository

**Written:** 2026-08-19 · **Branch:** `text/coref-verification` (45 commits ahead of `main`)
**Purpose:** orientation for anyone — human or agent — picking this repo up cold. It says what each
file is, what the pipeline does in order, which artifacts are authoritative, and which parts of the
repository are stale, dead, or actively misleading.

Read §1 and §2 first. §7 is the list of things that will waste your time if you don't know them.

This file describes *structure and status*. It is not a results log (`FINDINGS.md`), not a task
handoff (`HANDOFF*.md`), and not a decision record — decisions live in the notebooks and in the long
comment blocks inside the modules, which are the real documentation of this codebase.

---

## 1. What the project is

Predict TSLA price movement from financial news sentiment. Two people work in one repo:

| owner | area | directories |
|---|---|---|
| Person A | text & deep learning | `stock_predictor/text/`, `notebooks/text/` |
| Person B | market data & modeling | `stock_predictor/market/`, `stock_predictor/modeling/`, `notebooks/market/` |

**Do not edit `stock_predictor/market/` — it belongs to Person B.**

The text side is mature and is where essentially all the work has gone. The modeling side is
scaffolding: `modeling/train.py`, `modeling/predict.py`, `features.py`, `dataset.py`, `plots.py` are
each ~30 lines of untouched cookiecutter `main()` stubs. **No price model has been trained.** Every
accuracy number anywhere in this repo is a *text* metric — referent accuracy, sentiment plausibility —
never a trading result.

`FINDINGS.md` states the calibration that matters: 53–57% out-of-sample directional accuracy would be
a good result; 70%+ almost certainly means leakage.

---

## 2. The text pipeline, in execution order

The canonical driver is `notes/stage4-scripts/run_full_pipeline.py`. It is resumable — each phase
skips if its output already exists — and it writes to `data/interim/full_run/`, **never** over
`data/sentences.parquet`.

```
data/processed/processed_articles.parquet          2,124 articles (505 MB, includes raw HTML)
  │
  ├─ entity_filter.process_articles()              split into sentences, tag entities,
  │                                                run coref, flag boilerplate
  │     └─> sentences_tagged.parquet               71,410 sentences × 12-col SENTENCE_COLUMNS
  │
  ├─ sentiment.score_sentence_table()              FinBERT  → pos/neg/neu     (cached)
  ├─ sentiment.score_headlines()                   FinBERT on headlines       (cached)
  ├─ absa.score_sentence_table()                   DeBERTa  → absa_pos/neg/neu (cached)
  │     └─> sentences_scored.parquet
  │
  ├─ coref_judge (Qwen2.5-7B GGUF)                 referent verification      (cached)
  │     └─> sentences_judged.parquet               ~5.4 s/row, ~5 h for the corpus — THE SLOW PART
  │
  └─ sentiment.aggregate_article_features()
     fusion.aggregate_fusion_features()
     fusion.aggregate_provenance_features()
        └─> articles_ungated.parquet / articles_judge_gated.parquet    2,124 × 60 cols
```

### The five scoring channels

A sentence enters the target population by one of four `provenance_channel` routes. After the judge
gate, 12,934 sentences carry a score:

| channel | rows | meaning |
|---|---:|---|
| `surface` | 11,052 | "Tesla" appears literally in the sentence |
| `coref_span` | 1,424 | coref resolved an anaphor; sentence was **rewritten** with "Tesla" substituted |
| `coref_nospan` | 458 | coref tagged the sentence; nothing rewritten |
| `anaphora` | 0 | legacy recency heuristic — `USE_ANAPHORA_FALLBACK = False`, contributes nothing |

**This split is the single most important fact about the pipeline.** The branch spent its whole life
improving the coref channels, and a full-output audit then found those channels cause ~1.7% of the
remaining damage while the never-audited `surface` channel causes ~98%.

---

## 3. Package modules — `stock_predictor/`

Sizes are lines. The text modules are heavily commented by design: the comments carry the measured
evidence for each decision and are the primary documentation. Do not strip them.

### Core text pipeline (all live)

| module | lines | what it does |
|---|---:|---|
| `text/entity_filter.py` | 1452 | Sentence splitting (spaCy), alias matching across tiers, subject detection from the parse, anaphora resolution, coref cluster mapping, boilerplate flagging. Emits the 12-column `SENTENCE_COLUMNS` schema. The largest and most intricate module. |
| `text/sentiment.py` | 849 | FinBERT scoring with on-disk caching, headline scoring, and `aggregate_article_features()` — the article-level feature table. Also `analyze()`, the single-article path for the (unbuilt) Streamlit demo. |
| `text/fusion.py` | 818 | Combines FinBERT and ABSA into one signed score per sentence. Nine variants; three promoted to article level. **This is where the shipped score is defined.** See §4. |
| `text/absa.py` | 691 | Aspect-based sentiment (DeBERTa). Builds `(text, aspect)` pairs, including grammatical inflection when substituting a company name over a pronoun. Cached. |
| `text/coref.py` | 314 | fastcoref wrapper with a cluster cache. Deterministic per document text. |
| `text/coref_judge.py` | 304 | Local Qwen2.5-7B-Instruct judge (llama.cpp, GBNF-constrained to `yes`/`no`/`unsure`) that verifies whether a coref-resolved sentence really is about the target. Prompt `v3`. |
| `text/coref_eval.py` | 544 | The harness that measures a judge against hand labels. Wilson CIs, `JudgeContext`, verdict caching. Gates `coref_judge`. |
| `config.py` | 332 | Single source of truth for every constant. Company alias tiers, model ids, cache paths, thresholds. Read this before adding any constant anywhere else. |

### Not live

| module | status |
|---|---|
| `text/company_registry.py` (522) | **Untracked, imported by nothing.** Builds a sourced alias table from a Finnhub/EDGAR exchange listing. Has no test file. Referenced only from notebook 2.7. Its artifact `data/external/company_registry.parquet` is also untracked. Its fate is an open decision — see §7. |
| `market/timestamp_alignment.py` (48) | Person B's. Working, small. Do not touch. |
| `modeling/train.py`, `modeling/predict.py`, `features.py`, `dataset.py`, `plots.py` | ~30 lines each, cookiecutter stubs, never implemented. |

### Import graph

Everything imports `config`. Beyond that the text layer is a shallow chain:

```
entity_filter  →  coref
sentiment      →  entity_filter, fusion
absa           →  entity_filter, sentiment
coref          →  sentiment (hash_text only)
fusion         →  config only          ← no dependencies on the rest of the text layer
coref_eval, coref_judge  →  config
```

`fusion.py` depending on nothing but `config` is deliberate and worth preserving: it makes the scoring
definition testable in isolation.

---

## 4. The shipped score — read this before touching `fusion.py`

The number the price model is meant to consume is a **signed scalar per sentence**, then aggregated per
article. All the conf-graft variants are one formula in one parameter:

```
sign(absa) * |fin| * (floor + (1 - floor) * |absa|)
```

where `fin = pos - neg` (FinBERT) and `absa = absa_pos - absa_neg` (ABSA).

| floor | variant | status |
|---:|---|---|
| 0.0 | `conf_graft` | kept — backs every measurement before 2026-08-19 |
| 0.5 | `conf_graft_soft` | kept |
| **0.7** | **`conf_graft_floor`** | **SHIPPED** — `fusion.CONF_FLOOR` |
| 1.0 | `sign_graft` | kept, superseded |

**Direction always comes from ABSA; magnitude always from FinBERT.** No floor value can change a row's
sign — verified across all 12,934 scored rows. That property is what lets referent-accuracy findings
survive a floor change.

`CONF_FLOOR = 0.7` was set on 2026-08-19 against 2,500 hand-labelled sentences. The full reasoning,
the measured table, the cost it incurs and one rejected alternative are in **notebook 2.7 §13** and in
the comment block above `CONF_FLOOR` in `fusion.py`. The one-line summary of the surprising part:
*ABSA is more confident on the rows it gets wrong than on the rows it gets right*, so the confidence
multiplier was penalising correct rows hardest.

`AGGREGATED_VARIANTS` promotes three variants × `FUSION_AGGREGATIONS` (mean, median, lead, top3_pos,
top3_neg, spread) = **18 `fus_*` columns**. The two older variants stay promoted deliberately so that
switching the shipped scoring does not invalidate the record it was chosen against.

---

## 5. Data — which artifact is authoritative

`data/interim/*.parquet` is gitignored. `data/` itself is largely untracked. Nothing here regenerates
for free: the corpus took a scrape, and the judged corpus took ~5 h of local LLM inference.

| path | rows | status |
|---|---:|---|
| `data/processed/processed_articles.parquet` | 2,124 | **Source of truth for articles.** 505 MB. |
| `data/sentences.parquet` | 71,410 × 20 | The committed sentence table. Verified byte-identical to a fresh regeneration. |
| `data/interim/full_run/sentences_judged.parquet` | 71,410 | **The current working corpus** — tagged + scored + judged. |
| `data/interim/full_run/articles_judge_gated.parquet` | 2,124 × 60 | Article features, judge gate applied. The intended model input. |
| `data/interim/full_run/articles_ungated.parquet` | 2,124 | Same without the gate — the comparison arm. |
| `data/interim/{finbert,absa,coref}_cache.parquet` | — | Model caches. Each spot-checked against fresh inference to ~1e-6. **Expensive to lose.** |
| `data/external/company_registry.parquet` | — | Untracked, orphaned with its module. |
| `data/interim/sentences_before_2.{1,3}.parquet` | — | Historical snapshots for before/after comparisons in notebooks 2.1 and 2.3. |

### Hand labels — `data/eval/`

These are irreplaceable human work. **Never regenerate, never overwrite.**

| file | rows | what |
|---|---:|---|
| `coref_eval_labelled.parquet` | 270 | Referent ground truth that gates the judge |
| `audit_{surface,coref_span,coref_nospan}_labelled.parquet` | 450 | Referent identity audit |
| `impact_{surface_a,surface_b,coref_span,coref_nospan}_labelled.parquet` | 600 | Score-damage audit |
| `full_audit_0{1..8}_labelled.parquet` | 4,000 | Full-corpus score audit — **but see §6** |

---

## 6. The full-corpus audit — status and the trap in it

The goal was to label all 12,934 scored sentences with a five-way verdict
(`correct` / `implausible` / `benign` / `minor` / `harmful`). **It stalled at 3,000 usable rows
(23.2%) and was stopped deliberately.**

**Usable:** sheets 01, 03, 04, 05, 07 (2,500 rows, scored at floor 0) + sheet 08 (500 rows, floor 0.7).

**Not usable, still on disk:**
- **Sheets 02 and 06** (`full_audit_02/06_labelled.parquet`) — labelled with a divergent rule that
  filed every `|cg| < 0.05` row as `benign` regardless of referent. 407 rows share one identical
  reason string. Per-sheet `correct%` across 01–07 is 84.4 / **57.6** / 88.6 / 90.6 / 87.0 / **48.2** /
  87.0 while mean `|cg|` per sheet is flat at 0.17–0.28 — the divergence is the labeller, not the
  content. **These two files are present and unmarked; exclude them explicitly in any analysis.**

**Why it was stopped.** Two attempts on a cheap model failed in two different ways. The first produced
verdicts from a keyword script rather than reading — 13 distinct reason strings over 3,000 rows, and a
three-keyword rule reproduces its verdicts at 96.2% agreement. The second, one agent per sheet with
anti-shortcut guards, returned `correct%` of 89.0 / — / 51.0 / 45.2 / 61.8 / 98.2 on six random slices
of the same corpus. Only sheet 08 passed QC.

**The lesson to carry forward: a 53-point spread in labeller judgement is an order of magnitude larger
than the effect being measured.** Both diversity gates were gamed — one sheet beat a
"≥100 distinct reasons" check with `"Row N ..."` templates, another capped its filler string at exactly
the 60-row limit. Any future pass needs a model that holds the verdict scheme *and* a calibration
anchor proving it does, before the labels are trusted.

Audit sheets themselves: `notes/full_01.txt` … `full_26.txt` (500 rows each, floor-0 scores) and
`notes/full_08_w70.txt` … `full_13_w70.txt` (the same rows re-rendered at floor 0.7). Join keys in
`data/interim/full_audit_frame.parquet`.

---

## 7. Traps — the things that will waste your time

1. **`data/eval/full_audit_02` and `_06` are contaminated but not marked.** See §6.
2. **Notebooks 2.0, 2.1 and 2.3 cannot be re-executed against the current package.** They read columns
   that were deleted (`mentions_other`, `is_comparative`, `other_source`, `other_key`, and both
   `excl_comp` families). They are a historical record, not runnable code. See
3. **Quote figures from notebook 2.7 §11–§13 only.** §1–§10 predate two labelling-convention changes
   made on 2026-08-18. The eval set went 85 → 84 → 63 errors across them.
4. **Convention: inverse instruments (TSLQ, TSLS) are NOT the target.** Their sentiment is sign-flipped,
   so accepting them inverts the signal. A company's own products, joint referents including the
   target, funds holding it, and generic statements inside a target article all *do* count as target.
5. **Background bash jobs get killed almost immediately.** Long runs must be launched in the foreground
   and allowed to time out into the background — that survived ~3.3 h. Directly backgrounded jobs died
   within seconds, repeatedly.
6. **`python` has no pandas. Use `.venv\Scripts\python.exe`.**
7. **The 90.0/68.8/83.1 fusion figures are stale** with respect to `conf_graft_floor` — they were
   computed at floor 0.
8. **Notebook 2.4's 300-row sample is built on a deleted column and can never be redrawn.** If
   consolidation ever touches it, freeze it to `references/` first.
9. **Don't benchmark against the thing under test, and don't re-label only the rows a model agreed
   with.** Both mistakes have already been made on this project.

---

## 8. Notebooks — `notebooks/text/`

proposing 8 → 4 notebooks, classifying every section LIVE / SUPERSEDED / DEAD / NEVER BUILT. It has
**not** been acted on.

| notebook | subject | runnable? |
|---|---|---|
| `1.0-aw-corpus` | Building the article corpus; EDA; limitations of the Finnhub feed | historical |
| `1.1-aw-scraper-probe` | Whether scraping full article HTML is viable | historical |
| `1.2-aw-clean-and-convert` | Source / time / body cleaning stages; the processed corpus | historical |
| `2.0-aw-entity-sentiment-pipeline` | **The pipeline of record**, in data-flow order: split → tag → resolve → substitute → score. Merged from the former 2.0 + 2.1 + 2.3 | **no, by design** — see below |
| `2.4-aw-scorer-selection` | **Which fused score ships**, and as which article columns. Merged from the former 2.4 + 2.5 | **no, by design** |
| `2.6-aw-auditing-the-inference` | Part I: which tagging mechanisms earn their keep. Part II: the Task C closeout that corrects Part I's coref figure. **The single authoritative statement of coref accuracy** | yes |
| `2.7-aw-coref-verification` | The four-stage repair programme (Stages 4/4b/4c, 3, 2, 1), convention changes, and §13 the magnitude weighting | yes |

**The 2.0/2.1/2.3 merge (2026-08-19).** Those three notebooks were a chronological build
log ("Wave 1" → "Wave 6") describing a pipeline that had since been cut down twice. They
are now one notebook organised by data flow. 140 cells → 93 (66 copied verbatim with
provenance in each cell's `merged_from` metadata, 27 new framing/banner cells). Nothing
was paraphrased and no stored output was dropped.

Cells that read deleted columns were **kept and banner-marked**, not removed — they are
the only surviving evidence for decisions that are still live. The merged notebook is
therefore deliberately non-executable and says so at the top. Recover an original with
`git show HEAD:notebooks/text/2.1-aw-entity-sentiment-improvements.ipynb`.

**The 2.4/2.5 merge.** One argument split across two notebooks — 2.4 chose a scorer and
changed no artifact, 2.5 turned it into features. Now `2.4-aw-scorer-selection`, 84 → 94
cells. Two fixes the proposal marked non-optional are applied: the sample cell carries a
DO-NOT-REDRAW banner pointing at the frozen CSV, and 2.4's `mean_blend` verdict — which
contradicted 2.5 and never shipped — is corrected in place with the original text left as
the record.

**The 2.6/2.7 split (§7.3).** 2.7's §1–§7 were a *correction to 2.6* living in a different
file. They now sit in 2.6 as Part II, directly after the audit they correct. 2.6 went
16 → 41 cells, 2.7 went 56 → 35 and is now purely the four-stage programme. Per risk R3,
coref accuracy is stated in **exactly one place** — 2.6 Part II §2–§3 — and 2.6 Part I's
superseded 93.5% carries a pointer there rather than its own restatement.

Cut in the 2.0 merge: 2.0's CEO ablation (wrong by ~3×, superseded by 2.1's), 2.0 §3.3/§3.4
(pure back-references), 2.1's before/after delta tables (they compare two schemas,
neither of which is current), the gold-set sections, and the full-corpus re-run cells.

`notebooks/market/1.0-kk-initial-data`, `1.1-kk-timestamp-alignment` — Person B's, small, self-contained.

---

## 9. Tests

`pytest` — **295 passing.** No test touches a network or a real model except where marked
`@pytest.mark.slow` (the marker is unregistered and warns; harmless).

| file | lines | covers |
|---|---:|---|
| `test_sentiment.py` | 945 | FinBERT scoring, caching, article aggregation |
| `test_fusion.py` | 926 | All nine variants, the aggregations, provenance |
| `test_entity_filter.py` | 897 | Tagging, anaphora, coref mapping, boilerplate |
| `test_absa.py` | 648 | Pair building, inflection, substitution guard |
| `test_coref_eval.py` | 395 | The judge harness, Wilson CIs, context rebuilding |
| `test_coref_judge.py` | 243 | Prompt construction, fail-closed behaviour, `n_ctx` |
| `test_data.py` | 45 | Corpus shape |

`company_registry.py` has **no test file** — one reason its fate is still open.

---

## 10. Supporting directories

| path | contents |
|---|---|
| `notes/` | `removed-mechanisms.md`, two Maverick/fastcoref comparison write-ups, `fusion-weight-sweep.txt`, and `stage4-scripts/` (17 scripts, including `run_full_pipeline.py` and `freeze_fusion_sample.py`). The generated audit sheets (`full_*.txt`, `audit_*.txt`, `impact_*.txt`) are gitignored — the labels derived from them live in `data/eval/`. |
| `references/` | Labelling protocols and the raw label CSVs from the fusion, context and inference audits. Tracked. |
| `reports/` | `1.2-processed-corpus.md` and `figures/` — five PNGs from notebooks 2.1–2.5. |
| `docs/`, `app/` | Empty (`.gitkeep` only). The Streamlit demo referenced by `sentiment.analyze()` was never built. |
| `models/` | Local model weights, gitignored. |

Root markdown: `README.md` (project layout), `FINDINGS.md` (results log, essentially empty),
`HANDOFF.md` (607 lines, the main handoff), `HANDOFF-coref-verification.md` (357),
`HANDOFF-full-corpus-audit.md` (192, the most recent).

---

## 11. Open items

Carried from the handoffs; none of these are done.

- **Rotate the Finnhub API key.** It was pasted in chat in an earlier session. Still outstanding.
- **Uncommitted work on this branch:** modifications to `fusion.py`, `sentiment.py`, `test_fusion.py`,
  `test_sentiment.py` (the `CONF_FLOOR` change), plus the untracked hand labels in `data/eval/` and
  the notes/sheets. Nothing has been staged.
- **Decide the fate of `company_registry.py`** + its parquet: adopt it (needs tests and an importer)
  or delete it.
- **The two highest-value fixes found by the audit, both non-ML, neither attempted:**
  1. a **publisher boilerplate blocklist** — template text with a filled-in variable
     (*"…has had 46 moves greater than 5% over the last year"*) evades `flag_boilerplate()`, which
     matches on exact text across ≥5 articles. Normalising numbers before counting would catch it.
     This is also the principled fix for the off-target regression `CONF_FLOOR = 0.7` introduces.
  2. a **negation / concession check** — ABSA reliably inverts on *"excluding Tesla…"* and
     *"…despite selling a fraction of the cars"*.

  Together these plausibly take the 5.51% harmful rate to ~2–3%, more error removed than the entire
  four-stage coref effort achieved.
- **No price model exists.** Until one is trained and evaluated, every quality claim in this repo is a
  proxy measured on hand labels, not an outcome.
