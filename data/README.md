# `data/`

What lives here, what is tracked, and what can be rebuilt.

Most of this directory is ignored by git. Only the market source tables, the labelled eval set and
the final model tables are committed; everything else is derived and is rebuilt by running the
pipeline. The sizes below are the current working copy, not what a fresh clone gets.

## `raw/` — the market source pull, tracked

The market layer's inputs, kept because they pin the exact rows the market notebooks were run
against. The article pull is not tracked: the per-ticker raw pulls and the large processed corpus are
regenerable (see `interim/`), so `raw/` holds only the small market tables.

- **`raw_ohlcv.parquet`**: daily OHLCV bars for SPY and the four tickers.
- **`raw_schedule.parquet`**: NYSE market open and close times.
- **`{TICKER}_raw_earnings.parquet`** (AAPL, AMZN, NVDA, TSLA): EPS estimate, reported EPS and
  surprise, by earnings date.

Re-fetchable with `python -m news_sentiment.market.prices`, though a re-pull may not return the same
window.

## `eval/` — labelled ground truth, tracked

- **`coref_eval_labelled.parquet`** (80 KB, 270 rows): the referent ground truth that gates the
  coreference judge, 100 rows with a mention span and 170 without. Loaded through
  `config.COREF_EVAL_PATH`, read by `coref_eval.load_eval_set()`, and asserted by two tests. It is
  the set behind the 90.0% span and 68.8% no-span figures quoted throughout the notebooks.

Nothing in the package regenerates it. Four further labelling rounds once lived here and were
deleted on 2026-08-21: a per-channel referent audit, a per-channel score-impact audit, a
full-corpus score audit and a sample from the withdrawn Maverick trial. They were model-labelled
calibration checks rather than curated ground truth, and their figures are recorded in
`notebooks/text/2.1` section 5 and `notebooks/text/2.2` section 12.

## `interim/` — working artifacts, ignored

Nothing here is tracked. All of it is rebuildable, but not all of it is cheap.

- **`processed_articles.parquet`** (506 MB): the cleaned article table the pipeline consumes, one
  row per article with `processed_body`. Written by notebook 1.2 and addressed through
  `config.PROCESSED_ARTICLES_PATH`. It sits here rather than in `processed/` because it is a large
  regenerable intermediate rather than a deliverable.
- **The four model caches** (2.6 MB total), each keyed so that re-running costs only what actually
  changed:
  - `coref_judge_cache.parquet` (20 KB) holds 3,553 judge verdicts keyed on
    `(article_id, sent_idx, target, model_id, prompt_version)`. It is the smallest file here and
    represents roughly five hours of local LLM inference. **Back it up before any cleanup**; a
    `git clean -xfd` deletes it along with everything else in this directory.
  - `coref_cache.parquet` (1.2 MB), `finbert_cache.parquet` (764 KB) and `absa_cache.parquet`
    (768 KB) are keyed by content hash and cost roughly nine, five and seven minutes respectively.
- **`full_run/`** (21 MB): the pipeline's phase outputs, each file the previous one plus the
  columns its phase adds.
  - `sentences_tagged.parquet` — 71,410 sentences over the 10-column `SENTENCE_COLUMNS` schema,
    Phase A. A resume point: the driver skips Phase A when it exists.
  - `sentences_scored.parquet` — the same rows over 18 columns, adding FinBERT and ABSA scores.
    The second resume point.
  - `sentences_judged.parquet` — 21 columns, adding `provenance_channel`, `judge_answer` and
    `judge_accepted`.
  - `articles_judge_gated.parquet` — 2,124 articles over 47 columns, aggregated over accepted
    sentences only.
  - `articles_ungated.parquet` — the same table with the gate not applied, kept as the comparison
    baseline. It is the only file here that answers a question the others cannot: what the gate
    actually cost.

## `processed/pipeline_run/` — the deliverable, tracked

- **`article_features.parquet`** (272 KB, 1,976 rows): the final model-facing table. One row per
  article, judge-gated, every score a fusion of both scorers, with no raw probability triples and
  no provenance columns.
- **`article_features.md`**: the data dictionary for it, generated from the frame itself by
  `run_pipeline.write_feature_dictionary()` so the two cannot drift apart.

## `external/` — empty

Held a sourced company registry that was measured net harmful and removed with its module.

## Rebuilding

`python -m news_sentiment.text.run_pipeline` runs the whole text layer from
`processed_articles.parquet` and writes every artifact above except `raw/` and `eval/`. With warm
caches that is roughly 25 minutes; from cold it is about five hours, almost all of it the judge.

The chain is `raw_articles` → scrape → `processed_articles` → pipeline → `article_features`. Only
the first and last links are in git, so a fresh clone cannot rebuild the middle without re-scraping,
and the scrape will not return the identical corpus.
