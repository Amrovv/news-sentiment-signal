# AAPL: text-market merge

The two pipelines meet here. This report covers the join of `article_features.parquet` onto `market_features.parquet` on `article_id`, what it dropped, and whether the key and the features can still be trusted.

All checks passed.

**Merged rows:** 7,041 | **Columns:** 36 | **Span:** 2025-09-01 to 2026-08-01

## 1. Did the key hold?

`article_id` carries the whole join, so these run on every merge rather than being assumed. The third is the one that matters most: two tables can each hold unique ids and still disagree about what an id *means*, and a join across that disagreement pairs one article's sentiment with another article's return, which nothing downstream could detect.

| check | result | detail |
|---|---|---|
| article_id is unique within AAPL text features | **pass** | 7,041 rows, 7,041 distinct article_id |
| article_id is unique within AAPL market features | **pass** | 7,527 rows, 7,527 distinct article_id |
| one article_id means one article (text vs market) | **pass** | all 7,041 shared ids agree on timestamp_utc |
| every text id exists in the corpus | **pass** | all 7,041 ids found in the 7,527-article corpus |
| every market id exists in the corpus | **pass** | all 7,527 ids found in the 7,527-article corpus |

## 2. Is `momentum_1d` still leak-free?

Every feature must be computable at the moment the article was published. `momentum_1d` is reconstructed from raw OHLCV under both hypotheses and compared against the stored column:

- **leaky**: anchored to the article's own trading day, using a close that had not happened yet for any pre-market or market-hours article. This was a real bug; `reports/findings/momentum_1d-leakage-finding.md` documents it.
- **correct**: anchored to the day before, which is what `cumulative_return`'s `ref_date.normalize()` cutoff produces.

| metric | value |
| --- | --- |
| verdict | **pass** |
| rows tested | 7,527 |
| match the pre-publication formula | 7,527 (100.0%) |
| real leaks | 0 |
| undecidable | 717 (published on a non-trading day, where both formulas are arithmetically identical) |


By session, since the bug could only ever show up before a close:

| session | rows | match correct | real leaks | undecidable |
|---|---:|---:|---:|---:|
| after-hours | 1,590 | 1,590 (100.0%) | 0 | 713 |
| market-hours | 3,132 | 3,132 (100.0%) | 0 | 0 |
| pre-market | 2,805 | 2,805 (100.0%) | 0 | 4 |

## 3. What did not merge?

Two filters remove articles, and neither is a fault. The text layer drops articles that never mention the target; the market layer drops articles with no price bar to label against. A merged table smaller than expected for any other reason would show up as a failed check in section 1, not here.

| population | articles |
| --- | --- |
| corpus | 7,527 |
| text feature rows | 7,041 |
| market feature rows | 7,527 |
| merged | 7,041 |
| text rows that found a market partner | 7,041 of 7,041 (100.0%) |
| in text, not in market | 0 (no price bar to label against) |
| in market, not in text | 486 (no mention of AAPL in body or headline) |


![AAPL join composition](figures/AAPL_merge_join_composition.png)

Dropped articles should be spread across the corpus. A month losing far more than its neighbours points at something that failed for a stretch, a rate-limited scrape or a gap in price history, rather than at a filter doing its job.

![AAPL dropped by month](figures/AAPL_merge_dropped_by_month.png)

## 4. Does the merged table look like data?

A weak relationship between sentiment and return is the expected result and not the point of this section. The point is that a mis-paired join produces no relationship at all: identical boxes and a structureless scatter. These are read as a merge check, never as a finding.

| metric | value |
| --- | --- |
| rows with both scores | 7,041 |
| correlation, sentiment vs return | +0.0171 |
| mean sentiment, up articles | +0.0555 |
| mean sentiment, down articles | +0.0514 |
| difference | +0.0040 |


![AAPL sentiment vs return](figures/AAPL_merge_sentiment_vs_return.png)

## 5. Shared articles

Finnhub returns one story for every ticker it mentions, so an article about several companies appears in several corpora. Those rows carry the same text but different sentiment, scored toward a different target, and a different company's return as the label. They are genuinely different training examples, which is why the pooled table is keyed on `(article_id, ticker)` and never on `article_id` alone.

This is also the contamination a cross-firm transfer test has to exclude: training on AAPL and testing on a ticker below means the model has already seen that share of the test text.

| also appears under | shared articles | share of this table |
|---|---:|---:|
| NVDA | 1,408 | 20.0% |
| AMZN | 1,132 | 16.1% |
| TSLA | 803 | 11.4% |

## Reading this table

- **Join key.** `article_id` is unique here. In the pooled table it is not; the key there is `(article_id, ticker)`.
- **`label_direction` has three values.** 0 covers a flat or missing return. Filter it out before training a binary classifier.
- **`session` is categorical.** Cast it to a pandas `category` before handing it to LightGBM.

Column definitions live with the layers that produce them: `reports/market/{ticker}_market_features_report.md` for the market columns, `data/processed/pipeline_run/{ticker}/article_features.md` for the text ones.
