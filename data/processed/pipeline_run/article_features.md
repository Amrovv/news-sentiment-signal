# `article_features.parquet`

One row per article, judge-gated. The final output of the text pipeline and the table a price model consumes.

**Rows:** 1,976 | **Columns:** 23 | **Span:** 2025-08-22 to 2026-08-07

Produced by `stock_predictor.text.run_pipeline`, Phase E. Every score is a fusion of FinBERT and DeBERTa ABSA at `CONF_FLOOR`; no raw probability triples and no provenance columns are carried. Sentences the referent judge rejected contribute to nothing here.

## Identity

| column | type | description |
|---|---|---|
| `article_id` | int64 | Publisher-assigned identifier, unique per row. |
| `ticker` | str | The target company the features were computed for. |
| `timestamp_utc` | datetime64[us, UTC] | Publication time. The join key for the market layer. |
| `source` | str | Publisher name. |

## Features

`coverage` is the share of rows that are non-null. NaN means the population was empty, never that the score was zero.

| column | type | coverage | mean | std | description |
|---|---|---:|---:|---:|---|
| `n_total_sents` | int64 | 100.0% | 26.610 | 28.670 | Sentences in the article after splitting, excluding scraper residue. |
| `n_entity_sents` | int64 | 100.0% | 6.546 | 7.317 | Sentences tagged as being about the target, non-boilerplate. |
| `n_ceo_sents` | int64 | 100.0% | 1.030 | 2.549 | Sentences mentioning the CEO but not the target itself. |
| `n_boilerplate_sents` | int64 | 100.0% | 6.555 | 7.762 | Sentences whose exact text repeats across five or more articles. |
| `entity_share` | float64 | 100.0% | 0.305 | 0.271 | n_entity_sents over the article's non-boilerplate sentence count. |
| `article_length` | int64 | 100.0% | 3986.251 | 3136.178 | Characters in the article as published, boilerplate included. |
| `fus_conf_graft_floor_mean` | float64 | 100.0% | 0.017 | 0.358 | Mean fused score over the target sentences. |
| `fus_conf_graft_floor_median` | float64 | 100.0% | 0.026 | 0.385 | Median of the same population. Disagrees on sign with the mean on about a fifth of articles, so it detects skew. |
| `fus_conf_graft_floor_lead` | float64 | 100.0% | 0.004 | 0.359 | Mean over target sentences in the opening window only. 0.0, not NaN, when the window holds none. |
| `fus_conf_graft_floor_top3_pos` | float64 | 100.0% | 0.247 | 0.436 | Mean of the three highest sentence scores. Reads the tail rather than the centre. |
| `fus_conf_graft_floor_top3_neg` | float64 | 100.0% | -0.222 | 0.461 | Mean of the three lowest sentence scores. |
| `fus_conf_graft_floor_spread` | float64 | 100.0% | 0.469 | 0.553 | top3_pos minus top3_neg. Separates a contested article from a quiet one, which the mean cannot. |
| `fus_ceo_mean` | float64 | 49.4% | 0.039 | 0.347 | Mean fused score over CEO-only sentences. NaN where the article has none. |
| `fus_headline` | float64 | 100.0% | -0.006 | 0.431 | The headline through both scorers, grafted. One string, one number. |
| `fus_maxmag` | float64 | 100.0% | -0.087 | 0.692 | Signed score of the single loudest target sentence, chosen by largest absolute fused score. |
| `fus_trusted_mean` | float64 | 100.0% | 0.016 | 0.360 | Mean fused score over the surface and coref_span channels only, excluding the weakest-measured channel. |
| `fus_scorer_gap` | float64 | 100.0% | 0.400 | 0.254 | Mean absolute difference between the two scorers over the target population. High means the article was contested between them. |
| `fus_headline_gap` | float64 | 100.0% | -0.023 | 0.471 | fus_headline minus the body mean. How far the headline leads or lags the article. |
| `fus_lead_gap` | float64 | 100.0% | -0.013 | 0.334 | The lead score minus the body mean. How far the opening leads or lags the rest. |

## Reading the columns

- **NaN over zero.** 0.0 is a real fused score, so an empty population is NaN rather than 0. The exception is `fus_conf_graft_floor_lead`, which is 0.0 when the opening window holds no target sentence: an article that does not mention the target early delivers no early sentiment, and that is a measurement.
- **Headline-only articles** keep their row with body scores filled to 0.0. They are identifiable by `n_entity_sents == 0`, and no other row has that, so the fill is reversible.
- **Sign** follows `pos - neg`: positive is favourable to the target.
- **Relevance.** Articles with no target sentence in the body and no target named in the headline are dropped, which is why the row count is below the corpus size.

Every design decision behind these columns is argued in `notebooks/text/2.2` and `notebooks/text/2.3`.
