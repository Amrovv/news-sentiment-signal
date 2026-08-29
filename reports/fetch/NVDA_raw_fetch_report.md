# NVDA fetch report (raw)

Generated 2026-08-27 22:19 UTC from 38,702 raw articles.

## 1. Overview

| metric | value |
| --- | --- |
| total articles | 38,702 |
| date span | 2025-09-01 to 2026-08-01 (335 days) |
| active days | 335 of 335 (100.0%) |
| longest gap | 0 consecutive days with no article |


## 2. Articles by month

| month | articles | active days |
| --- | --- | --- |
| 2025-09 | 1,366 | 30 of 30 |
| 2025-10 | 1,622 | 31 of 31 |
| 2025-11 | 1,647 | 30 of 30 |
| 2025-12 | 1,375 | 31 of 31 |
| 2026-01 | 2,489 | 31 of 31 |
| 2026-02 | 3,793 | 28 of 28 |
| 2026-03 | 5,451 | 31 of 31 |
| 2026-04 | 4,472 | 30 of 30 |
| 2026-05 | 5,137 | 31 of 31 |
| 2026-06 | 5,620 | 30 of 30 |
| 2026-07 | 5,618 | 31 of 31 |
| 2026-08 | 112 | 1 of 1 |


![NVDA articles per month](figures/NVDA_raw_fetch_monthly.png)


## 3. Articles by day

min 5, median 102, max 250 articles/day.


![NVDA articles per day](figures/NVDA_raw_fetch_daily.png)


## 4. Burst check

The original Finnhub pull clustered into roughly 13 tail-of-month bursts: 92 of 351 days active (26.2%), gaps up to 27 days, traced to `company_news` capping results per call and backfilling from the most recent news first (`notebooks/modelling/3.0`, section 6). Section 1 and 2 above run the same check against this pull; a steady fetch reads as active-day share well above that 26.2% baseline and no gap anywhere near 27 days, with no month reading as a short tail-end burst in section 2.
