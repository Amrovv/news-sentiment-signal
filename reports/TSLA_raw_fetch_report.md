# TSLA fetch report (raw)

Generated 2026-08-27 20:46 UTC from 14,974 raw articles.

## 1. Overview

| metric | value |
| --- | --- |
| total articles | 14,974 |
| date span | 2025-09-01 to 2026-08-01 (335 days) |
| active days | 335 of 335 (100.0%) |
| longest gap | 0 consecutive days with no article |


## 2. Articles by month

| month | articles | active days |
| --- | --- | --- |
| 2025-09 | 936 | 30 of 30 |
| 2025-10 | 1,149 | 31 of 31 |
| 2025-11 | 1,010 | 30 of 30 |
| 2025-12 | 991 | 31 of 31 |
| 2026-01 | 1,459 | 31 of 31 |
| 2026-02 | 1,162 | 28 of 28 |
| 2026-03 | 1,341 | 31 of 31 |
| 2026-04 | 2,036 | 30 of 30 |
| 2026-05 | 1,448 | 31 of 31 |
| 2026-06 | 1,588 | 30 of 30 |
| 2026-07 | 1,837 | 31 of 31 |
| 2026-08 | 17 | 1 of 1 |


![TSLA articles per month](figures/TSLA_raw_fetch_monthly.png)


## 3. Articles by day

min 1, median 44, max 194 articles/day.


![TSLA articles per day](figures/TSLA_raw_fetch_daily.png)


## 4. Burst check

The original Finnhub pull clustered into roughly 13 tail-of-month bursts: 92 of 351 days active (26.2%), gaps up to 27 days, traced to `company_news` capping results per call and backfilling from the most recent news first (`notebooks/modelling/3.0`, section 6). Section 1 and 2 above run the same check against this pull; a steady fetch reads as active-day share well above that 26.2% baseline and no gap anywhere near 27 days, with no month reading as a short tail-end burst in section 2.
