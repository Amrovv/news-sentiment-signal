# NVDA fetch report (processed)

Generated 2026-08-28 14:28 UTC from 17,294 processed articles.

## 1. Overview

| metric | value |
| --- | --- |
| total articles | 17,294 |
| date span | 2025-09-01 to 2026-08-01 (335 days) |
| active days | 333 of 335 (99.4%) |
| longest gap | 1 consecutive days with no article |


## 2. Articles by month

| month | articles | active days |
| --- | --- | --- |
| 2025-09 | 809 | 30 of 30 |
| 2025-10 | 976 | 31 of 31 |
| 2025-11 | 940 | 30 of 30 |
| 2025-12 | 777 | 29 of 31 |
| 2026-01 | 1,655 | 31 of 31 |
| 2026-02 | 1,594 | 28 of 28 |
| 2026-03 | 1,660 | 31 of 31 |
| 2026-04 | 1,463 | 30 of 30 |
| 2026-05 | 3,046 | 31 of 31 |
| 2026-06 | 2,299 | 30 of 30 |
| 2026-07 | 2,056 | 31 of 31 |
| 2026-08 | 19 | 1 of 1 |


![NVDA articles per month](figures/NVDA_processed_fetch_monthly.png)


## 3. Articles by day

min 0, median 46, max 177 articles/day.


![NVDA articles per day](figures/NVDA_processed_fetch_daily.png)


## 4. Burst check

`company_news` caps results per call regardless of window width and fills most-recent-first, which is what a whole-month pull turns into a handful of tail-of-month bursts instead of a steady year -- first diagnosed on TSLA (`notebooks/modelling/3.0`, section 6) and the reason `pull_company_news` windows by day instead. There's no single active-day-share or gap-length number that applies across every ticker, since a genuinely low-news company (a control stock, say) will legitimately have fewer active days than a heavily-covered one, and that's real signal, not a bug. What to actually look for, in sections 1-2 above:

- **A repeated, flat monthly count.** If several months show close to the same article count despite covering different numbers of active days, that's the truncation signature -- the call is hitting a ceiling, not describing real volume. This corpus's monthly counts run from 19 to 3,046; genuine variation across that range is the healthy sign, a tight repeated band is not.
- **A long gap relative to the corpus span.** This pull's longest gap is 1 day(s) against a 335-day span. A gap that spans what should be active trading days, rather than a real quiet period for this specific company, is worth checking against the daily chart above.
- **A low active-day share is not evidence on its own.** This pull sits at 99.4%; whether that's healthy depends on how newsy the company actually is, not on matching another ticker's number.
