# AAPL fetch report (raw)

Generated 2026-08-27 22:47 UTC from 14,465 raw articles.

## 1. Overview

| metric | value |
| --- | --- |
| total articles | 14,465 |
| date span | 2025-09-01 to 2026-08-01 (335 days) |
| active days | 335 of 335 (100.0%) |
| longest gap | 0 consecutive days with no article |


## 2. Articles by month

| month | articles | active days |
| --- | --- | --- |
| 2025-09 | 978 | 30 of 30 |
| 2025-10 | 1,060 | 31 of 31 |
| 2025-11 | 896 | 30 of 30 |
| 2025-12 | 851 | 31 of 31 |
| 2026-01 | 1,491 | 31 of 31 |
| 2026-02 | 1,254 | 28 of 28 |
| 2026-03 | 1,118 | 31 of 31 |
| 2026-04 | 1,571 | 30 of 30 |
| 2026-05 | 1,553 | 31 of 31 |
| 2026-06 | 1,629 | 30 of 30 |
| 2026-07 | 2,044 | 31 of 31 |
| 2026-08 | 20 | 1 of 1 |


![AAPL articles per month](figures/AAPL_raw_fetch_monthly.png)


## 3. Articles by day

min 3, median 39, max 164 articles/day.


![AAPL articles per day](figures/AAPL_raw_fetch_daily.png)


## 4. Burst check

`company_news` caps results per call regardless of window width and fills most-recent-first, which is what a whole-month pull turns into a handful of tail-of-month bursts instead of a steady year -- first diagnosed on TSLA (`notebooks/modelling/3.0`, section 6) and the reason `pull_company_news` windows by day instead. There's no single active-day-share or gap-length number that applies across every ticker, since a genuinely low-news company (a control stock, say) will legitimately have fewer active days than a heavily-covered one, and that's real signal, not a bug. What to actually look for, in sections 1-2 above:

- **A repeated, flat monthly count.** If several months show close to the same article count despite covering different numbers of active days, that's the truncation signature -- the call is hitting a ceiling, not describing real volume. This corpus's monthly counts run from 20 to 2,044; genuine variation across that range is the healthy sign, a tight repeated band is not.
- **A long gap relative to the corpus span.** This pull's longest gap is 0 day(s) against a 335-day span. A gap that spans what should be active trading days, rather than a real quiet period for this specific company, is worth checking against the daily chart above.
- **A low active-day share is not evidence on its own.** This pull sits at 100.0%; whether that's healthy depends on how newsy the company actually is, not on matching another ticker's number.
