Hey — found a leakage bug in the market features while doing EDA on the merged table, wanted to flag it before we train anything on it.

**The issue**

`momentum_1d`, `momentum_5d`, `momentum_20d`, `volatility_20d`, `beta_20d`, `relative_volume_20d`, and `daily_range_ratio_1d` are all built off the same helper pattern in `1.4-kk-labels-and-features.ipynb`:

```python
hist = series[series.index < ref_date].sort_index().tail(window)
```

`series` here is the daily OHLCV data, so its index is date-only (midnight). `ref_date` is the article's full publication timestamp, with real time-of-day. For an article published at, say, 14:00 on a trading day, that same day's own midnight-indexed row is still `< ref_date` (00:00 < 14:00), so it passes the filter and ends up counted as "prior" history — even though that day hasn't closed yet.

**Why it's a problem**

For any article published pre-market or during market hours, these "pre-publication" features are quietly including that same day's own close. That means the feature is partly built from the same price move the label (`abnormal_return_1d`) is trying to predict — they overlap.

I checked how often this actually happens, not just in theory. Reconstructed `momentum_1d` independently from raw OHLCV and compared:

- **market-hours articles: 741/741 (100%)** match the "uses today's own close" formula
- **pre-market articles: 757/774 (97.8%)** match it too
- **after-hours articles: 233/461 (50.5%)** — this one's expected/fine, since by after-hours that day's close genuinely is the most recent one available

That's **1,498 of 1,976 rows, 75.8% of the corpus**, where a "pre-publication" feature is using data that didn't exist yet at publication time. Did the same reconstruction test on `volatility_20d` independently and got the same result (397/400 sample match, 100% on market-hours) — so it's not just `momentum_1d`, it's every feature built off that shared helper (`cumulative_return`, `rolling_volatility`, `beta_vs_market`, `relative_volume`, `daily_range_ratio`, `prior_return`).

Not affected, checked separately: `days_to_earnings`, `session`, `news_volume` (all use different, correct logic), and the labels themselves (`abnormal_return_1d/3d`), which are built off `align_timestamp` and correctly start after publication.

**Why it matters for the model**

This is exactly the mechanism that produces a fake-looking result. If a model trains on these features as-is, it can score suspiciously well without having learned anything real about news → price — because for 3 out of 4 articles, part of the "prediction" was already baked into a feature. That's the kind of thing that'd blow up a walk-forward eval later, or worse, not get caught and just make the whole result untrustworthy.

**Possible fix**

The helper functions need to anchor their windows to the last *fully closed* trading day before the article's timestamp, not just before the raw timestamp compared against a date-only index. Two ways to do it:

1. Branch on session the same way `session_for` already does: pre-market/market-hours articles should only see through *yesterday's* close; after-hours articles can correctly include today's.
2. Cleaner option — reuse whatever `align_timestamp` already does for the labels (it clearly gets the trading-day alignment right there) as the anchor for every rolling window too, so labels and features go through one shared, leakage-safe alignment function instead of two separate ones that happen to disagree.

Happy to pair on this if useful — didn't want to touch the market layer myself since it's not my code, but wanted to get this in front of you before it goes into a model. Full writeup with the numbers is in `reports/momentum_1d-leakage-finding.md` if you want the detail.
