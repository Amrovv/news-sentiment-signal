### 11. Leakage smell test

Every feature must pass one test: could this have been computed at the moment the article was
published. `momentum_1d` is the one to check directly, since it's a rolling return anchored to a
trading day, and that anchor could legitimately be "the close before this article" or, wrongly, "the
close of the day this article was published on," which for a pre-market or market-hours article
hasn't happened yet. Reconstruct it independently from raw OHLCV and see which one the stored column
actually matches.

```python
ts = df["timestamp_utc"].dt.tz_localize(None)
df["_day"] = ts.dt.normalize()

same_day_close_return = {}
for d in df["_day"].unique():
    if d not in tsla_close.index:
        continue
    idx = tsla_close.index.get_loc(d)
    if idx < 1:
        continue
    same_day_close_return[d] = tsla_close.iloc[idx] / tsla_close.iloc[idx - 1] - 1

df["_same_day_hypothesis"] = df["_day"].map(same_day_close_return)
df["_matches_same_day"] = np.isclose(df["momentum_1d"], df["_same_day_hypothesis"], atol=1e-6)

print(df.groupby("session")["_matches_same_day"].agg(["sum", "count", "mean"]))
print()
print("overall row-level match rate:", df["_matches_same_day"].mean())
```

```
              sum  count      mean
session                           
after-hours   233    461  0.505423
market-hours  741    741  1.000000
pre-market    757    774  0.978036

overall row-level match rate: 0.8760121457489879
```

Not a smell, a confirmed formula. `momentum_1d` matches `close[day] / close[day-1] - 1`, the
return ending at that trading day's own close, for 100% of market-hours rows (741/741) and 97.8% of
pre-market rows (757/774). Neither session should have that day's close available: it hadn't happened
yet when the article was published. After-hours rows match at 50.5% (233/461), which is the expected
baseline rather than evidence of anything wrong, since by after-hours that day's close genuinely is
the most recent one, using it is correct there, and a roughly even split with whatever the alternate
formula would have given is what coincidence looks like on a real return series.

1,498 of 1,976 rows, 75.8% of the corpus, carry a `momentum_1d` computed from a price that did not
exist at publication time. This is confined to the momentum/volatility/beta/relative-volume family
built the same way in the market layer, `momentum_1d` is the only one verified directly here; the
labels (`abnormal_return_1d/3d`, defined to start the first trading day *after* publication) and the
text-layer sentiment features are unaffected by this specific mechanism.

**Decision.** This is flagged, not fixed, here. Correcting it means changing which close
`momentum_1d` (and its rolling siblings) anchor to in the market layer's own feature computation, not
something to patch by reassigning values inside this merged table. It is the most consequential
finding in this notebook to carry forward: a model trained on this table's market features is partly
learning from information that would not have existed at inference time, which is exactly the
mechanism a suspiciously high accuracy number, the 70%+ this notebook's own calibration would treat as
a red flag, would actually come from. This should be resolved in the market layer before any of the
correlations measured in this notebook are trusted as a ceiling on what the sentiment side can do.