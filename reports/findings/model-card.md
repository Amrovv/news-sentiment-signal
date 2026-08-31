# Model card: session-level abnormal-return direction classifier

Predicts the sign of a ticker-session's next abnormal return (target return minus
SPY) from aggregated financial-news sentiment and pre-publication market features.
Produced by `stock_predictor.modeling.train`.

## Model

- **Type:** LightGBM binary classifier (`dart`), one combined model over text and market features. Not an ensemble.
- **Hyperparameters:** objective=binary, boosting_type=dart, num_leaves=31, learning_rate=0.08, min_child_samples=8, n_estimators=100, subsample=1.0, subsample_freq=3, colsample_bytree=0.5, reg_alpha=1.0, reg_lambda=5.0, max_depth=-1, is_unbalance=False
- **Features:** 36 (24 text, 12 market), built by `features.build_session_table`.
- **Label:** `label_direction`, the sign of `abnormal_return_1d` as -1 / 1.

## Training data

- 924 ticker-sessions aggregated from the pooled article table.
- Tickers: AAPL, AMZN, NVDA, TSLA.
- Span: 2025-09-02 to 2026-08-01.
- Fit on every row, no split (`market.evaluate.fit_final_model`). Trained 2026-08-31T22:14:45+00:00.

## Performance

Not measured on the shipped artifact, which is fit on all data. The numbers below are
the locked-holdout result for this exact configuration from `notebooks/modelling/3.7`
section 4 and `4.0`.

| metric | value |
|---|---|
| holdout accuracy | 0.5798 |
| majority baseline | 0.4309 |
| edge | +0.1489 |
| AUC | 0.6191 |
| McNemar p | 0.0019 |

Both accuracy figures sit inside the 53 to 57 percent band `3.2` calibrated as a real
result on this label.

## Intended use and limitations

- **Research artifact, not trading advice.** The edge is small and single-horizon.
- **Does not generalise across firms yet.** The holdout signal is carried by AAPL; NVDA sits on its baseline and AMZN and TSLA below theirs (`4.0` section 4.3).
- **The sentiment signal is largely backward-looking**, correlating with the previous session's return rather than the next one (`3.8`, `4.0` section 5).
- **`dart` is not bit-reproducible across builds**, so a retrain may land slightly off the published number.

## Reproducing

```bash
python -m stock_predictor.modeling.train      # writes models/session_model.joblib
python -m stock_predictor.modeling.predict INPUT.parquet   # scores new sessions
```
