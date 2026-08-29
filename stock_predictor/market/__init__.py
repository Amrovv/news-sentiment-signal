"""Person B — Market & Modeling layer.

Delivers data/processed/labels_features.parquet (one row per article) plus the
evaluation harness every model in the project runs through. This package never
imports `text/`; the layers meet only at the Parquet tables and shared
constants in `config.py`.

Modules:
    prices.py               yfinance OHLCV + earnings, the NYSE calendar
    timestamp_alignment.py  timestamp -> trading day  (critical; unit-tested in tests/)
    features.py             one pre-publication feature per function, ticker-agnostic
    labels.py               abnormal returns + market features, one row per article
    run_pipeline.py         corpus in, market_features.parquet out, per ticker
    evaluate.py             walk-forward harness      (every model runs through this)

Still to promote from notebooks when stable / reused:
    ladder.py          LightGBM stages (a) and (b)
    analysis.py        event study, horizon, lead-lag, backtest
    transfer.py        cross-firm runs
    explain.py         SHAP + live market-state fetcher (for the demo)
"""
