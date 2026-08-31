"""Market & modeling layer.

Delivers a per-ticker market_features.parquet (one row per article) plus the
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
"""
