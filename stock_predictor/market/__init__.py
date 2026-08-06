"""Person B — Market & Modeling layer.

Delivers data/processed/labels_features.parquet (one row per article) plus the
evaluation harness every model in the project runs through. This package never
imports `text/`; the layers meet only at the Parquet tables and shared
constants in `config.py`.

Modules (promote from notebooks/b_*.ipynb when stable / reused):
    prices.py          yfinance OHLCV + earnings
    calendar_align.py  timestamp -> trading day  (critical; unit-tested in tests/)
    labels.py          abnormal returns + market features
    evaluate.py        walk-forward harness      (every model runs through this)
    ladder.py          LightGBM stages (a) and (b)
    analysis.py        event study, horizon, lead-lag, backtest
    transfer.py        cross-firm runs
    explain.py         SHAP + live market-state fetcher (for the demo)
"""
