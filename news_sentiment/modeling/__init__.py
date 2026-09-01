"""Model layer: fit and score the shipped session-level classifier.

Reads the merged, model-ready pooled article table, aggregates it to sessions
through `news_sentiment.features.build_session_table` (the single definition
shared with the notebooks), and trains `3.7`'s winning LightGBM configuration.
Never imports `text/` or `market/` beyond the evaluation harness; the layers
meet only at the Parquet tables and `config.py`.

Modules:
    train.py    fit on every row, serialize models/session_model.joblib + card
    predict.py  score new ticker-sessions with the serialized bundle
"""
