"""Tests for the model-facing feature layer and the train/predict roundtrip.

Uses a small synthetic pooled article table rather than the real data file, so the
session-aggregation contract and the serialize/load/predict path are checked in
isolation and without needing the merge pipeline to have run.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from stock_predictor import features
from stock_predictor.modeling import predict as predict_mod
from stock_predictor.modeling import train as train_mod

SESSIONS_PER_TICKER = 30
TICKERS = ["AAA", "BBB"]

# Every article-level column build_session_table aggregates from, with a plausible
# dtype. The values do not matter to the structural tests; only that the schema is
# complete.
ARTICLE_NUMERIC = [
    "n_total_sents",
    "n_entity_sents",
    "n_ceo_sents",
    "n_boilerplate_sents",
    "article_length",
    "entity_share",
    "has_ceo_mention",
    "fus_conf_graft_floor_mean",
    "fus_conf_graft_floor_median",
    "fus_conf_graft_floor_lead",
    "fus_conf_graft_floor_top3_pos",
    "fus_conf_graft_floor_top3_neg",
    "fus_conf_graft_floor_spread",
    "fus_ceo_mean",
    "fus_headline",
    "fus_maxmag",
    "fus_trusted_mean",
    "fus_scorer_gap",
    "fus_headline_gap",
    "fus_lead_gap",
    *features.MARKET_SNAPSHOT,
]


def make_pooled(seed: int = 0, articles_per_session: int = 3) -> pd.DataFrame:
    """A synthetic pooled article table with the full schema build_session_table reads."""
    rng = np.random.default_rng(seed)
    rows = []
    aid = 0
    base = pd.Timestamp("2025-09-01T13:30:00Z")
    for s in range(SESSIONS_PER_TICKER):
        session_open = base + pd.Timedelta(days=s)
        for ticker in TICKERS:
            label = 1 if (s + hash(ticker)) % 2 == 0 else -1
            for _ in range(articles_per_session):
                aid += 1
                row = {
                    "article_id": aid,
                    "ticker": ticker,
                    "session_open": session_open,
                    "timestamp_utc": session_open - pd.Timedelta(hours=float(rng.integers(1, 6))),
                    "source": f"src{rng.integers(0, 3)}",
                    "session": rng.choice(features.SESSION_CATEGORIES),
                    "label_direction": label,
                    "abnormal_return_1d": float(rng.normal()) * 0.01,
                }
                for col in ARTICLE_NUMERIC:
                    row[col] = float(rng.normal())
                rows.append(row)
    return pd.DataFrame(rows)


def test_build_session_table_one_row_per_ticker_session():
    df = make_pooled()
    sessions = features.build_session_table(df)
    assert len(sessions) == SESSIONS_PER_TICKER * len(TICKERS)
    assert not sessions.duplicated(["session_open", "ticker"]).any()
    # Every model feature is present, plus the identity and label columns.
    for col in [*features.SESSION_MODEL_FEATURES, "label_direction", "session_open", "ticker"]:
        assert col in sessions.columns


def test_build_session_table_shares_sum_to_one_and_std_filled():
    df = make_pooled()
    sessions = features.build_session_table(df)
    shares = sessions[features.SESSION_MIX_FEATURES].sum(axis=1)
    assert np.allclose(shares, 1.0)
    # A single-article session has no spread; it must be 0, never NaN.
    assert sessions["fus_sentiment_std"].notna().all()


def test_single_article_session_has_zero_std():
    df = make_pooled(articles_per_session=1)
    sessions = features.build_session_table(df)
    assert (sessions["fus_sentiment_std"] == 0.0).all()


def test_train_predict_roundtrip(tmp_path):
    df = make_pooled()
    sessions = features.build_session_table(df)

    model = lgb.LGBMClassifier(**features.FINAL_MODEL_PARAMS, random_state=42, verbosity=-1)
    model.fit(sessions[features.SESSION_MODEL_FEATURES], sessions[features.LABEL])
    metadata = {
        "feature_cols": list(features.SESSION_MODEL_FEATURES),
        "label": features.LABEL,
        "params": dict(features.FINAL_MODEL_PARAMS),
        "trained_at": "test",
        "n_sessions": len(sessions),
        "tickers": TICKERS,
        "span": ["2025-09-01", "2025-09-30"],
    }
    path = tmp_path / "model.joblib"
    train_mod.save(model, metadata, path)

    bundle = predict_mod.load_model(path)
    preds = predict_mod.predict(df, bundle=bundle)

    assert len(preds) == len(sessions)
    assert set(preds.columns) == {
        "session_open",
        "ticker",
        "timestamp_utc",
        "pred_direction",
        "proba_up",
    }
    assert set(preds["pred_direction"].unique()).issubset({-1, 1})
    assert ((preds["proba_up"] >= 0) & (preds["proba_up"] <= 1)).all()


def test_predict_missing_features_raises():
    # A bundle that expects a feature build_session_table does not produce must be
    # caught rather than silently scored against the wrong columns.
    df = make_pooled()
    sessions = features.build_session_table(df)
    model = lgb.LGBMClassifier(**features.FINAL_MODEL_PARAMS, random_state=42, verbosity=-1)
    model.fit(sessions[features.SESSION_MODEL_FEATURES], sessions[features.LABEL])
    bundle = {
        "model": model,
        "feature_cols": [*features.SESSION_MODEL_FEATURES, "phantom_feature"],
    }
    with pytest.raises(ValueError, match="missing features"):
        predict_mod.predict(df, bundle=bundle)
