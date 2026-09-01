"""The model-facing feature layer: the pooled article table in, the session table
the shipped model trains and predicts on out.

The project's model is session-level, not article-level. `notebooks/modelling/3.6`
and `3.7` found that aggregating the merged article rows to one row per
`(session_open, ticker)` is the only transform that moves the text features from
0.0% of importance to the majority of it, and `3.7`/`4.0` fit the shipped model on
exactly this table. That aggregation lived only inside those notebooks; it is here
so `modeling.train` and `modeling.predict` build features the same way and cannot
drift from each other or from the notebook they were validated in.

    build_session_table()   pooled article rows -> one row per ticker-session
    load_model_ready()      read the merged, model-ready pooled article table

The winning LightGBM configuration and the exact feature lists are constants here,
the single source of truth for both training and inference.
"""

import numpy as np
import pandas as pd

from news_sentiment.config import PROCESSED_DATA_DIR

# The merged, model-ready pooled article table `3.5` section 15 wrote: one row per
# (article_id, ticker), sentiment and market features joined to the 1-day label.
MODEL_READY_POOLED_PATH = PROCESSED_DATA_DIR / "merged" / "model_ready_pooled.parquet"

LABEL = "label_direction"
GROUP_KEYS = ["session_open", "ticker"]

# The three market sessions, fixed by market hours. Held in this order so the
# published-share columns are always built the same way regardless of which
# sessions a given corpus happens to contain.
SESSION_CATEGORIES = ["pre-market", "market-hours", "after-hours"]

# Article-level market columns carried into the session as a snapshot of the
# latest-published article's value (see build_session_table on why latest, not
# mean). `session` is excluded: it becomes the published-share columns below.
MARKET_SNAPSHOT = [
    "momentum_1d",
    "momentum_5d",
    "momentum_20d",
    "volatility_20d",
    "beta_20d",
    "relative_volume_20d",
    "daily_range_ratio_1d",
    "days_to_earnings",
    "news_volume",
]

SESSION_MIX_FEATURES = [f"published_{s}_share" for s in SESSION_CATEGORIES]

SESSION_TEXT_FEATURES = [
    "n_articles",
    "n_sources",
    "n_total_sents",
    "n_entity_sents",
    "n_ceo_sents",
    "n_boilerplate_sents",
    "article_length",
    "entity_share",
    "has_ceo_mention",
    "fus_sentiment_mean",
    "fus_sentiment_std",
    "fus_sentiment_loudest",
    "fus_median",
    "fus_lead",
    "fus_top3_pos",
    "fus_top3_neg",
    "fus_spread",
    "fus_ceo_mean",
    "fus_headline",
    "fus_maxmag",
    "fus_trusted_mean",
    "fus_scorer_gap",
    "fus_headline_gap",
    "fus_lead_gap",
]

SESSION_MARKET_FEATURES = MARKET_SNAPSHOT + SESSION_MIX_FEATURES

# The features the shipped model reads, in a fixed order. A serialized model
# carries its own copy of this list; keeping the order stable here means a table
# built now lines up with a model trained earlier.
SESSION_MODEL_FEATURES = SESSION_TEXT_FEATURES + SESSION_MARKET_FEATURES

# `3.7` section 3.4's winning trial (4023), the combined session-level model. Not
# an ensemble: no ensemble configuration placed in the top 15 of either search
# phase. `random_state`/`verbosity` are execution settings, kept out so the model
# hyperparameters read as one block.
FINAL_MODEL_PARAMS = {
    "objective": "binary",
    "boosting_type": "dart",
    "num_leaves": 31,
    "learning_rate": 0.08,
    "min_child_samples": 8,
    "n_estimators": 100,
    "subsample": 1.0,
    "subsample_freq": 3,
    "colsample_bytree": 0.5,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "max_depth": -1,
    "is_unbalance": False,
}


def load_model_ready(path=MODEL_READY_POOLED_PATH) -> pd.DataFrame:
    """Read the merged, model-ready pooled article table, sorted by publication."""
    if not path.exists():
        raise FileNotFoundError(
            f"No model-ready table at {path}. Build it through the merge pipeline "
            "(python -m news_sentiment.merge.run_pipeline --all)."
        )
    return pd.read_parquet(path).sort_values("timestamp_utc").reset_index(drop=True)


def _signed_max_abs(values: pd.Series) -> float:
    """The single loudest score in the group, keeping its sign. NaN on an empty group."""
    values = values.dropna()
    if values.empty:
        return np.nan
    return values.iloc[values.abs().to_numpy().argmax()]


def build_session_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the pooled article rows to one row per `(session_open, ticker)`.

    The recipe is `3.7` section 2.1's, unchanged: counts sum, scores average,
    `fus_conf_graft_floor_mean`'s single loudest value keeps its sign, and the
    market features are taken from the session's latest-published article rather
    than averaged. A session spans a median of two calendar days, so it holds two
    versions of `momentum_1d`; by the session's own open the later close is known,
    so the latest article's snapshot is the freshest information legitimately
    available at the moment being predicted, and averaging would blend it with a
    stale one.

    `session` becomes three published-share columns describing the session's
    coverage timing, not a grouping key. Rows are sorted by
    `(timestamp_utc, ticker)`, the tie-break that makes the order deterministic
    when an article covering two companies is the last article for both of their
    ticker-sessions.

    Returns the session table carrying `SESSION_MODEL_FEATURES`, the label,
    `session_open`, `ticker`, `abnormal_return_1d` and `timestamp_utc`.
    """
    df = df.copy()
    df["session"] = pd.Categorical(df["session"], categories=SESSION_CATEGORIES)

    grouped = df.groupby(GROUP_KEYS, observed=True)
    latest_idx = grouped["timestamp_utc"].idxmax()
    snapshot = df.loc[latest_idx, GROUP_KEYS + MARKET_SNAPSHOT].set_index(GROUP_KEYS)

    sessions = grouped.agg(
        label_direction=("label_direction", "first"),
        abnormal_return_1d=("abnormal_return_1d", "first"),
        timestamp_utc=("timestamp_utc", "max"),
        n_articles=("article_id", "size"),
        n_sources=("source", "nunique"),
        n_total_sents=("n_total_sents", "sum"),
        n_entity_sents=("n_entity_sents", "sum"),
        n_ceo_sents=("n_ceo_sents", "sum"),
        n_boilerplate_sents=("n_boilerplate_sents", "sum"),
        article_length=("article_length", "sum"),
        entity_share=("entity_share", "mean"),
        has_ceo_mention=("has_ceo_mention", "mean"),
        fus_sentiment_mean=("fus_conf_graft_floor_mean", "mean"),
        fus_sentiment_std=("fus_conf_graft_floor_mean", "std"),
        fus_sentiment_loudest=("fus_conf_graft_floor_mean", _signed_max_abs),
        fus_median=("fus_conf_graft_floor_median", "mean"),
        fus_lead=("fus_conf_graft_floor_lead", "mean"),
        fus_top3_pos=("fus_conf_graft_floor_top3_pos", "mean"),
        fus_top3_neg=("fus_conf_graft_floor_top3_neg", "mean"),
        fus_spread=("fus_conf_graft_floor_spread", "mean"),
        fus_ceo_mean=("fus_ceo_mean", "mean"),
        fus_headline=("fus_headline", "mean"),
        fus_maxmag=("fus_maxmag", "mean"),
        fus_trusted_mean=("fus_trusted_mean", "mean"),
        fus_scorer_gap=("fus_scorer_gap", "mean"),
        fus_headline_gap=("fus_headline_gap", "mean"),
        fus_lead_gap=("fus_lead_gap", "mean"),
    )

    # Coverage timing as shares of the session's articles, in SESSION_CATEGORIES
    # order and reindexed so a session missing one of the three reads 0, not NaN.
    mix = (
        df.groupby(GROUP_KEYS + ["session"], observed=True).size().unstack("session", fill_value=0)
    )
    mix = mix.reindex(columns=SESSION_CATEGORIES, fill_value=0)
    mix = mix.div(mix.sum(axis=1), axis=0)
    mix.columns = SESSION_MIX_FEATURES

    sessions = sessions.join(snapshot).join(mix).reset_index()
    sessions = sessions.sort_values(["timestamp_utc", "ticker"]).reset_index(drop=True)

    # A single-article session has no within-session spread; that is a real 0, not
    # a missing measurement, so it is filled rather than left NaN.
    sessions["fus_sentiment_std"] = sessions["fus_sentiment_std"].fillna(0.0)

    return sessions
