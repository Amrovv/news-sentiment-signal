"""Fit and serialize the shipped session-level model.

Builds the session table from the merged, model-ready pooled article table, fits
`3.7`'s winning LightGBM configuration on every row of it, and writes the fitted
model plus its feature list and provenance to `models/session_model.joblib`.

This is `market.evaluate.fit_final_model`'s job: no split, no score. The model's
performance is not measured here and must not be, since scoring it on any data
would mean that data had already been part of its training set. The real
performance numbers are the walk-forward and locked-holdout results in
`notebooks/modelling/3.7` and `4.0`; this step only produces the artifact to ship,
and records those numbers in the model card beside it.

Run with `python -m news_sentiment.modeling.train`. Nothing runs on import.
"""

import argparse
from datetime import UTC, datetime

import joblib
import lightgbm as lgb
from loguru import logger

from news_sentiment import features
from news_sentiment.config import MODELS_DIR, REPORTS_DIR, report_dir
from news_sentiment.market.evaluate import fit_final_model

MODEL_PATH = MODELS_DIR / "session_model.joblib"
MODEL_CARD_PATH = REPORTS_DIR / "findings" / "model-card.md"

# The established performance of this exact configuration, from the notebooks that
# selected and validated it. Reported in the model card rather than recomputed
# here, because a number this file produced by scoring its own training data would
# be meaningless. See notebooks/modelling/3.7 section 4 and 4.0.
HOLDOUT_METRICS = {
    "accuracy": 0.5798,
    "majority_baseline": 0.4309,
    "edge": 0.1489,
    "auc": 0.6191,
    "mcnemar_p": 0.0019,
}


def model_factory():
    """A fresh, unfitted LightGBM with the winning configuration."""
    return lgb.LGBMClassifier(**features.FINAL_MODEL_PARAMS, random_state=42, verbosity=-1)


def train(model_ready_path=features.MODEL_READY_POOLED_PATH):
    """Build the session table, fit the shipped model, and return (model, metadata)."""
    df = features.load_model_ready(model_ready_path)
    sessions = features.build_session_table(df)
    logger.info(
        f"session table: {len(sessions)} ticker-sessions from {len(df):,} article rows, "
        f"{sessions['ticker'].nunique()} tickers"
    )

    model = fit_final_model(
        model_factory, sessions, features.SESSION_MODEL_FEATURES, label_col=features.LABEL
    )

    metadata = {
        "feature_cols": list(features.SESSION_MODEL_FEATURES),
        "label": features.LABEL,
        "params": dict(features.FINAL_MODEL_PARAMS),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_sessions": len(sessions),
        "tickers": sorted(sessions["ticker"].unique().tolist()),
        "span": [
            str(sessions["timestamp_utc"].min().date()),
            str(sessions["timestamp_utc"].max().date()),
        ],
    }
    return model, metadata


def save(model, metadata, path=MODEL_PATH):
    """Serialize the model and its metadata together, so predict cannot load one
    without the feature list and label the other half needs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, **metadata}, path)
    logger.info(f"wrote {path}")


def write_model_card(metadata, path=MODEL_CARD_PATH):
    """Emit the model card beside the findings report, from the training metadata
    and the established holdout numbers."""
    report_dir("findings")  # ensure the directory exists
    m = HOLDOUT_METRICS
    params = ", ".join(f"{k}={v}" for k, v in metadata["params"].items())
    n_text = len(features.SESSION_TEXT_FEATURES)
    n_market = len(features.SESSION_MARKET_FEATURES)
    card = f"""# Model card: session-level abnormal-return direction classifier

Predicts the sign of a ticker-session's next abnormal return (target return minus
SPY) from aggregated financial-news sentiment and pre-publication market features.
Produced by `news_sentiment.modeling.train`.

## Model

- **Type:** LightGBM binary classifier (`{metadata["params"]["boosting_type"]}`), one combined model over text and market features. Not an ensemble.
- **Hyperparameters:** {params}
- **Features:** {len(metadata["feature_cols"])} ({n_text} text, {n_market} market), built by `features.build_session_table`.
- **Label:** `{metadata["label"]}`, the sign of `abnormal_return_1d` as -1 / 1.

## Training data

- {metadata["n_sessions"]} ticker-sessions aggregated from the pooled article table.
- Tickers: {", ".join(metadata["tickers"])}.
- Span: {metadata["span"][0]} to {metadata["span"][1]}.
- Fit on every row, no split (`market.evaluate.fit_final_model`). Trained {metadata["trained_at"]}.

## Performance

Not measured on the shipped artifact, which is fit on all data. The numbers below are
the locked-holdout result for this exact configuration from `notebooks/modelling/3.7`
section 4 and `4.0`.

| metric | value |
|---|---|
| holdout accuracy | {m["accuracy"]:.4f} |
| majority baseline | {m["majority_baseline"]:.4f} |
| edge | +{m["edge"]:.4f} |
| AUC | {m["auc"]:.4f} |
| McNemar p | {m["mcnemar_p"]:.4f} |

Both accuracy figures sit inside the 53 to 57 percent band `3.2` calibrated as a real
result on this label.

## Intended use and limitations

- **Research artifact, not trading advice.** The edge is small and single-horizon.
- **Does not generalise across firms yet.** The holdout signal is carried by AAPL; NVDA sits on its baseline and AMZN and TSLA below theirs (`4.0` section 4.3).
- **The sentiment signal is largely backward-looking**, correlating with the previous session's return rather than the next one (`3.8`, `4.0` section 5).
- **`dart` is not bit-reproducible across builds**, so a retrain may land slightly off the published number.

## Reproducing

```bash
python -m news_sentiment.modeling.train      # writes models/session_model.joblib
python -m news_sentiment.modeling.predict INPUT.parquet   # scores new sessions
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(card, encoding="utf-8")
    logger.info(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and serialize the shipped session model.")
    parser.add_argument(
        "--model-ready",
        default=str(features.MODEL_READY_POOLED_PATH),
        help="Path to the merged, model-ready pooled article table.",
    )
    parser.add_argument("--out", default=str(MODEL_PATH), help="Where to write the model.")
    parser.add_argument("--no-card", action="store_true", help="Skip writing the model card.")
    args = parser.parse_args()

    from pathlib import Path

    model, metadata = train(Path(args.model_ready))
    save(model, metadata, Path(args.out))
    if not args.no_card:
        write_model_card(metadata)
    logger.info("training complete")


if __name__ == "__main__":
    main()
