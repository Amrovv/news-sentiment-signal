"""Score new ticker-sessions with the shipped model.

Loads `models/session_model.joblib`, builds the session table from a pooled
article table the same way `train` did (through `features.build_session_table`,
so training and inference cannot diverge), and returns one predicted direction and
probability per `(session_open, ticker)`.

Input is a merged, model-ready pooled article table: the output of the merge
pipeline, one row per `(article_id, ticker)` with the sentiment and market feature
columns. The model never sees raw article text; it reads the aggregated features.

Run with `python -m stock_predictor.modeling.predict INPUT.parquet [--out OUT]`.
Nothing runs on import.
"""

import argparse
from pathlib import Path

import joblib
from loguru import logger
import pandas as pd

from stock_predictor import features
from stock_predictor.modeling.train import MODEL_PATH


def load_model(path=MODEL_PATH) -> dict:
    """Load the serialized model bundle: the fitted model plus its feature list."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No model at {path}. Train it first: python -m stock_predictor.modeling.train"
        )
    bundle = joblib.load(path)
    missing = {"model", "feature_cols"} - set(bundle)
    if missing:
        raise ValueError(f"Model bundle at {path} is missing {sorted(missing)}")
    return bundle


def predict(df: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    """Predict direction and probability for each ticker-session in `df`.

    `df` is a pooled article table (article-level). It is aggregated to sessions
    here, so the same table a model was trained on scores identically. The feature
    columns come from the bundle, not from the current `features` module, so a model
    trained against an older feature list still lines up with its own columns.

    Returns one row per `(session_open, ticker)`: the identity columns, the
    predicted `label_direction`, and `proba_up`, the model's probability of a
    positive abnormal return.
    """
    bundle = bundle or load_model()
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    sessions = features.build_session_table(df)
    missing = [c for c in feature_cols if c not in sessions.columns]
    if missing:
        raise ValueError(
            f"the session table is missing features the model needs: {missing}. "
            "The input table does not carry the columns build_session_table aggregates from."
        )

    X = sessions[feature_cols]
    pos_label = max(model.classes_)
    pos_col = list(model.classes_).index(pos_label)

    out = sessions[["session_open", "ticker", "timestamp_utc"]].copy()
    out["pred_direction"] = model.predict(X)
    out["proba_up"] = model.predict_proba(X)[:, pos_col]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Score ticker-sessions with the shipped model.")
    parser.add_argument("input", help="Pooled article table (parquet) to score.")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Path to the model bundle.")
    parser.add_argument(
        "--out",
        default=None,
        help="Where to write predictions (parquet). Prints a summary if omitted.",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    preds = predict(df, bundle=load_model(args.model))
    logger.info(
        f"scored {len(preds)} ticker-sessions; "
        f"{(preds['pred_direction'] > 0).mean():.1%} predicted up"
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(out_path, index=False)
        logger.info(f"wrote {out_path}")
    else:
        print(preds.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
