"""The shared walk-forward evaluation harness every model in this project runs through.

Two key points:

1. **Splitting.** Walk-forward CV isn't one train/test split, it's several -- each fold
   retrains on a larger expanding window and tests on the block right after it. Fitting
   and splitting are inseparable here, so both live in `evaluate()` rather than being left
   for each model script to reimplement (and potentially get subtly wrong: a random
   shuffle leaks the future, and a plain row-level split can divide a single calendar
   day's articles between train and test -- see `day_grouped_splits`).
2. **What "beats the baseline" means.** A model that's a hair better than always guessing
   the majority class on a 200-row fold isn't necessarily better at all -- it could be
   noise. `evaluate()` reports a paired significance test (McNemar's) against the
   baseline, not just a boolean accuracy comparison.

The flow this module is built for:

    1. Define model_factory -- a zero-argument callable returning a fresh, unfitted
       model each time it's called (e.g. `lambda: lgb.LGBMClassifier(**params)`). A
       factory rather than a fitted instance, because a walk-forward evaluation needs
       a genuinely fresh model per fold, and because a factory works for any model type
       -- a plain callable has no requirement to be an sklearn BaseEstimator the way
       sklearn.base.clone() would, so this also covers something like a FinBERT
       fine-tune that evaluate() otherwise has no opinion about.
    2. evaluate(model_factory, data, feature_cols, ...) across walk-forward folds ->
       an EvalResult holding every fold's model, predictions, indices, and metrics,
       plus the aggregate across folds.
    3. Tuning wraps around evaluate() as an outer loop: call it repeatedly with
       different model_factory variants, compare EvalResult.aggregate, pick a winner.
       evaluate() itself never tunes anything.
    4. fit_final_model(model_factory, data, feature_cols, ...) refits once on every
       row of `data` -- no split at all -- for the artifact that actually ships.

There is deliberately no held-out block anywhere in this module. Every walk-forward
fold's test score is already out-of-sample for that fold, so the fold metrics from
step 2 are the only real performance numbers this project reports; the model from step
4 is trained on strictly more data than any fold model and is never itself scored, since
scoring it on anything would mean that thing had already been part of its training data.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

DEFAULT_TARGET_COL = "label_direction"
DEFAULT_DATE_COL = "timestamp_utc"


def day_grouped_splits(dates: pd.Series, n_splits: int = 5):
    """Yield (train_mask, test_mask) row-level boolean arrays per walk-forward fold.

    TimeSeriesSplit runs over unique sorted calendar days, not rows, so every row
    published on a given day always falls entirely in train or entirely in test --
    never split between them. Without this, a plain row-level TimeSeriesSplit could
    divide one day's articles across train and test; since same-day articles can
    share an identical label and several identical market features, that would let a
    model be scored on rows whose answer is already implied by sibling rows it
    trained on.
    """
    days = dates.dt.normalize()
    unique_days = np.sort(days.unique())

    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_day_idx, test_day_idx in tscv.split(unique_days):
        train_days = set(unique_days[train_day_idx])
        test_days = set(unique_days[test_day_idx])
        train_mask = days.isin(train_days).to_numpy()
        test_mask = days.isin(test_days).to_numpy()
        yield train_mask, test_mask


def _mcnemar_p_value(y_true, model_pred, baseline_pred) -> float:
    """One-sided McNemar's exact test: is the model right more often than the
    baseline is, among the rows where they disagree?

    Only the discordant pairs matter -- rows both get right or both get wrong carry
    no information about which is better. b = model right & baseline wrong, c =
    model wrong & baseline right. Under "no real difference", b should be a fair
    coin flip of b+c, so this is a one-sided exact binomial test of b against
    Binomial(b + c, 0.5). NaN when there's no discordance to test (identical
    predictions), rather than a misleadingly confident p-value.
    """
    model_correct = np.asarray(model_pred) == np.asarray(y_true)
    baseline_correct = np.asarray(baseline_pred) == np.asarray(y_true)
    b = int(np.sum(model_correct & ~baseline_correct))
    c = int(np.sum(~model_correct & baseline_correct))
    if b + c == 0:
        return np.nan
    return binomtest(b, b + c, p=0.5, alternative="greater").pvalue


def _score(y_train, y_test, y_pred, y_score, pos_label) -> dict:
    """Every metric a fold reports."""
    y_test_binary = (y_test == pos_label).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    majority_class = y_train.mode().iloc[0]
    baseline_pred = np.full(len(y_test), majority_class)
    baseline_accuracy = accuracy_score(y_test, baseline_pred)

    p_value = _mcnemar_p_value(y_test, y_pred, baseline_pred)

    return {
        "n_train": len(y_train),
        "n_test": len(y_test),
        "accuracy": accuracy,
        "precision": precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0),
        "recall": recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0),
        "f1": f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0),
        "auc": roc_auc_score(y_test, y_score),
        "log_loss": log_loss(y_test_binary, y_score, labels=[0, 1]),
        "brier_score": brier_score_loss(y_test_binary, y_score),
        "majority_baseline_accuracy": baseline_accuracy,
        "beats_baseline": accuracy > baseline_accuracy,
        "mcnemar_p_value": p_value,
        "significantly_beats_baseline": bool(p_value < 0.05) if not np.isnan(p_value) else False,
    }


@dataclass
class FoldResult:
    """Everything about one walk-forward fold: the model it produced, what that
    model predicted, which rows it trained/tested on, and how it scored."""

    fold: int
    model: object
    train_idx: np.ndarray
    test_idx: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    metrics: dict


@dataclass
class EvalResult:
    """The output of evaluate(): one FoldResult per walk-forward fold, plus the
    tidy/aggregate views built from them.

    Nothing here is computed lazily from data evaluate() has discarded -- metrics_df
    and aggregate are pure reshapes of what's already in .folds, so results printed
    from either always agree with the fold objects themselves.
    """

    folds: list = field(default_factory=list)

    @property
    def models(self) -> list:
        """The fold-fitted models, in fold order. None of these is "the" model to
        deploy -- see fit_final_model() for that."""
        return [f.model for f in self.folds]

    @property
    def metrics_df(self) -> pd.DataFrame:
        """One row per fold: fold index, date ranges, and every metric from _score()."""
        return pd.DataFrame([{"fold": f.fold, **f.metrics} for f in self.folds])

    @property
    def aggregate(self) -> pd.DataFrame:
        """Mean and std of every numeric metric, across folds."""
        numeric = self.metrics_df.drop(columns=["fold"]).select_dtypes(include="number")
        return numeric.agg(["mean", "std"])


def evaluate(
    model_factory,
    data: pd.DataFrame,
    feature_cols: list,
    label_col: str = DEFAULT_TARGET_COL,
    n_splits: int = 5,
    date_col: str = DEFAULT_DATE_COL,
) -> EvalResult:
    """Walk-forward evaluation: expanding-window splits grouped by calendar day.

    model_factory must be a zero-argument callable returning a fresh, unfitted model
    each time it's called -- e.g. `lambda: lgb.LGBMClassifier(**params)`. That model
    must implement .fit(X, y), .predict(X), and .predict_proba(X). A fresh model is
    requested for every fold; none of them carries state from one fold into the next.

    Use this, not fit_final_model(), for model selection: choosing between feature
    sets, hyperparameters, or model families -- call it repeatedly with different
    model_factory variants and compare EvalResult.aggregate. Every model compared
    this way sees the same fold boundaries, which is what makes the comparison
    meaningful. evaluate() itself never tunes anything; tuning is an outer loop
    around it.

    Returns an EvalResult: one FoldResult per fold (the fitted model, its
    predictions and probabilities, the train/test row indices into `data` used, and
    that fold's metrics), plus .metrics_df and .aggregate views across folds.
    """
    data = data.sort_values(date_col).reset_index(drop=True)
    X_all = data[feature_cols]
    y_all = data[label_col]
    dates = data[date_col]

    folds = []
    for fold_num, (train_mask, test_mask) in enumerate(
        day_grouped_splits(dates, n_splits=n_splits), start=1
    ):
        train_idx = np.flatnonzero(train_mask)
        test_idx = np.flatnonzero(test_mask)

        X_train, y_train = X_all.iloc[train_idx], y_all.iloc[train_idx]
        X_test, y_test = X_all.iloc[test_idx], y_all.iloc[test_idx]

        model = model_factory()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        pos_label = max(model.classes_)
        pos_col = list(model.classes_).index(pos_label)
        y_proba = model.predict_proba(X_test)
        y_score = y_proba[:, pos_col]

        metrics = _score(y_train, y_test, y_pred, y_score, pos_label)
        metrics["train_start"] = dates.iloc[train_idx].min()
        metrics["train_end"] = dates.iloc[train_idx].max()
        metrics["test_start"] = dates.iloc[test_idx].min()
        metrics["test_end"] = dates.iloc[test_idx].max()

        folds.append(
            FoldResult(
                fold=fold_num,
                model=model,
                train_idx=train_idx,
                test_idx=test_idx,
                y_pred=y_pred,
                y_proba=y_proba,
                metrics=metrics,
            )
        )

    return EvalResult(folds=folds)


def fit_final_model(
    model_factory,
    data: pd.DataFrame,
    feature_cols: list,
    label_col: str = DEFAULT_TARGET_COL,
):
    """Fit the model that actually ships: model_factory(), trained on every row of
    `data`, no split at all.

    Call this once, after evaluate() has been used (directly, or as the inner loop
    of a hyperparameter search) to pick model_factory, feature_cols, and every other
    decision. At that point there is nothing left for held-out data to protect
    against: this model is not evaluated by this function or by anything else,
    because scoring it on any data would mean that data had already been part of
    its training set. The walk-forward fold metrics from evaluate() are, and remain,
    the only real performance numbers for this project -- this call exists purely to
    produce the artifact to save and deploy.
    """
    X, y = data[feature_cols], data[label_col]
    model = model_factory()
    model.fit(X, y)
    return model


def feature_importance(result: EvalResult, feature_cols: list) -> pd.DataFrame | None:
    """Mean +/- std feature importance across the fold models an EvalResult already
    holds, for models that expose `.feature_importances_` after fitting (LightGBM
    and sklearn's tree ensembles). Returns None for models that don't --
    interpretability is a bonus this harness offers when it can, not a requirement
    every model must satisfy.

    Reuses result.models rather than fitting anything itself: the whole point of
    EvalResult carrying its fold models is that analysis like this doesn't need a
    second, separate walk-forward pass.
    """
    per_fold = []
    for model in result.models:
        if not hasattr(model, "feature_importances_"):
            return None
        per_fold.append(model.feature_importances_)

    importances = np.vstack(per_fold)
    return (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "importance_mean": importances.mean(axis=0),
                "importance_std": importances.std(axis=0),
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
