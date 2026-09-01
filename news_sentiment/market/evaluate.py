"""The shared walk-forward evaluation harness every model in this project runs through.

Three key points:

1. **Splitting.** Walk-forward CV is several train/test splits, each fold retraining
   on a larger expanding window and testing on the block right after it. Fitting and
   splitting live together in `evaluate()` rather than being left for each model
   script to reimplement and potentially get subtly wrong (a random shuffle leaks
   the future, a plain row-level split can divide rows sharing an answer between
   train and test; see `row_grouped_splits`).
2. **What is kept whole.** Rows sharing a label and market features must not be
   split across a fold boundary, or a model is scored on rows whose answer sibling
   rows already told it. That group is the market session, which is what
   `row_grouped_splits` groups on (grouping by calendar day instead cuts across
   sessions one way and merges them the other).
3. **What "beats the baseline" means.** A model a hair better than the majority
   class on a 200-row fold may just be noise, so `evaluate()` reports a paired
   significance test (McNemar's), not just an accuracy comparison.

Flow: define model_factory (a zero-arg callable returning a fresh, unfitted model
each call, so a walk-forward pass gets a genuinely fresh model per fold and any
callable works, not just an sklearn BaseEstimator); run evaluate() across
walk-forward folds to get an EvalResult holding every fold's model, predictions,
indices and metrics (weighted_aggregate() weights that summary by test size);
tune by calling evaluate() repeatedly with different model_factory variants and
comparing EvalResult.aggregate (evaluate() itself never tunes); finally
fit_final_model() refits once on every row of `data`, no split at all, for the
artifact that ships.

There is deliberately no held-out block anywhere: every walk-forward fold's test
score is already out-of-sample, so the fold metrics from evaluate() are the only
real performance numbers this project reports. The final model is never itself
scored, since scoring it on anything would mean that thing had already been part
of its training data.
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

DEFAULT_TARGET_COL = "label_direction"
DEFAULT_DATE_COL = "timestamp_utc"

# The column evaluate() reads each row's market session from.
DEFAULT_GROUP_COL = "session_open"

__all__ = [
    "EvalResult",
    "FoldResult",
    "evaluate",
    "feature_importance",
    "fit_final_model",
    "row_grouped_splits",
    "weighted_aggregate",
]


def _mcnemar_p_value(y_true, model_pred, baseline_pred) -> float:
    """One-sided McNemar's exact test: is the model right more often than the
    baseline, among the rows where they disagree?

    Only discordant pairs matter (both right or both wrong carries no information).
    b = model right & baseline wrong, c = model wrong & baseline right. Under "no
    real difference", b should be a fair coin flip of b+c, so this is a one-sided
    exact binomial test of b against Binomial(b + c, 0.5). NaN when there's no
    discordance to test, rather than a misleadingly confident p-value.
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
    tidy/aggregate views built from them. metrics_df and aggregate are pure
    reshapes of what's already in .folds, nothing computed from discarded data.
    """

    folds: list = field(default_factory=list)

    @property
    def models(self) -> list:
        """The fold-fitted models, in fold order. None of these is "the" model to
        deploy; see fit_final_model() for that."""
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


def row_grouped_splits(groups: pd.Series, n_splits: int = 5):
    """Yield (train_mask, test_mask) row-level boolean arrays per walk-forward fold,
    each fold's test window sized by row count rather than group count.

    `groups` is the unit no fold may divide, one value per row: the aligned market
    session (`market.labels.align_sessions`). Grouping matters because rows sharing
    a label and market features would otherwise be scored against siblings the
    model trained on. An article published after Friday's close and one published
    before Monday's open anchor to the same session and carry an identical label,
    three calendar days apart; grouping by calendar day instead (this function's
    old behavior) split those apart and leaked the answer on 2-4% of rows.

    Unique groups are laid out in chronological order and cut into `n_splits + 1`
    contiguous chunks so each chunk's row count is as close to
    `len(groups) / (n_splits + 1)` as the group boundaries allow, cutting at
    whichever boundary is nearest the target, never inside a group. Fold `k`'s
    train set is chunks `1..k` combined (the expanding window); its test set is
    chunk `k + 1` alone, mirroring `TimeSeriesSplit` but on group-sized blocks.

    The nearest-boundary choice is greedy per cut, so it is a heuristic, not a
    guarantee (a locally nearer boundary can leave a globally worse fold), but it
    wins on all four of this project's corpora. On a corpus that publishes in
    uneven bursts, row-count sizing is what keeps every fold's test set a
    comparable amount of data.
    """
    groups = groups.reset_index(drop=True)
    group_order = np.sort(groups.unique())
    counts = groups.value_counts().reindex(group_order).to_numpy()
    cum_counts = np.cumsum(counts)
    total_rows = cum_counts[-1]

    n_chunks = n_splits + 1
    targets = [total_rows * k / n_chunks for k in range(1, n_chunks)]

    cut_idx = []
    last_cut = -1
    for target in targets:
        idx = int(np.searchsorted(cum_counts, target, side="left"))
        if idx > 0 and abs(cum_counts[idx - 1] - target) < abs(cum_counts[idx] - target):
            idx -= 1
        idx = max(idx, last_cut + 1)
        idx = min(idx, len(group_order) - 2)
        if idx <= last_cut:
            continue
        cut_idx.append(idx)
        last_cut = idx

    chunk_bounds = [-1, *cut_idx, len(group_order) - 1]
    chunks = [
        group_order[chunk_bounds[i] + 1 : chunk_bounds[i + 1] + 1]
        for i in range(len(chunk_bounds) - 1)
    ]

    for k in range(1, len(chunks)):
        train_groups = set(np.concatenate(chunks[:k]))
        test_groups = set(chunks[k])
        train_mask = groups.isin(train_groups).to_numpy()
        test_mask = groups.isin(test_groups).to_numpy()
        yield train_mask, test_mask


def evaluate(
    model_factory,
    data: pd.DataFrame,
    feature_cols: list,
    label_col: str = DEFAULT_TARGET_COL,
    n_splits: int = 5,
    date_col: str = DEFAULT_DATE_COL,
    group_col: str = DEFAULT_GROUP_COL,
) -> EvalResult:
    """Walk-forward evaluation with folds sized by row count instead of day count
    (see `row_grouped_splits` and the module docstring for why).

    `group_col` names the column holding the unit no fold may divide: the aligned
    market session, which `market.labels.align_sessions` produces.
    """
    data = data.sort_values(date_col).reset_index(drop=True)
    X_all = data[feature_cols]
    y_all = data[label_col]
    dates = data[date_col]
    if group_col not in data.columns:
        raise ValueError(
            f"evaluate() needs a {group_col!r} column holding each row's market session; "
            f"build it with market.labels.align_sessions(data[{date_col!r}], schedule)"
        )

    folds = []
    for fold_num, (train_mask, test_mask) in enumerate(
        row_grouped_splits(data[group_col], n_splits=n_splits), start=1
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


def weighted_aggregate(result: EvalResult, weight_col: str = "n_test") -> pd.DataFrame:
    """Row-count-weighted mean of every numeric metric across folds.

    `EvalResult.aggregate` takes a plain mean across folds, so a 176-row fold and a
    529-row fold each count as "one fold" even though the smaller fold's score
    rests on roughly a third as much evidence. This weights each fold's
    contribution by `weight_col` (test-set row count by default) instead.

    Returns a single-row DataFrame, in the same units and columns as
    `EvalResult.aggregate.loc["mean"]`, so the two are directly comparable.
    """
    metrics_df = result.metrics_df
    weights = metrics_df[weight_col]
    numeric = metrics_df.drop(columns=["fold"]).select_dtypes(include="number")
    weighted_mean = numeric.multiply(weights, axis=0).sum() / weights.sum()
    return weighted_mean.to_frame(name=f"weighted_mean_by_{weight_col}").T


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
    decision. This model is never evaluated, by this function or anything else,
    since scoring it on any data would mean that data had already been part of its
    training set. The walk-forward fold metrics from evaluate() remain the only
    real performance numbers for this project; this call exists purely to produce
    the artifact to save and deploy.
    """
    X, y = data[feature_cols], data[label_col]
    model = model_factory()
    model.fit(X, y)
    return model


def feature_importance(result: EvalResult, feature_cols: list) -> pd.DataFrame | None:
    """Mean +/- std feature importance across the fold models an EvalResult already
    holds, for models that expose `.feature_importances_` after fitting (LightGBM
    and sklearn's tree ensembles). Returns None for models that don't.

    Reuses result.models rather than fitting anything itself, so this needs no
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
