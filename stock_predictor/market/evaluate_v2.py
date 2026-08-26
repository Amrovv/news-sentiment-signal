"""Iteration 2 of the walk-forward evaluation harness: row-count-grouped splits.

`evaluate.py` (iteration 1) grouped folds by *active-day count*: "the next 15 active
days," the same for every fold. `notebooks/modelling/3.2` found that broke down on this
corpus, because the corpus does not publish continuously -- 92 of 351 calendar days
carry any article at all, clustered into ~13 short bursts of very different sizes. A
fixed day count is not a fixed amount of data when the days themselves aren't
interchangeable: 3.2's 5 folds carried anywhere from 176 to 529 test rows and swung
from 19.9% to 69.4% positive, which made `majority_baseline_accuracy` -- meant to be a
stable ~50% reference -- swing right along with it, and made fold-to-fold accuracy
comparisons unreliable regardless of what model produced them.

This module keeps everything about the harness that 3.2 did *not* implicate --
`model_factory`, `EvalResult`/`FoldResult`, every metric in `_score()`, McNemar
significance, `fit_final_model()`, `feature_importance()` -- imported directly from
`evaluate.py` rather than copied, and makes the two changes 3.2 section 6.2
recommended together: `day_grouped_splits` is replaced by `row_grouped_splits`, which
sizes each fold by *row count* instead of day count, snapped to day boundaries so a day
is still never divided between train and test (that guarantee wasn't the problem; 3.2's
finding was that a fixed *number* of days is the wrong proxy for a fixed *amount* of
data); and `weighted_aggregate()` is added alongside `EvalResult.aggregate`, so a
176-row fold and a 529-row fold don't count equally toward the mean just because each
counts as "one fold." Row-count splitting mostly addresses what the weighting gap was
covering for -- once folds are close to equal size, weighted and unweighted means
should nearly agree -- but "mostly" and "exactly" aren't the same claim, and
`weighted_aggregate()` exists to let that be checked rather than assumed.

This is a separate file rather than an edit to `evaluate.py` so both eras of the
harness stay independently readable: `evaluate.py`'s docstring and `notebooks/
modelling/3.1` still describe and validate exactly the day-grouped harness that
`notebooks/modelling/3.2` actually ran through, and this module documents what changed
and why, rather than silently rewriting history underneath an already-run notebook.
"""

import numpy as np
import pandas as pd

from stock_predictor.market.evaluate import (
    DEFAULT_DATE_COL,
    DEFAULT_TARGET_COL,
    EvalResult,
    FoldResult,
    _score,
    feature_importance,
    fit_final_model,
)

__all__ = [
    "evaluate",
    "feature_importance",
    "fit_final_model",
    "row_grouped_splits",
    "weighted_aggregate",
]


def row_grouped_splits(dates: pd.Series, n_splits: int = 5):
    """Yield (train_mask, test_mask) row-level boolean arrays per walk-forward fold,
    each fold's test window sized by row count rather than day count.

    Unique calendar days are laid out in chronological order and cut into
    `n_splits + 1` contiguous chunks so that each chunk's row count is as close to
    `len(dates) / (n_splits + 1)` as the day boundaries allow -- a cut point is placed
    at the day boundary nearest each target row count, never inside a day, so (like
    `day_grouped_splits`) no day is ever split between train and test. Fold `k`'s
    train set is chunks `1..k` combined (the expanding window); its test set is chunk
    `k + 1` alone. This mirrors what `TimeSeriesSplit` does for individual rows, just
    applied to day-sized blocks sized by the rows inside them instead of by count of
    days.

    On a corpus that publishes in uneven bursts (this one: 92 active days out of 351,
    clustered into ~13 bursts of very different sizes), this is what keeps every
    fold's test set a comparable amount of data -- the property `day_grouped_splits`
    was meant to guarantee but, on this corpus, didn't.
    """
    days = dates.dt.normalize()
    day_order = np.sort(days.unique())
    counts = days.value_counts().reindex(day_order).to_numpy()
    cum_counts = np.cumsum(counts)
    total_rows = cum_counts[-1]

    n_chunks = n_splits + 1
    targets = [total_rows * k / n_chunks for k in range(1, n_chunks)]

    cut_idx = []
    last_cut = -1
    for target in targets:
        idx = int(np.searchsorted(cum_counts, target, side="left"))
        idx = max(idx, last_cut + 1)
        idx = min(idx, len(day_order) - 2)
        if idx <= last_cut:
            continue
        cut_idx.append(idx)
        last_cut = idx

    chunk_bounds = [-1, *cut_idx, len(day_order) - 1]
    chunks = [
        day_order[chunk_bounds[i] + 1 : chunk_bounds[i + 1] + 1]
        for i in range(len(chunk_bounds) - 1)
    ]

    for k in range(1, len(chunks)):
        train_days = set(np.concatenate(chunks[:k]))
        test_days = set(chunks[k])
        train_mask = days.isin(train_days).to_numpy()
        test_mask = days.isin(test_days).to_numpy()
        yield train_mask, test_mask


def evaluate(
    model_factory,
    data: pd.DataFrame,
    feature_cols: list,
    label_col: str = DEFAULT_TARGET_COL,
    n_splits: int = 5,
    date_col: str = DEFAULT_DATE_COL,
) -> EvalResult:
    """`evaluate.evaluate()`, with folds sized by row count instead of day count --
    see `row_grouped_splits` and this module's docstring for why. Same contract,
    same `EvalResult`/`FoldResult` shapes, same metrics; only which rows land in
    which fold changes.
    """
    data = data.sort_values(date_col).reset_index(drop=True)
    X_all = data[feature_cols]
    y_all = data[label_col]
    dates = data[date_col]

    folds = []
    for fold_num, (train_mask, test_mask) in enumerate(
        row_grouped_splits(dates, n_splits=n_splits), start=1
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
    """Row-count-weighted mean of every numeric metric across folds -- `3.2` section
    6.2's other recommended fix, alongside `row_grouped_splits`.

    `EvalResult.aggregate` (`evaluate.py`) takes a plain mean across folds: a 176-row
    fold and a 529-row fold each count as "one fold," even though the smaller fold's
    score rests on roughly a third as much evidence. This weights each fold's
    contribution by `weight_col` (test-set row count by default) instead, so a fold's
    influence on the aggregate matches how much data it was actually scored on.

    Returns a single-row DataFrame, in the same units and columns as
    `EvalResult.aggregate.loc["mean"]`, so the two are directly comparable.
    """
    metrics_df = result.metrics_df
    weights = metrics_df[weight_col]
    numeric = metrics_df.drop(columns=["fold"]).select_dtypes(include="number")
    weighted_mean = numeric.multiply(weights, axis=0).sum() / weights.sum()
    return weighted_mean.to_frame(name=f"weighted_mean_by_{weight_col}").T
