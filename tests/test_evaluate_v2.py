"""Tests for stock_predictor.market.evaluate_v2.

`evaluate.py`'s test suite (`tests/test_evaluate.py`) already covers every metric,
McNemar significance, and EvalResult's shape -- none of that changed here, since
`evaluate_v2.evaluate()` reuses `_score`, `FoldResult`, and `EvalResult` directly from
`evaluate.py`. What's new, and what needs its own coverage, is `row_grouped_splits`:
it must still honor the guarantee `day_grouped_splits` already had (no day split
across train/test, expanding train window), and it must actually fix the problem
`day_grouped_splits` had on a bursty corpus -- much more even test-fold sizes -- which
is the whole reason this module exists (`notebooks/modelling/3.2`); and
`weighted_aggregate`, which needs to be checked against a hand-computed weighted mean
directly, not just trusted to "look reasonable."
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from stock_predictor.market.evaluate import EvalResult, FoldResult, day_grouped_splits
from stock_predictor.market.evaluate_v2 import evaluate, row_grouped_splits, weighted_aggregate


class MajorityClassifier(BaseEstimator, ClassifierMixin):
    """Always predicts the majority class seen in training."""

    def fit(self, X, y):
        values, counts = np.unique(y, return_counts=True)
        self.classes_ = values
        self.majority_ = values[np.argmax(counts)]
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_)

    def predict_proba(self, X):
        proba = np.zeros((len(X), len(self.classes_)))
        proba[:, list(self.classes_).index(self.majority_)] = 1.0
        return proba


def _make_uniform_df(n_days=60, rows_per_day=6, seed=0):
    """A synthetic table with an even number of rows per day -- a corpus with no
    burstiness at all, the easy case both splitters should handle identically well.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n_days, freq="D", tz="UTC")
    rows = []
    for day in dates:
        for _ in range(rows_per_day):
            hour = rng.integers(0, 24)
            ts = day + pd.Timedelta(hours=int(hour), minutes=int(rng.integers(0, 60)))
            rows.append(
                {"timestamp_utc": ts, "x": rng.normal(), "label_direction": rng.choice([-1, 1])}
            )
    return pd.DataFrame(rows).sort_values("timestamp_utc").reset_index(drop=True)


def _make_bursty_df(seed=0):
    """A synthetic table that reproduces the shape `3.2` found in the real corpus:
    a handful of active days, clustered into bursts of very different sizes, most of
    the calendar span carrying nothing at all. This is the case `row_grouped_splits`
    exists to fix and `day_grouped_splits` is known to handle badly.
    """
    rng = np.random.default_rng(seed)
    burst_sizes = [23, 8, 13, 18, 16, 7, 21, 30, 3, 4, 11, 32, 36, 5, 67, 6, 30, 41, 45, 27, 30]
    rows = []
    day_offset = 0
    for burst_len in burst_sizes:
        n_active_days = max(1, burst_len // 15)
        for _ in range(burst_len):
            day = day_offset + int(rng.integers(0, n_active_days))
            ts = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(
                days=day, hours=int(rng.integers(0, 24))
            )
            rows.append(
                {"timestamp_utc": ts, "x": rng.normal(), "label_direction": rng.choice([-1, 1])}
            )
        day_offset += n_active_days + int(rng.integers(5, 25))  # a gap between bursts
    return pd.DataFrame(rows).sort_values("timestamp_utc").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# row_grouped_splits -- same guarantees as day_grouped_splits ...
# --------------------------------------------------------------------------- #


class TestRowGroupedSplitsGuarantees:
    def test_no_day_appears_in_both_train_and_test(self):
        dates = _make_bursty_df()["timestamp_utc"]
        for train_mask, test_mask in row_grouped_splits(dates, n_splits=5):
            train_days = set(dates[train_mask].dt.normalize())
            test_days = set(dates[test_mask].dt.normalize())
            assert train_days.isdisjoint(test_days)

    def test_test_starts_strictly_after_train_ends(self):
        dates = _make_bursty_df()["timestamp_utc"]
        for train_mask, test_mask in row_grouped_splits(dates, n_splits=5):
            assert dates[test_mask].min() > dates[train_mask].max()

    def test_train_window_expands_across_folds(self):
        dates = _make_bursty_df()["timestamp_utc"]
        train_sizes = [tr.sum() for tr, _ in row_grouped_splits(dates, n_splits=5)]
        assert train_sizes == sorted(train_sizes)

    def test_train_and_test_never_overlap(self):
        dates = _make_bursty_df()["timestamp_utc"]
        for train_mask, test_mask in row_grouped_splits(dates, n_splits=5):
            assert not (train_mask & test_mask).any()

    def test_a_multi_row_day_is_never_split(self):
        dates = _make_uniform_df(n_days=40, rows_per_day=10)["timestamp_utc"]
        for train_mask, test_mask in row_grouped_splits(dates, n_splits=4):
            for day in dates[train_mask].dt.normalize().unique():
                day_mask = dates.dt.normalize() == day
                assert not (day_mask & test_mask).any()

    def test_yields_at_most_the_requested_number_of_folds(self):
        dates = _make_uniform_df()["timestamp_utc"]
        folds = list(row_grouped_splits(dates, n_splits=4))
        assert len(folds) == 4


# --------------------------------------------------------------------------- #
# ... but sized by row count, which is the actual fix
# --------------------------------------------------------------------------- #


class TestRowGroupedSplitsSizing:
    def test_test_fold_sizes_close_to_target_on_uniform_data(self):
        dates = _make_uniform_df(n_days=60, rows_per_day=6)["timestamp_utc"]
        n_splits = 5
        target = len(dates) / (n_splits + 1)
        for _, test_mask in row_grouped_splits(dates, n_splits=n_splits):
            assert abs(test_mask.sum() - target) < target * 0.5

    def test_test_fold_sizes_are_far_more_even_than_day_grouped_on_bursty_data(self):
        # This is the whole point of the module: on the bursty shape 3.2 found in the
        # real corpus, day_grouped_splits produced wildly uneven test folds (176 to
        # 529 rows, a >4x range) because it fixes day *count*, not row count.
        # row_grouped_splits must do substantially better on the same data.
        dates = _make_bursty_df()["timestamp_utc"]

        day_sizes = [test_mask.sum() for _, test_mask in day_grouped_splits(dates, n_splits=5)]
        row_sizes = [test_mask.sum() for _, test_mask in row_grouped_splits(dates, n_splits=5)]

        day_spread = np.std(day_sizes) / np.mean(day_sizes)
        row_spread = np.std(row_sizes) / np.mean(row_sizes)
        assert row_spread < day_spread * 0.5

    def test_test_fold_sizes_never_zero(self):
        dates = _make_bursty_df()["timestamp_utc"]
        for _, test_mask in row_grouped_splits(dates, n_splits=5):
            assert test_mask.sum() > 0


# --------------------------------------------------------------------------- #
# evaluate() -- correctly wired to row_grouped_splits, everything else reused as-is
# --------------------------------------------------------------------------- #


class TestEvaluate:
    def test_majority_classifier_matches_its_own_baseline(self):
        # Same sanity check test_evaluate.py runs against evaluate.evaluate() --
        # confirms the metrics/EvalResult plumbing carried over correctly, not just
        # that the module imports without error.
        df = _make_bursty_df()
        result = evaluate(MajorityClassifier, df, ["x"], n_splits=5)
        metrics = result.metrics_df
        assert np.allclose(metrics["accuracy"], metrics["majority_baseline_accuracy"])

    def test_fold_test_sizes_match_row_grouped_splits_directly(self):
        df = _make_bursty_df()
        sorted_df = df.sort_values("timestamp_utc").reset_index(drop=True)
        result = evaluate(MajorityClassifier, df, ["x"], n_splits=5)

        expected_sizes = [
            test_mask.sum()
            for _, test_mask in row_grouped_splits(sorted_df["timestamp_utc"], n_splits=5)
        ]
        actual_sizes = [len(f.test_idx) for f in result.folds]
        assert actual_sizes == expected_sizes

    def test_test_fold_sizes_are_even_end_to_end(self):
        df = _make_bursty_df()
        result = evaluate(MajorityClassifier, df, ["x"], n_splits=5)
        sizes = result.metrics_df["n_test"]
        assert sizes.std() / sizes.mean() < 0.3


# --------------------------------------------------------------------------- #
# weighted_aggregate -- a fold's influence on the mean should match how much data
# it was actually scored on, not count as "one fold" regardless of size
# --------------------------------------------------------------------------- #


def _fold(fold: int, n_test: int, accuracy: float) -> FoldResult:
    """A minimal FoldResult carrying only what weighted_aggregate reads (metrics_df),
    so its arithmetic can be checked in isolation from evaluate()/model fitting.
    """
    return FoldResult(
        fold=fold,
        model=None,
        train_idx=np.array([]),
        test_idx=np.arange(n_test),
        y_pred=np.array([]),
        y_proba=np.array([]),
        metrics={"n_train": 100, "n_test": n_test, "accuracy": accuracy},
    )


class TestWeightedAggregate:
    def test_matches_a_hand_computed_weighted_mean(self):
        result = EvalResult(folds=[_fold(1, 100, 0.9), _fold(2, 300, 0.5)])
        agg = weighted_aggregate(result)
        expected = (100 * 0.9 + 300 * 0.5) / (100 + 300)
        assert np.isclose(agg.loc["weighted_mean_by_n_test", "accuracy"], expected)

    def test_differs_from_the_unweighted_mean_on_uneven_folds(self):
        # The whole point: a big, weak fold should pull the mean toward it harder
        # than a plain average would let it.
        result = EvalResult(folds=[_fold(1, 100, 0.9), _fold(2, 300, 0.5)])
        unweighted = result.aggregate.loc["mean", "accuracy"]
        weighted = weighted_aggregate(result).loc["weighted_mean_by_n_test", "accuracy"]
        assert not np.isclose(unweighted, weighted)
        assert weighted < unweighted

    def test_matches_the_unweighted_mean_when_folds_are_equal_size(self):
        # Equal weights everywhere collapses to the same thing as a plain mean --
        # this is the case row_grouped_splits aims to make the common one.
        result = EvalResult(folds=[_fold(1, 200, 0.9), _fold(2, 200, 0.5)])
        unweighted = result.aggregate.loc["mean", "accuracy"]
        weighted = weighted_aggregate(result).loc["weighted_mean_by_n_test", "accuracy"]
        assert np.isclose(unweighted, weighted)

    def test_respects_a_custom_weight_column(self):
        folds = [_fold(1, 100, 0.9), _fold(2, 300, 0.5)]
        folds[0].metrics["n_train"] = 900
        folds[1].metrics["n_train"] = 100
        result = EvalResult(folds=folds)

        by_test = weighted_aggregate(result, weight_col="n_test").loc[
            "weighted_mean_by_n_test", "accuracy"
        ]
        by_train = weighted_aggregate(result, weight_col="n_train").loc[
            "weighted_mean_by_n_train", "accuracy"
        ]
        assert not np.isclose(by_test, by_train)

    def test_end_to_end_with_evaluate(self):
        df = _make_bursty_df()
        result = evaluate(MajorityClassifier, df, ["x"], n_splits=5)
        agg = weighted_aggregate(result)
        assert list(agg.index) == ["weighted_mean_by_n_test"]
        assert "accuracy" in agg.columns
