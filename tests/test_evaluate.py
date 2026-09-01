"""Tests for news_sentiment.market.evaluate.

Metrics, EvalResult and the final refit are covered first; the splitter and the
fold-level guarantees follow, from what was tests/test_evaluate_v2.py before the
two modules were merged.


Three things get tested that a plain "does it run" check wouldn't catch:

- The harness must correctly report *no* skill when there is none (majority-only
  and random-probability dummies) AND correctly report real skill when it's
  actually there (a model trained on a feature that perfectly determines the
  label). Only checking the first direction would pass a harness that's simply
  broken in a way that always looks pessimistic.
- The grouping must never leak a market session across the train/test boundary,
  checked directly against the function under test rather than by eyeballing
  notebook output.
- EvalResult's fold models, predictions, and indices must be the genuine article
  for that fold -- not a shared or stale object, and not silently inconsistent
  with metrics_df/aggregate, which are reshapes of the same fold data rather than
  independently computed.
"""

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from news_sentiment.market.evaluate import (
    EvalResult,
    FoldResult,
    evaluate,
    feature_importance,
    fit_final_model,
    row_grouped_splits,
    weighted_aggregate,
)


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


class RandomClassifier(BaseEstimator, ClassifierMixin):
    """Predicts random probabilities, ignoring X entirely."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        rng = np.random.default_rng(self.random_state)
        p = rng.random(len(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def _make_df(n_days=60, rows_per_day=6, separable=False, seed=0):
    """A synthetic label_and_features.parquet-shaped table.

    separable=True gives a single feature that perfectly determines the label
    (plus tiny noise so it isn't a constant column) -- a positive control the
    harness should recognise as real signal.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n_days, freq="D", tz="UTC")

    rows = []
    for day in dates:
        for _ in range(rows_per_day):
            hour = rng.integers(0, 24)
            ts = day + pd.Timedelta(hours=int(hour), minutes=int(rng.integers(0, 60)))
            if separable:
                x = rng.normal(loc=0, scale=1)
                label = 1 if x > 0 else -1
                x = x + rng.normal(scale=0.01)  # a hair of noise, not a constant
            else:
                x = rng.normal()
                label = rng.choice([-1, 1])
            rows.append({"timestamp_utc": ts, "x": x, "label_direction": label})

    frame = pd.DataFrame(rows).sort_values("timestamp_utc").reset_index(drop=True)
    # No market calendar behind synthetic data, so the normalised day stands in
    # for the session the row is labelled against.
    frame["session_open"] = frame["timestamp_utc"].dt.normalize()
    return frame


# evaluate() -- must report both "no skill" and "real skill" correctly
# --------------------------------------------------------------------------- #


class TestEvaluateNoSkill:
    def test_majority_classifier_matches_its_own_baseline(self):
        df = _make_df()
        result = evaluate(MajorityClassifier, df, ["x"], n_splits=5)
        metrics = result.metrics_df
        assert np.allclose(metrics["accuracy"], metrics["majority_baseline_accuracy"])
        assert not metrics["beats_baseline"].any()

    def test_majority_classifier_has_no_discordant_pairs(self):
        # Model IS the baseline predictor, so McNemar has nothing to compare.
        df = _make_df()
        result = evaluate(MajorityClassifier, df, ["x"], n_splits=5)
        metrics = result.metrics_df
        assert metrics["mcnemar_p_value"].isna().all()
        assert not metrics["significantly_beats_baseline"].any()

    def test_random_classifier_auc_near_half(self):
        df = _make_df()
        result = evaluate(RandomClassifier, df, ["x"], n_splits=5)
        assert abs(result.metrics_df["auc"].mean() - 0.5) < 0.1


class TestEvaluateRealSkill:
    def _model_factory(self):
        return lambda: LGBMClassifier(n_estimators=50, max_depth=3, random_state=42, verbose=-1)

    def test_separable_feature_is_detected(self):
        df = _make_df(n_days=80, rows_per_day=8, separable=True)
        result = evaluate(self._model_factory(), df, ["x"], n_splits=5)
        metrics = result.metrics_df

        assert (metrics["accuracy"] > 0.9).all()
        assert (metrics["auc"] > 0.9).all()
        assert metrics["beats_baseline"].all()

    def test_separable_feature_is_significant(self):
        df = _make_df(n_days=80, rows_per_day=8, separable=True)
        result = evaluate(self._model_factory(), df, ["x"], n_splits=5)
        assert result.metrics_df["significantly_beats_baseline"].all()


# --------------------------------------------------------------------------- #
# EvalResult -- fold models, predictions, and indices must be the real article
# --------------------------------------------------------------------------- #


class TestEvalResultShape:
    def test_one_fold_result_per_split(self):
        df = _make_df()
        result = evaluate(MajorityClassifier, df, ["x"], n_splits=5)
        assert len(result.folds) == 5
        assert [f.fold for f in result.folds] == [1, 2, 3, 4, 5]

    def test_metrics_df_matches_fold_metrics_exactly(self):
        df = _make_df()
        result = evaluate(MajorityClassifier, df, ["x"], n_splits=5)
        for fold_result, row in zip(result.folds, result.metrics_df.itertuples(), strict=True):
            assert row.accuracy == fold_result.metrics["accuracy"]
            assert row.fold == fold_result.fold

    def test_aggregate_is_mean_and_std_of_metrics_df(self):
        df = _make_df()
        result = evaluate(MajorityClassifier, df, ["x"], n_splits=5)
        expected_mean_accuracy = result.metrics_df["accuracy"].mean()
        assert np.isclose(result.aggregate.loc["mean", "accuracy"], expected_mean_accuracy)

    def test_models_property_returns_one_model_per_fold_in_order(self):
        df = _make_df()
        result = evaluate(MajorityClassifier, df, ["x"], n_splits=5)
        assert result.models == [f.model for f in result.folds]
        assert len(result.models) == 5

    def test_fold_models_are_distinct_fresh_instances(self):
        df = _make_df()
        result = evaluate(MajorityClassifier, df, ["x"], n_splits=5)
        models = result.models
        assert len({id(m) for m in models}) == len(models)

    def test_train_and_test_idx_are_positions_into_the_sorted_data(self):
        df = _make_df()
        sorted_df = df.sort_values("timestamp_utc").reset_index(drop=True)
        result = evaluate(MajorityClassifier, df, ["x"], n_splits=5)

        for fold_result in result.folds:
            train_dates = sorted_df.loc[fold_result.train_idx, "timestamp_utc"]
            test_dates = sorted_df.loc[fold_result.test_idx, "timestamp_utc"]
            assert train_dates.max() < test_dates.min()

    def test_train_and_test_idx_never_overlap(self):
        df = _make_df()
        result = evaluate(MajorityClassifier, df, ["x"], n_splits=5)
        for fold_result in result.folds:
            assert set(fold_result.train_idx).isdisjoint(set(fold_result.test_idx))

    def test_y_pred_and_y_proba_have_the_right_shape(self):
        df = _make_df()
        result = evaluate(RandomClassifier, df, ["x"], n_splits=5)
        for fold_result in result.folds:
            n_test = len(fold_result.test_idx)
            assert fold_result.y_pred.shape == (n_test,)
            assert fold_result.y_proba.shape == (n_test, 2)

    def test_metric_columns_present_and_bounded(self):
        df = _make_df()
        result = evaluate(RandomClassifier, df, ["x"], n_splits=5)
        metrics = result.metrics_df
        for col in ["accuracy", "precision", "recall", "f1", "auc", "majority_baseline_accuracy"]:
            assert metrics[col].between(0, 1).all(), col
        assert (metrics["brier_score"] >= 0).all()
        assert (metrics["log_loss"] >= 0).all()


# --------------------------------------------------------------------------- #
# fit_final_model -- no split, every row, no score
# --------------------------------------------------------------------------- #


class TestFitFinalModel:
    def test_fits_on_the_full_dataset(self):
        df = _make_df(n_days=50, rows_per_day=4)
        # A majority classifier trained on the full df must reflect the full
        # df's majority class.
        fitted = fit_final_model(MajorityClassifier, df, ["x"])
        expected_majority = df["label_direction"].mode().iloc[0]
        assert fitted.majority_ == expected_majority

    def test_uses_every_row_not_a_subset(self):
        class RowCountingClassifier(BaseEstimator, ClassifierMixin):
            """Records exactly how many rows it was trained on."""

            def fit(self, X, y):
                self.n_samples_seen_ = len(X)
                self.classes_ = np.unique(y)
                return self

            def predict(self, X):
                return np.full(len(X), self.classes_[0])

            def predict_proba(self, X):
                return np.tile(np.eye(len(self.classes_))[0], (len(X), 1))

        df = _make_df(n_days=50, rows_per_day=4)
        fitted = fit_final_model(RowCountingClassifier, df, ["x"])
        assert fitted.n_samples_seen_ == len(df)

    def test_returned_model_is_ready_to_predict(self):
        df = _make_df(n_days=80, rows_per_day=6, separable=True)

        def model_factory():
            return LGBMClassifier(n_estimators=50, max_depth=3, random_state=42, verbose=-1)

        fitted = fit_final_model(model_factory, df, ["x"])

        new_data = _make_df(n_days=10, rows_per_day=6, separable=True, seed=999)
        preds = fitted.predict(new_data[["x"]])
        accuracy = (preds == new_data["label_direction"]).mean()
        assert accuracy > 0.9


# --------------------------------------------------------------------------- #
# feature_importance
# --------------------------------------------------------------------------- #


class TestFeatureImportance:
    def test_ranks_the_informative_feature_highest(self):
        rng = np.random.default_rng(0)
        df = _make_df(n_days=80, rows_per_day=8, separable=True)
        df["noise"] = rng.normal(size=len(df))

        def model_factory():
            return LGBMClassifier(n_estimators=50, max_depth=3, random_state=42, verbose=-1)

        result = evaluate(model_factory, df, ["x", "noise"], n_splits=5)
        importances = feature_importance(result, ["x", "noise"])

        ranked = importances.set_index("feature")["importance_mean"]
        assert ranked["x"] > ranked["noise"]

    def test_returns_none_for_a_model_without_feature_importances(self):
        df = _make_df()
        result = evaluate(MajorityClassifier, df, ["x"], n_splits=3)
        assert feature_importance(result, ["x"]) is None


# --------------------------------------------------------------------------- #
# row_grouped_splits: session-grouped, row-count-sized folds
#
# The splitter must keep every group whole, expand the train window, and size
# folds by row count rather than by group count, which is what 3.2 found the
# old day-count splitter failing at on a bursty corpus. weighted_aggregate is
# checked against a hand-computed weighted mean, not trusted to look right.
# --------------------------------------------------------------------------- #


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
    frame = pd.DataFrame(rows).sort_values("timestamp_utc").reset_index(drop=True)
    # No market calendar behind synthetic data, so the normalised day stands in for
    # the session that evaluate() groups on.
    frame["session_open"] = frame["timestamp_utc"].dt.normalize()
    return frame


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
    frame = pd.DataFrame(rows).sort_values("timestamp_utc").reset_index(drop=True)
    frame["session_open"] = frame["timestamp_utc"].dt.normalize()
    return frame


# --------------------------------------------------------------------------- #
# row_grouped_splits -- same guarantees as day_grouped_splits ...
# --------------------------------------------------------------------------- #


class TestRowGroupedSplitsGuarantees:
    def test_no_day_appears_in_both_train_and_test(self):
        dates = _make_bursty_df()["timestamp_utc"]
        for train_mask, test_mask in row_grouped_splits(dates.dt.normalize(), n_splits=5):
            train_days = set(dates[train_mask].dt.normalize())
            test_days = set(dates[test_mask].dt.normalize())
            assert train_days.isdisjoint(test_days)

    def test_test_starts_strictly_after_train_ends(self):
        dates = _make_bursty_df()["timestamp_utc"]
        for train_mask, test_mask in row_grouped_splits(dates.dt.normalize(), n_splits=5):
            assert dates[test_mask].min() > dates[train_mask].max()

    def test_train_window_expands_across_folds(self):
        dates = _make_bursty_df()["timestamp_utc"]
        train_sizes = [tr.sum() for tr, _ in row_grouped_splits(dates.dt.normalize(), n_splits=5)]
        assert train_sizes == sorted(train_sizes)

    def test_train_and_test_never_overlap(self):
        dates = _make_bursty_df()["timestamp_utc"]
        for train_mask, test_mask in row_grouped_splits(dates.dt.normalize(), n_splits=5):
            assert not (train_mask & test_mask).any()

    def test_a_multi_row_day_is_never_split(self):
        dates = _make_uniform_df(n_days=40, rows_per_day=10)["timestamp_utc"]
        for train_mask, test_mask in row_grouped_splits(dates.dt.normalize(), n_splits=4):
            for day in dates[train_mask].dt.normalize().unique():
                day_mask = dates.dt.normalize() == day
                assert not (day_mask & test_mask).any()

    def test_yields_at_most_the_requested_number_of_folds(self):
        dates = _make_uniform_df()["timestamp_utc"]
        folds = list(row_grouped_splits(dates.dt.normalize(), n_splits=4))
        assert len(folds) == 4


# --------------------------------------------------------------------------- #
# ... but sized by row count, which is the actual fix
# --------------------------------------------------------------------------- #


class TestRowGroupedSplitsSizing:
    def test_test_fold_sizes_close_to_target_on_uniform_data(self):
        dates = _make_uniform_df(n_days=60, rows_per_day=6)["timestamp_utc"]
        n_splits = 5
        target = len(dates) / (n_splits + 1)
        for _, test_mask in row_grouped_splits(dates.dt.normalize(), n_splits=n_splits):
            assert abs(test_mask.sum() - target) < target * 0.5

    def test_test_fold_sizes_stay_even_on_bursty_data(self):
        # The reason this module exists. On the shape 3.2 found in the real corpus,
        # the old day-count splitter produced test folds from 176 to 529 rows, a
        # >4x range, because it fixed the number of days rather than the amount of
        # data. Sizing by row count has to hold folds close to the target instead.
        frame = _make_bursty_df()
        n_splits = 5
        target = len(frame) / (n_splits + 1)

        sizes = [
            int(test.sum())
            for _, test in row_grouped_splits(frame["session_open"], n_splits=n_splits)
        ]
        assert max(sizes) - min(sizes) < target * 0.5
        for size in sizes:
            assert abs(size - target) < target * 0.5

    def test_test_fold_sizes_never_zero(self):
        dates = _make_bursty_df()["timestamp_utc"]
        for _, test_mask in row_grouped_splits(dates.dt.normalize(), n_splits=5):
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
            for _, test_mask in row_grouped_splits(
                sorted_df["timestamp_utc"].dt.normalize(), n_splits=5
            )
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


# ---------------------------------------------------------------------------
# row_grouped_splits(groups=...): the unit a fold may not divide
# ---------------------------------------------------------------------------


def _session_frame():
    """Three articles per session, where each session spans two calendar days.

    Mirrors the real shape: an article published after one day's close and one
    published before the next day's open anchor to the same market session and
    therefore carry the same label, while sitting on different calendar days.
    """
    rows = []
    for s in range(12):
        base = pd.Timestamp("2025-01-06", tz="UTC") + pd.Timedelta(days=s)
        session = base + pd.Timedelta(days=1)
        rows += [
            {"timestamp_utc": base + pd.Timedelta(hours=21), "session_open": session},
            {"timestamp_utc": base + pd.Timedelta(hours=23), "session_open": session},
            {"timestamp_utc": session + pd.Timedelta(hours=11), "session_open": session},
        ]
    return pd.DataFrame(rows).sort_values("timestamp_utc").reset_index(drop=True)


def test_session_grouping_never_splits_a_session():
    frame = _session_frame()
    for train, test in row_grouped_splits(frame["session_open"], n_splits=3):
        in_train = set(frame.loc[train, "session_open"])
        in_test = set(frame.loc[test, "session_open"])
        assert not (in_train & in_test)


def test_session_grouping_keeps_folds_contiguous_and_ordered():
    """Walk-forward still holds: every test session is later than every train one."""
    frame = _session_frame()
    for train, test in row_grouped_splits(frame["session_open"], n_splits=3):
        assert frame.loc[train, "session_open"].max() < frame.loc[test, "session_open"].min()


def test_evaluate_accepts_a_group_column():
    frame = _session_frame()
    rng = np.random.default_rng(0)
    frame["feature"] = rng.normal(size=len(frame))
    frame["label_direction"] = np.where(rng.normal(size=len(frame)) > 0, 1, -1)

    result = evaluate(
        MajorityClassifier,
        frame,
        ["feature"],
        n_splits=3,
        group_col="session_open",
    )
    assert len(result.folds) == 3


def test_a_session_is_never_split_across_folds():
    frame = _session_frame()
    for train, test in row_grouped_splits(frame["session_open"], n_splits=3):
        assert not set(frame.loc[train, "session_open"]) & set(frame.loc[test, "session_open"])


def test_session_folds_still_walk_forward():
    frame = _session_frame()
    for train, test in row_grouped_splits(frame["session_open"], n_splits=3):
        assert frame.loc[train, "session_open"].max() < frame.loc[test, "session_open"].min()


def test_evaluate_names_the_missing_session_column():
    """The one thing a caller can get wrong now that grouping is not optional."""
    frame = _make_uniform_df(n_days=30, rows_per_day=4).drop(columns=["session_open"])
    with pytest.raises(ValueError, match="align_sessions"):
        evaluate(MajorityClassifier, frame, ["x"], n_splits=3)
