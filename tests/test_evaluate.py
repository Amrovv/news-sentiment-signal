"""Tests for stock_predictor.market.evaluate.

Three things get tested that a plain "does it run" check wouldn't catch:

- The harness must correctly report *no* skill when there is none (majority-only
  and random-probability dummies) AND correctly report real skill when it's
  actually there (a model trained on a feature that perfectly determines the
  label). Only checking the first direction would pass a harness that's simply
  broken in a way that always looks pessimistic.
- The day-grouping must never leak a calendar day across the train/test boundary,
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
from sklearn.base import BaseEstimator, ClassifierMixin

from stock_predictor.market.evaluate import (
    day_grouped_splits,
    evaluate,
    feature_importance,
    fit_final_model,
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

    return pd.DataFrame(rows).sort_values("timestamp_utc").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# day_grouped_splits
# --------------------------------------------------------------------------- #


class TestDayGroupedSplits:
    def test_no_day_appears_in_both_train_and_test(self):
        df = _make_df()
        dates = df["timestamp_utc"]
        for train_mask, test_mask in day_grouped_splits(dates, n_splits=5):
            train_days = set(dates[train_mask].dt.normalize())
            test_days = set(dates[test_mask].dt.normalize())
            assert train_days.isdisjoint(test_days)

    def test_test_starts_strictly_after_train_ends(self):
        df = _make_df()
        dates = df["timestamp_utc"]
        for train_mask, test_mask in day_grouped_splits(dates, n_splits=5):
            assert dates[test_mask].min() > dates[train_mask].max()

    def test_train_window_expands_across_folds(self):
        df = _make_df()
        dates = df["timestamp_utc"]
        train_sizes = [tr.sum() for tr, _ in day_grouped_splits(dates, n_splits=5)]
        assert train_sizes == sorted(train_sizes)

    def test_yields_requested_number_of_folds(self):
        df = _make_df()
        dates = df["timestamp_utc"]
        folds = list(day_grouped_splits(dates, n_splits=4))
        assert len(folds) == 4

    def test_a_multi_row_day_is_never_split(self):
        # Every day in _make_df() has rows_per_day rows; confirm none of them
        # straddle a fold boundary for any fold.
        df = _make_df(n_days=40, rows_per_day=10)
        dates = df["timestamp_utc"]
        for train_mask, test_mask in day_grouped_splits(dates, n_splits=4):
            for day in dates[train_mask].dt.normalize().unique():
                day_mask = dates.dt.normalize() == day
                assert not (day_mask & test_mask).any()


# --------------------------------------------------------------------------- #
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
