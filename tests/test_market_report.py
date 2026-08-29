"""Tests for stock_predictor.market.report.

The report is generated from the frame it describes, so what matters is that it
cannot silently disagree with that frame: every figure it links has to exist,
every column has to be listed, and the label counts have to be the frame's own.
The charts themselves are only checked for being produced -- their appearance is
not something a test can usefully assert.
"""

import numpy as np
import pandas as pd
import pytest

from stock_predictor.market import labels, report


@pytest.fixture
def final() -> pd.DataFrame:
    """A table shaped exactly like build_market_features' output.

    Two months of daily articles, one deliberately thin so the month-size guard
    has something to exclude.
    """
    rng = np.random.default_rng(0)
    dense = pd.date_range("2025-01-01", "2025-03-31", freq="6h", tz="UTC")
    thin = pd.date_range("2025-04-01", periods=5, freq="D", tz="UTC")
    ts = dense.append(thin)
    n = len(ts)

    returns = rng.normal(0.001, 0.02, n)
    frame = pd.DataFrame(
        {
            "article_id": [f"a{i}" for i in range(n)],
            "ticker": "TSLA",
            "timestamp_utc": ts,
            "abnormal_return_1d": returns,
            "label_direction": np.sign(returns).astype(int),
            "session": rng.choice(["pre-market", "market-hours", "after-hours"], n),
            "news_volume": rng.integers(0, 50, n),
            "days_to_earnings": rng.integers(0, 90, n),
        }
    )
    for column in labels.FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = rng.normal(0, 0.03, n)
    return frame[labels.IDENTITY_COLUMNS + labels.LABEL_COLUMNS + labels.FEATURE_COLUMNS]


@pytest.fixture
def written(final, tmp_path, monkeypatch):
    """Write a real report into tmp_path, figures included."""
    figures = tmp_path / "figures"
    figures.mkdir()
    monkeypatch.setattr(report, "report_figures_dir", lambda section: figures)
    monkeypatch.setattr(report, "report_dir", lambda section: tmp_path)
    path = report.write_market_report(final, "TSLA", n_articles=len(final) + 10)
    return path, path.read_text(encoding="utf-8"), tmp_path


def test_report_is_written_to_the_reports_directory(written):
    path, _, tmp_path = written
    assert path.parent == tmp_path
    assert path.name == "TSLA_market_features_report.md"


def test_every_linked_figure_exists(written):
    """A broken image link is the failure mode a generated report invites."""
    _, text, tmp_path = written
    linked = [
        line.split("](")[1].rstrip(")") for line in text.splitlines() if line.startswith("![")
    ]
    assert linked, "report links no figures at all"
    for rel in linked:
        assert (tmp_path / rel).exists(), f"missing figure: {rel}"


def test_figures_go_beside_the_report_not_into_data(written):
    _, text, _ = written
    assert "figures/" in text
    assert "data/processed" in text  # the table's location is still stated
    assert "](data/" not in text  # but nothing is linked out of it


def test_every_column_is_listed(written, final):
    _, text, _ = written
    for column in labels.LABEL_COLUMNS + labels.FEATURE_COLUMNS:
        assert f"`{column}`" in text


def test_label_counts_match_the_frame(written, final):
    _, text, _ = written
    up = int((final["label_direction"] > 0).sum())
    down = int((final["label_direction"] < 0).sum())
    assert f"{up:,}" in text
    assert f"{down:,}" in text


def test_majority_class_rate_is_the_larger_share(written, final):
    _, text, _ = written
    up = (final["label_direction"] > 0).mean()
    majority = max(up, 1 - up)
    assert f"{majority:.1%}" in text


def test_thin_months_are_excluded_from_the_reported_extremes(written):
    """The 5-article April must not set the spread."""
    _, text, _ = written
    assert f"under {report.MIN_MONTH_ARTICLES} articles" in text
    assert "(2025-04," not in text


def test_no_three_day_column_is_mentioned(written):
    _, text, _ = written
    assert "abnormal_return_3d" not in text
