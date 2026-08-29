"""Tests for stock_predictor.merge.integrity.

The checks exist to catch a key that has stopped meaning one article, so each
one is tested against a table where that has actually happened, not only against
a clean one. `check_one_id_one_article` is the important case: it is the only
thing standing between a reused id and a table that joins one company's
sentiment to another company's return.
"""

import pandas as pd

from stock_predictor.merge import integrity


def _frame(ids, timestamps=None, headlines=None) -> pd.DataFrame:
    frame = pd.DataFrame({"article_id": ids})
    if timestamps is not None:
        frame["timestamp_utc"] = pd.to_datetime(timestamps, utc=True)
    if headlines is not None:
        frame["headline"] = headlines
    return frame


# --- uniqueness ---------------------------------------------------------------


def test_unique_ids_pass():
    check = integrity.check_unique_within(_frame(["a", "b", "c"]), "text")
    assert check.passed
    assert check.status == "pass"


def test_duplicate_ids_fail_and_name_the_offender():
    check = integrity.check_unique_within(_frame(["a", "b", "b"]), "text")
    assert not check.passed
    assert "b" in check.offenders


# --- subset of the corpus ------------------------------------------------------


def test_ids_from_the_corpus_pass():
    corpus = _frame(["a", "b", "c"])
    assert integrity.check_subset_of_corpus(_frame(["a", "b"]), corpus, "text").passed


def test_an_id_absent_from_the_corpus_fails():
    """A table holding ids the corpus never had came from somewhere else."""
    check = integrity.check_subset_of_corpus(_frame(["a", "z"]), _frame(["a", "b"]), "text")
    assert not check.passed
    assert check.offenders == ["z"]


# --- one id, one article -------------------------------------------------------


def test_agreeing_sides_pass():
    times = ["2025-01-01 10:00", "2025-01-02 10:00"]
    left = _frame(["a", "b"], times)
    right = _frame(["a", "b"], times)
    assert integrity.check_one_id_one_article(left, right, "text", "market").passed


def test_a_reused_id_is_caught():
    """The failure the merge cannot survive: same id, different article."""
    left = _frame(["a", "b"], ["2025-01-01 10:00", "2025-01-02 10:00"])
    right = _frame(["a", "b"], ["2025-01-01 10:00", "2025-06-30 18:00"])
    check = integrity.check_one_id_one_article(left, right, "text", "market")
    assert not check.passed
    assert check.offenders == ["b"]


def test_ids_present_on_only_one_side_are_not_a_conflict():
    """A missing partner is filtering, and a different check's business."""
    left = _frame(["a", "b"], ["2025-01-01 10:00", "2025-01-02 10:00"])
    right = _frame(["a"], ["2025-01-01 10:00"])
    assert integrity.check_one_id_one_article(left, right, "text", "market").passed


def test_the_check_is_skipped_rather_than_faked_without_a_witness():
    left = _frame(["a"])
    right = _frame(["a"])
    check = integrity.check_one_id_one_article(left, right, "text", "market")
    assert check.passed
    assert "skipped" in check.detail


# --- across tickers ------------------------------------------------------------


def test_shared_ids_agreeing_across_corpora_pass():
    """Finnhub returns one story for several tickers; that is expected."""
    corpora = {
        "AMZN": _frame(["x", "y"], headlines=["Big Tech earnings", "Amazon only"]),
        "NVDA": _frame(["x", "z"], headlines=["Big Tech earnings", "Nvidia only"]),
    }
    check = integrity.check_cross_ticker_ids(corpora)
    assert check.passed
    assert "1 shared ids compared" in check.detail or "1 shared" in check.detail


def test_a_shared_id_meaning_two_articles_is_caught():
    corpora = {
        "AMZN": _frame(["x"], headlines=["Amazon beats expectations"]),
        "NVDA": _frame(["x"], headlines=["Nvidia announces a chip"]),
    }
    check = integrity.check_cross_ticker_ids(corpora)
    assert not check.passed
    assert check.offenders


def test_corpora_with_no_overlap_pass():
    corpora = {
        "AMZN": _frame(["x"], headlines=["a"]),
        "NVDA": _frame(["y"], headlines=["b"]),
    }
    assert integrity.check_cross_ticker_ids(corpora).passed


# --- the expected gap ----------------------------------------------------------


def test_gap_separates_the_two_directions():
    gap = integrity.describe_gap({"a", "b"}, {"b", "c"}, "text", "market")
    assert gap["shared"] == 1
    assert gap["only_left"] == ["a"]
    assert gap["only_right"] == ["c"]
