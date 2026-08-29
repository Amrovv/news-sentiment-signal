"""Tests for stock_predictor.merge.merge.

Covers the join itself and the accounting around it: that the two expected kinds
of drop are reported separately and correctly, that shared columns are not
duplicated into the merged table, and that the pooled table is keyed on
`(article_id, ticker)` rather than on `article_id` alone.
"""

import pandas as pd
import pytest

from stock_predictor.merge import merge as merge_module


def _text(ids, ticker="TSLA") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": ids,
            "ticker": ticker,
            "timestamp_utc": pd.to_datetime(
                [f"2025-01-{i + 1:02d} 12:00" for i in range(len(ids))], utc=True
            ),
            "fus_conf_graft_floor_mean": [0.1 * i for i in range(len(ids))],
        }
    )


def _market(ids, ticker="TSLA") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": ids,
            "ticker": ticker,
            "timestamp_utc": pd.to_datetime(
                [f"2025-01-{i + 1:02d} 12:00" for i in range(len(ids))], utc=True
            ),
            "abnormal_return_1d": [0.01 * i for i in range(len(ids))],
            "label_direction": [1 if i % 2 else -1 for i in range(len(ids))],
        }
    )


def test_inner_join_keeps_the_intersection():
    result = merge_module.merge_ticker(_text(["a", "b", "c"]), _market(["a", "b"]), "TSLA")
    assert len(result.merged) == 2
    assert set(result.merged["article_id"]) == {"a", "b"}


def test_the_two_drop_reasons_are_reported_separately():
    """text-only means no price bar; market-only means no target mention."""
    result = merge_module.merge_ticker(_text(["a", "b"]), _market(["b", "c"]), "TSLA")
    assert result.text_only == ["a"]
    assert result.market_only == ["c"]


def test_shared_columns_are_not_duplicated():
    """A `ticker_y` column is something a model could be trained on by mistake."""
    result = merge_module.merge_ticker(_text(["a"]), _market(["a"]), "TSLA")
    assert "ticker_x" not in result.merged.columns
    assert "ticker_y" not in result.merged.columns
    assert list(result.merged.columns).count("ticker") == 1


def test_merged_table_is_sorted_by_time():
    result = merge_module.merge_ticker(_text(["a", "b", "c"]), _market(["a", "b", "c"]), "TSLA")
    assert result.merged["timestamp_utc"].is_monotonic_increasing


def test_join_rate_is_against_the_text_side():
    result = merge_module.merge_ticker(_text(["a", "b", "c", "d"]), _market(["a", "b"]), "TSLA")
    assert result.join_rate == pytest.approx(0.5)


def test_checks_run_and_pass_on_clean_input():
    result = merge_module.merge_ticker(_text(["a", "b"]), _market(["a", "b"]), "TSLA")
    assert result.passed
    assert result.failed_checks == []


def test_a_reused_id_surfaces_as_a_failed_check():
    market = _market(["a", "b"])
    market.loc[1, "timestamp_utc"] = pd.Timestamp("2025-09-09 12:00", tz="UTC")
    result = merge_module.merge_ticker(_text(["a", "b"]), market, "TSLA")
    assert not result.passed
    assert any("one article" in check.name for check in result.failed_checks)


def test_duplicate_ids_raise_rather_than_fan_out():
    """A one-to-many join would silently multiply rows; pandas must refuse."""
    text = pd.concat([_text(["a"]), _text(["a"])], ignore_index=True)
    with pytest.raises(Exception, match="unique|one_to_one|merge keys"):
        merge_module.merge_ticker(text, _market(["a"]), "TSLA")


# --- pooling -------------------------------------------------------------------


def _result(ticker, ids):
    return merge_module.merge_ticker(_text(ids, ticker), _market(ids, ticker), ticker)


def test_pooled_table_stacks_every_ticker():
    results = {"TSLA": _result("TSLA", ["a", "b"]), "NVDA": _result("NVDA", ["b", "c"])}
    pooled = merge_module.pool(results)
    assert len(pooled) == 4


def test_pooled_key_is_article_id_and_ticker():
    """`b` appears under both tickers; that is legitimate, and the pair is unique."""
    results = {"TSLA": _result("TSLA", ["a", "b"]), "NVDA": _result("NVDA", ["b", "c"])}
    pooled = merge_module.pool(results)
    assert pooled["article_id"].duplicated().any()
    assert not pooled.duplicated(["article_id", "ticker"]).any()


def test_pooling_nothing_raises():
    with pytest.raises(ValueError, match="Nothing to pool"):
        merge_module.pool({})


def test_overlap_summary_counts_shared_articles():
    results = {
        "TSLA": _result("TSLA", ["a", "b", "c"]),
        "NVDA": _result("NVDA", ["b", "c"]),
        "AAPL": _result("AAPL", ["z"]),
    }
    overlap = merge_module.overlap_summary(results, "TSLA")
    by_ticker = overlap.set_index("other_ticker")["shared_articles"].to_dict()
    assert by_ticker == {"NVDA": 2, "AAPL": 0}


def test_overlap_summary_excludes_the_ticker_itself():
    results = {"TSLA": _result("TSLA", ["a"]), "NVDA": _result("NVDA", ["a"])}
    overlap = merge_module.overlap_summary(results, "TSLA")
    assert "TSLA" not in set(overlap["other_ticker"])
