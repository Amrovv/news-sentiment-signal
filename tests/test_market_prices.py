"""Tests for news_sentiment.market.prices.

Only the offline half: the pulls themselves need the network and are not
exercised here. `_empty_symbols` is, because it is the guard against the one
failure this module has actually hit -- `yf.download` logging a failed symbol
to stdout while returning that symbol's columns full-width and entirely NaN.
The benchmark is one of the symbols requested, so a missed empty column nulls
every abnormal return in every table downstream without raising anything.
"""

import numpy as np
import pandas as pd
import pytest

from news_sentiment.market.prices import _empty_symbols

FIELDS = ["Close", "High", "Low", "Volume"]


def _frame(data: dict[str, list | None]) -> pd.DataFrame:
    """A yfinance-shaped frame; a symbol mapped to None gets all-NaN columns."""
    index = pd.date_range("2025-01-01", periods=3, name="Date")
    columns, values = [], []
    for symbol, series in data.items():
        for field in FIELDS:
            columns.append((field, symbol))
            values.append([np.nan] * len(index) if series is None else series)
    frame = pd.DataFrame(dict(zip(map(tuple, columns), values, strict=True)), index=index)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns, names=["Price", "Ticker"])
    return frame


def test_no_empty_symbols_when_every_column_has_bars():
    frame = _frame({"TSLA": [1.0, 2.0, 3.0], "SPY": [4.0, 5.0, 6.0]})
    assert _empty_symbols(frame, ["TSLA", "SPY"]) == []


def test_all_nan_symbol_is_reported():
    """The exact shape of the failure seen in practice."""
    frame = _frame({"TSLA": [1.0, 2.0, 3.0], "SPY": None})
    assert _empty_symbols(frame, ["TSLA", "SPY"]) == ["SPY"]


def test_absent_symbol_is_reported():
    frame = _frame({"TSLA": [1.0, 2.0, 3.0]})
    assert _empty_symbols(frame, ["TSLA", "SPY"]) == ["SPY"]


def test_partially_null_symbol_is_accepted():
    """A holiday or a halted session is a gap, not a failed download."""
    frame = _frame({"TSLA": [1.0, np.nan, 3.0], "SPY": [4.0, 5.0, 6.0]})
    assert _empty_symbols(frame, ["TSLA", "SPY"]) == []


def test_only_requested_symbols_are_checked():
    frame = _frame({"TSLA": [1.0, 2.0, 3.0], "SPY": [4.0, 5.0, 6.0], "KO": None})
    assert _empty_symbols(frame, ["TSLA", "SPY"]) == []


@pytest.mark.parametrize("empty", [["SPY"], ["TSLA", "SPY"]])
def test_every_empty_symbol_is_listed_not_just_the_first(empty):
    data = {s: None for s in empty}
    data.update({s: [1.0, 2.0, 3.0] for s in {"TSLA", "SPY"} - set(empty)})
    assert sorted(_empty_symbols(_frame(data), ["TSLA", "SPY"])) == sorted(empty)
