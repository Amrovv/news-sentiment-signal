import pytest

from news_sentiment.market.timestamp_alignment import align_timestamp, make_dt
from news_sentiment.config import NEWS_START_DATE, NEWS_END_DATE
import datetime as dt
import pandas as pd
import pandas_market_calendars as mcal

@pytest.fixture
def make_schedule():
    nyse = mcal.get_calendar("NYSE")

    def make(start_date: str, end_date: str):
        schedule_start = pd.Timestamp(start_date) - dt.timedelta(days=1)
        print(schedule_start)
        schedule_end = pd.Timestamp(end_date) + dt.timedelta(days=10)
        return nyse.schedule(start_date=schedule_start, end_date=schedule_end)

    return make

@pytest.mark.parametrize("input_datetime, expected_timestamp", [
    (make_dt("2026-01-02", "19:00:00"), pd.Timestamp(make_dt("2026-01-05", "14:30:00"))),  # Fri market hours -> Mon open
    (make_dt("2026-01-02", "14:30:00"), pd.Timestamp(make_dt("2026-01-05", "14:30:00"))),  # Fri on open -> Mon open
    (make_dt("2026-01-07", "19:59:00"), pd.Timestamp(make_dt("2026-01-08", "14:30:00"))),  # Wed just before close -> Thu open
    (make_dt("2026-01-07", "20:00:00"), pd.Timestamp(make_dt("2026-01-08", "14:30:00"))),  # Wed on close -> Thu open
    (make_dt("2026-01-04", "13:00:00"), pd.Timestamp(make_dt("2026-01-05", "14:30:00"))),  # Sunday -> Mon open
    (make_dt("2026-01-06", "13:29:00"), pd.Timestamp(make_dt("2026-01-06", "14:30:00"))),  # Tue just before open -> same day
    (make_dt("2026-01-01", "12:00:00"), pd.Timestamp(make_dt("2026-01-02", "14:30:00"))),  # New Year's Day (holiday) -> next day open
    (make_dt(NEWS_START_DATE, "10:00:00"), pd.Timestamp(make_dt(NEWS_START_DATE, "13:30:00"))),  # On day of schedule start
    (make_dt("2026-07-31", "14:00:00"), pd.Timestamp(make_dt("2026-08-03", "13:30:00")))  # On day of schedule end
])

def test_align_timestamp_valid_schedule(input_datetime, expected_timestamp, make_schedule):
    schedule = make_schedule(NEWS_START_DATE, NEWS_END_DATE)
    assert align_timestamp(input_datetime, schedule) == expected_timestamp

@pytest.mark.parametrize("input_datetime", [
    make_dt("2026-01-02", "19:00:00"), # Outside schedule
    make_dt("2026-01-31", "19:00:00"), # Day before schedule start
    make_dt("2026-02-21", "10:00:00"), # Day after schedule end
])

def test_align_timestamp_narrow_schedule(input_datetime, make_schedule):
    schedule = make_schedule("2026-02-01", "2026-02-10")
    with pytest.raises(ValueError):
        align_timestamp(input_datetime, schedule)