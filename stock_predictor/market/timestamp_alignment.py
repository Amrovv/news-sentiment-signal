import datetime as dt

import pandas as pd

from stock_predictor.config import NEWS_END_DATE, NEWS_START_DATE


def align_timestamp(pub_datetime: dt.datetime, schedule: pd.DataFrame) -> pd.Timestamp:
    """
    Given a UTC datetime for article publication dates, return the start of our label window as a pd.Timestamp.

    pub_datetime: datetime containing publication timestamp
    schedule: NYSE schedule for our timeframe
    """

    # Convert datetime to timestamp
    pub_ts = pd.Timestamp(pub_datetime)

    # Get all market openings
    market_openings = schedule["market_open"]

    # Get first and last open
    first_open = schedule["market_open"].iloc[0]
    last_open = schedule["market_open"].iloc[-1]

    # Check that the pub date is within schedule
    if (pub_ts > last_open) or (pub_ts < first_open):
        raise ValueError(
            f"Publication datetime {pub_ts} is outside the passed schedule \n"
            f"First opening: {first_open} \n"
            f"Last opening: {last_open} \n"
        )

    # Filter to openings STRICTLY AFTER publication date
    valid_openings = market_openings[market_openings > pub_ts]

    # Return first valid opening
    return valid_openings.iloc[0]


def make_dt(date_str: str, time_str: str) -> dt.datetime:
    """Convenience for building dt.datetime inputs with timezone awareness"""

    return dt.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=dt.UTC
    )
