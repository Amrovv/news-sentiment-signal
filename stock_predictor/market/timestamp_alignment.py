import datetime as dt

import pandas as pd


def align_timestamp(pub_datetime: dt.datetime, schedule: pd.DataFrame) -> pd.Timestamp:
    """Given a UTC article publication datetime, return the start of the label
    window: the first NYSE open strictly after publication.

    Raises if `pub_datetime` falls outside `schedule`'s coverage.
    """
    pub_ts = pd.Timestamp(pub_datetime)
    market_openings = schedule["market_open"]
    first_open = market_openings.iloc[0]
    last_open = market_openings.iloc[-1]

    if (pub_ts > last_open) or (pub_ts < first_open):
        raise ValueError(
            f"Publication datetime {pub_ts} is outside the passed schedule \n"
            f"First opening: {first_open} \n"
            f"Last opening: {last_open} \n"
        )

    valid_openings = market_openings[market_openings > pub_ts]
    return valid_openings.iloc[0]


def make_dt(date_str: str, time_str: str) -> dt.datetime:
    """Convenience for building tz-aware (UTC) dt.datetime inputs."""
    return dt.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=dt.UTC
    )
