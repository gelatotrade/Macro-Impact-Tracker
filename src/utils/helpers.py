"""Utility functions for the Macro Impact Tracker."""

from datetime import datetime, timedelta
from typing import Optional, Tuple
import pytz
import pandas as pd
import numpy as np


def convert_timezone(
    dt: datetime,
    from_tz: str = "UTC",
    to_tz: str = "America/New_York"
) -> datetime:
    """
    Convert datetime between timezones.

    Args:
        dt: Input datetime
        from_tz: Source timezone
        to_tz: Target timezone

    Returns:
        Converted datetime
    """
    if dt.tzinfo is None:
        source = pytz.timezone(from_tz)
        dt = source.localize(dt)

    target = pytz.timezone(to_tz)
    return dt.astimezone(target)


def get_trading_hours(
    market: str = "NYSE"
) -> Tuple[datetime, datetime]:
    """
    Get trading hours for a given market.

    Args:
        market: Market identifier (NYSE, LSE, etc.)

    Returns:
        Tuple of (open_time, close_time)
    """
    market_hours = {
        "NYSE": ("09:30", "16:00", "America/New_York"),
        "NASDAQ": ("09:30", "16:00", "America/New_York"),
        "LSE": ("08:00", "16:30", "Europe/London"),
        "TSE": ("09:00", "15:00", "Asia/Tokyo"),
        "FOREX": ("00:00", "23:59", "UTC"),  # 24/5
    }

    if market not in market_hours:
        raise ValueError(f"Unknown market: {market}")

    open_str, close_str, tz_str = market_hours[market]
    tz = pytz.timezone(tz_str)
    today = datetime.now(tz).date()

    open_time = tz.localize(datetime.combine(today, datetime.strptime(open_str, "%H:%M").time()))
    close_time = tz.localize(datetime.combine(today, datetime.strptime(close_str, "%H:%M").time()))

    return open_time, close_time


def calculate_returns(
    prices: pd.Series,
    method: str = "simple"
) -> pd.Series:
    """
    Calculate returns from a price series.

    Args:
        prices: Price series
        method: 'simple' or 'log'

    Returns:
        Returns series
    """
    if method == "simple":
        return prices.pct_change()
    elif method == "log":
        return np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"Unknown method: {method}")


def format_timestamp(
    dt: datetime,
    fmt: str = "%Y-%m-%d %H:%M:%S %Z"
) -> str:
    """
    Format a datetime object to string.

    Args:
        dt: Datetime object
        fmt: Format string

    Returns:
        Formatted string
    """
    return dt.strftime(fmt)


def get_event_window(
    event_time: datetime,
    pre_minutes: int = 30,
    post_minutes: int = 120
) -> Tuple[datetime, datetime]:
    """
    Get the time window around an event.

    Args:
        event_time: The event timestamp
        pre_minutes: Minutes before the event
        post_minutes: Minutes after the event

    Returns:
        Tuple of (start_time, end_time)
    """
    start = event_time - timedelta(minutes=pre_minutes)
    end = event_time + timedelta(minutes=post_minutes)
    return start, end


def normalize_to_event(
    series: pd.Series,
    event_idx: int
) -> pd.Series:
    """
    Normalize a series relative to the value at event time.

    Args:
        series: Input series
        event_idx: Index of the event in the series

    Returns:
        Normalized series (rebased to 100 at event)
    """
    if event_idx >= len(series):
        event_idx = len(series) - 1

    base_value = series.iloc[event_idx]
    if base_value == 0:
        return series

    return (series / base_value) * 100


def calculate_volatility(
    returns: pd.Series,
    window: int = 20,
    annualize: bool = True
) -> pd.Series:
    """
    Calculate rolling volatility.

    Args:
        returns: Returns series
        window: Rolling window size
        annualize: Whether to annualize

    Returns:
        Volatility series
    """
    vol = returns.rolling(window=window).std()

    if annualize:
        # Assuming minute data, ~252 trading days, ~390 minutes/day
        vol = vol * np.sqrt(252 * 390)

    return vol


def classify_surprise(
    actual: float,
    expected: float,
    prior: Optional[float] = None
) -> Tuple[str, float]:
    """
    Classify the surprise of an economic release.

    Args:
        actual: Actual value
        expected: Expected/forecast value
        prior: Prior period value

    Returns:
        Tuple of (classification, surprise_magnitude)
    """
    if expected == 0:
        surprise = actual
    else:
        surprise = (actual - expected) / abs(expected) * 100

    if surprise > 10:
        classification = "strong_beat"
    elif surprise > 2:
        classification = "beat"
    elif surprise > -2:
        classification = "inline"
    elif surprise > -10:
        classification = "miss"
    else:
        classification = "strong_miss"

    return classification, surprise
