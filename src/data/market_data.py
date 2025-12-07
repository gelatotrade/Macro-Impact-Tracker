"""
Market Data Fetcher

Fetches market data from various sources for:
- Equities (indices, ETFs, individual stocks)
- Foreign Exchange (major pairs)
- Interest Rates (Treasury yields, bond futures)
- Volume data
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


class MarketDataFetcher:
    """
    Fetches market data for impact analysis.

    Uses yfinance for real market data with fallback to simulated data
    for demonstration purposes.
    """

    # Default symbols for each asset class
    DEFAULT_SYMBOLS = {
        "equities": {
            "SPY": "S&P 500 ETF",
            "QQQ": "Nasdaq 100 ETF",
            "DIA": "Dow Jones ETF",
            "IWM": "Russell 2000 ETF",
            "VIX": "Volatility Index",
            "^GSPC": "S&P 500 Index",
            "^DJI": "Dow Jones Index",
            "^IXIC": "Nasdaq Composite",
        },
        "fx": {
            "EURUSD=X": "EUR/USD",
            "GBPUSD=X": "GBP/USD",
            "USDJPY=X": "USD/JPY",
            "USDCHF=X": "USD/CHF",
            "AUDUSD=X": "AUD/USD",
            "USDCAD=X": "USD/CAD",
            "DX-Y.NYB": "US Dollar Index",
        },
        "rates": {
            "^TNX": "10-Year Treasury Yield",
            "^FVX": "5-Year Treasury Yield",
            "^TYX": "30-Year Treasury Yield",
            "^IRX": "13-Week Treasury Bill",
            "TLT": "20+ Year Treasury ETF",
            "IEF": "7-10 Year Treasury ETF",
            "SHY": "1-3 Year Treasury ETF",
        },
        "commodities": {
            "GC=F": "Gold Futures",
            "SI=F": "Silver Futures",
            "CL=F": "Crude Oil Futures",
            "NG=F": "Natural Gas Futures",
        },
    }

    # Sector ETFs for sector analysis
    SECTOR_ETFS = {
        "XLF": "Financials",
        "XLK": "Technology",
        "XLE": "Energy",
        "XLV": "Healthcare",
        "XLI": "Industrials",
        "XLP": "Consumer Staples",
        "XLY": "Consumer Discretionary",
        "XLU": "Utilities",
        "XLB": "Materials",
        "XLRE": "Real Estate",
    }

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the MarketDataFetcher.

        Args:
            cache_dir: Directory for caching data
        """
        self.yf_available = YFINANCE_AVAILABLE

        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(__file__).parent.parent.parent / "data" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._cache: Dict[str, pd.DataFrame] = {}

    def get_intraday_data(
        self,
        symbol: str,
        start: Union[str, datetime],
        end: Union[str, datetime],
        interval: str = "1m"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch intraday market data.

        Args:
            symbol: Ticker symbol
            start: Start datetime
            end: End datetime
            interval: Data interval (1m, 5m, 15m, 30m, 1h)

        Returns:
            DataFrame with OHLCV data
        """
        if not self.yf_available:
            return self._generate_sample_data(symbol, start, end, interval)

        try:
            ticker = yf.Ticker(symbol)

            # Convert string dates to datetime if needed
            if isinstance(start, str):
                start = pd.to_datetime(start)
            if isinstance(end, str):
                end = pd.to_datetime(end)

            # yfinance has limitations on intraday data
            # For minute data, we can only get last 7 days
            data = ticker.history(
                start=start,
                end=end,
                interval=interval,
                prepost=True  # Include pre/post market
            )

            if data.empty:
                print(f"No data available for {symbol}")
                return self._generate_sample_data(symbol, start, end, interval)

            return data

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return self._generate_sample_data(symbol, start, end, interval)

    def get_event_window_data(
        self,
        symbol: str,
        event_time: datetime,
        pre_minutes: int = 30,
        post_minutes: int = 120,
        interval: str = "1m"
    ) -> Optional[pd.DataFrame]:
        """
        Get market data around an event time.

        Args:
            symbol: Ticker symbol
            event_time: The event timestamp
            pre_minutes: Minutes before the event
            post_minutes: Minutes after the event
            interval: Data interval

        Returns:
            DataFrame with market data around the event
        """
        start = event_time - timedelta(minutes=pre_minutes)
        end = event_time + timedelta(minutes=post_minutes)

        return self.get_intraday_data(symbol, start, end, interval)

    def get_multi_asset_data(
        self,
        event_time: datetime,
        pre_minutes: int = 30,
        post_minutes: int = 120,
        interval: str = "1m",
        asset_classes: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Get data for multiple asset classes around an event.

        Args:
            event_time: The event timestamp
            pre_minutes: Minutes before the event
            post_minutes: Minutes after the event
            interval: Data interval
            asset_classes: List of asset classes to fetch

        Returns:
            Nested dictionary of {asset_class: {symbol: DataFrame}}
        """
        if asset_classes is None:
            asset_classes = ["equities", "fx", "rates"]

        result = {}

        for asset_class in asset_classes:
            if asset_class not in self.DEFAULT_SYMBOLS:
                continue

            result[asset_class] = {}
            symbols = self.DEFAULT_SYMBOLS[asset_class]

            for symbol, name in symbols.items():
                data = self.get_event_window_data(
                    symbol,
                    event_time,
                    pre_minutes,
                    post_minutes,
                    interval
                )
                if data is not None and not data.empty:
                    result[asset_class][symbol] = data

        return result

    def get_daily_data(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch daily market data.

        Args:
            symbol: Ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with daily OHLCV data
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if not self.yf_available:
            return self._generate_sample_data(symbol, start_date, end_date, "1d")

        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            return data

        except Exception as e:
            print(f"Error fetching daily data for {symbol}: {e}")
            return None

    def calculate_returns(
        self,
        data: pd.DataFrame,
        column: str = "Close",
        method: str = "simple"
    ) -> pd.Series:
        """
        Calculate returns from price data.

        Args:
            data: DataFrame with price data
            column: Column to use for returns
            method: 'simple' or 'log'

        Returns:
            Series of returns
        """
        if column not in data.columns:
            column = data.columns[0]

        prices = data[column]

        if method == "simple":
            return prices.pct_change()
        elif method == "log":
            return np.log(prices / prices.shift(1))
        else:
            raise ValueError(f"Unknown method: {method}")

    def calculate_cumulative_returns(
        self,
        data: pd.DataFrame,
        event_idx: int,
        column: str = "Close"
    ) -> pd.Series:
        """
        Calculate cumulative returns relative to event time.

        Args:
            data: DataFrame with price data
            event_idx: Index of the event in the data
            column: Column to use

        Returns:
            Series of cumulative returns (in percentage)
        """
        if column not in data.columns:
            column = data.columns[0]

        prices = data[column]
        base_price = prices.iloc[event_idx] if event_idx < len(prices) else prices.iloc[0]

        return ((prices / base_price) - 1) * 100

    def get_volume_profile(
        self,
        data: pd.DataFrame,
        event_idx: int
    ) -> Dict[str, Any]:
        """
        Analyze volume around an event.

        Args:
            data: DataFrame with volume data
            event_idx: Index of the event

        Returns:
            Dictionary with volume analysis
        """
        if "Volume" not in data.columns:
            return {"available": False}

        volume = data["Volume"]
        pre_event = volume.iloc[:event_idx] if event_idx > 0 else pd.Series([])
        post_event = volume.iloc[event_idx:] if event_idx < len(volume) else pd.Series([])

        return {
            "available": True,
            "pre_event_avg": pre_event.mean() if len(pre_event) > 0 else 0,
            "post_event_avg": post_event.mean() if len(post_event) > 0 else 0,
            "event_volume": volume.iloc[event_idx] if event_idx < len(volume) else 0,
            "max_volume": volume.max(),
            "volume_spike": (
                volume.iloc[event_idx] / pre_event.mean()
                if event_idx < len(volume) and len(pre_event) > 0 and pre_event.mean() > 0
                else 0
            ),
            "total_volume": volume.sum(),
        }

    def get_volatility(
        self,
        data: pd.DataFrame,
        window: int = 20,
        column: str = "Close"
    ) -> pd.Series:
        """
        Calculate rolling volatility.

        Args:
            data: DataFrame with price data
            window: Rolling window size
            column: Column to use

        Returns:
            Series of volatility values
        """
        returns = self.calculate_returns(data, column)
        return returns.rolling(window=window).std() * np.sqrt(252 * 390)  # Annualized

    def _generate_sample_data(
        self,
        symbol: str,
        start: Union[str, datetime],
        end: Union[str, datetime],
        interval: str = "1m"
    ) -> pd.DataFrame:
        """
        Generate sample data for demonstration.

        Args:
            symbol: Ticker symbol
            start: Start datetime
            end: End datetime
            interval: Data interval

        Returns:
            DataFrame with simulated OHLCV data
        """
        if isinstance(start, str):
            start = pd.to_datetime(start)
        if isinstance(end, str):
            end = pd.to_datetime(end)

        # Determine frequency
        freq_map = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1h",
            "1d": "1D",
        }
        freq = freq_map.get(interval, "1min")

        # Generate date range
        dates = pd.date_range(start=start, end=end, freq=freq)

        if len(dates) == 0:
            dates = pd.date_range(start=start, periods=100, freq=freq)

        # Base prices for different symbols
        base_prices = {
            "SPY": 450.0,
            "QQQ": 380.0,
            "^GSPC": 4500.0,
            "EURUSD=X": 1.08,
            "GBPUSD=X": 1.26,
            "USDJPY=X": 150.0,
            "^TNX": 4.5,
            "GC=F": 2000.0,
        }
        base = base_prices.get(symbol, 100.0)

        # Generate random walk
        np.random.seed(hash(symbol) % 2**32)
        returns = np.random.normal(0, 0.001, len(dates))

        # Add event effect (spike at middle)
        mid = len(dates) // 2
        returns[mid:mid+5] = np.random.normal(0.002, 0.003, min(5, len(returns) - mid))

        prices = base * np.exp(np.cumsum(returns))

        # Generate OHLCV data
        data = pd.DataFrame({
            "Open": prices * (1 + np.random.uniform(-0.001, 0.001, len(dates))),
            "High": prices * (1 + np.random.uniform(0, 0.002, len(dates))),
            "Low": prices * (1 - np.random.uniform(0, 0.002, len(dates))),
            "Close": prices,
            "Volume": np.random.randint(100000, 1000000, len(dates)),
        }, index=dates)

        # Add volume spike at event
        data.iloc[mid:mid+10, data.columns.get_loc("Volume")] *= 3

        return data

    def get_key_levels(
        self,
        data: pd.DataFrame,
        event_idx: int
    ) -> Dict[str, float]:
        """
        Calculate key price levels around an event.

        Args:
            data: DataFrame with price data
            event_idx: Index of the event

        Returns:
            Dictionary with key levels
        """
        high = data["High"]
        low = data["Low"]
        close = data["Close"]

        pre_high = high.iloc[:event_idx].max() if event_idx > 0 else high.iloc[0]
        pre_low = low.iloc[:event_idx].min() if event_idx > 0 else low.iloc[0]
        post_high = high.iloc[event_idx:].max() if event_idx < len(high) else high.iloc[-1]
        post_low = low.iloc[event_idx:].min() if event_idx < len(low) else low.iloc[-1]

        return {
            "pre_event_high": pre_high,
            "pre_event_low": pre_low,
            "post_event_high": post_high,
            "post_event_low": post_low,
            "event_price": close.iloc[event_idx] if event_idx < len(close) else close.iloc[-1],
            "range_expansion": (post_high - post_low) / (pre_high - pre_low) if (pre_high - pre_low) > 0 else 1,
        }

    def get_correlation_matrix(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for multiple symbols.

        Args:
            symbols: List of ticker symbols
            start_date: Start date
            end_date: End date

        Returns:
            Correlation matrix DataFrame
        """
        returns_data = {}

        for symbol in symbols:
            data = self.get_daily_data(symbol, start_date, end_date)
            if data is not None and not data.empty:
                returns_data[symbol] = self.calculate_returns(data)

        if not returns_data:
            return pd.DataFrame()

        returns_df = pd.DataFrame(returns_data)
        return returns_df.corr()

    def get_all_symbols(self) -> Dict[str, Dict[str, str]]:
        """Get all available symbols organized by asset class."""
        return self.DEFAULT_SYMBOLS.copy()

    def get_sector_data(
        self,
        event_time: datetime,
        pre_minutes: int = 30,
        post_minutes: int = 120,
        interval: str = "1m"
    ) -> Dict[str, pd.DataFrame]:
        """
        Get sector ETF data around an event.

        Args:
            event_time: Event timestamp
            pre_minutes: Minutes before event
            post_minutes: Minutes after event
            interval: Data interval

        Returns:
            Dictionary of {symbol: DataFrame}
        """
        result = {}

        for symbol, sector_name in self.SECTOR_ETFS.items():
            data = self.get_event_window_data(
                symbol,
                event_time,
                pre_minutes,
                post_minutes,
                interval
            )
            if data is not None and not data.empty:
                result[symbol] = data

        return result
