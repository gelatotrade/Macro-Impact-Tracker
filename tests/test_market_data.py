"""Tests for the market data module."""

import pytest
from datetime import datetime, timedelta
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.market_data import MarketDataFetcher


class TestMarketDataFetcher:
    """Tests for MarketDataFetcher class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fetcher = MarketDataFetcher()

    def test_initialization(self):
        """Test fetcher initializes correctly."""
        assert self.fetcher is not None
        assert self.fetcher.DEFAULT_SYMBOLS is not None

    def test_get_all_symbols(self):
        """Test getting all available symbols."""
        symbols = self.fetcher.get_all_symbols()
        assert "equities" in symbols
        assert "fx" in symbols
        assert "rates" in symbols
        assert "SPY" in symbols["equities"]

    def test_generate_sample_data(self):
        """Test sample data generation."""
        data = self.fetcher._generate_sample_data(
            "SPY",
            datetime.now() - timedelta(hours=2),
            datetime.now(),
            interval="1m"
        )
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert "Open" in data.columns
        assert "High" in data.columns
        assert "Low" in data.columns
        assert "Close" in data.columns
        assert "Volume" in data.columns

    def test_calculate_returns(self):
        """Test returns calculation."""
        data = self.fetcher._generate_sample_data(
            "SPY",
            datetime.now() - timedelta(hours=2),
            datetime.now()
        )
        returns = self.fetcher.calculate_returns(data)
        assert isinstance(returns, pd.Series)
        assert len(returns) == len(data)

    def test_calculate_cumulative_returns(self):
        """Test cumulative returns calculation."""
        data = self.fetcher._generate_sample_data(
            "SPY",
            datetime.now() - timedelta(hours=2),
            datetime.now()
        )
        event_idx = len(data) // 2
        cum_returns = self.fetcher.calculate_cumulative_returns(data, event_idx)
        assert isinstance(cum_returns, pd.Series)
        # At event index, cumulative return should be 0
        assert cum_returns.iloc[event_idx] == pytest.approx(0, abs=0.01)

    def test_get_volume_profile(self):
        """Test volume profile calculation."""
        data = self.fetcher._generate_sample_data(
            "SPY",
            datetime.now() - timedelta(hours=2),
            datetime.now()
        )
        event_idx = len(data) // 2
        profile = self.fetcher.get_volume_profile(data, event_idx)
        assert profile["available"] == True
        assert "pre_event_avg" in profile
        assert "post_event_avg" in profile
        assert "volume_spike" in profile

    def test_get_key_levels(self):
        """Test key levels calculation."""
        data = self.fetcher._generate_sample_data(
            "SPY",
            datetime.now() - timedelta(hours=2),
            datetime.now()
        )
        event_idx = len(data) // 2
        levels = self.fetcher.get_key_levels(data, event_idx)
        assert "pre_event_high" in levels
        assert "pre_event_low" in levels
        assert "post_event_high" in levels
        assert "post_event_low" in levels
        assert "event_price" in levels


class TestEventWindowData:
    """Tests for event window data retrieval."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fetcher = MarketDataFetcher()

    def test_get_event_window_data(self):
        """Test getting data around an event."""
        event_time = datetime.now() - timedelta(hours=1)
        data = self.fetcher.get_event_window_data(
            "SPY",
            event_time,
            pre_minutes=30,
            post_minutes=30
        )
        assert data is not None
        assert not data.empty

    def test_get_multi_asset_data(self):
        """Test getting multi-asset data."""
        event_time = datetime.now() - timedelta(hours=1)
        data = self.fetcher.get_multi_asset_data(
            event_time,
            pre_minutes=30,
            post_minutes=30,
            asset_classes=["equities", "fx"]
        )
        assert isinstance(data, dict)
        assert "equities" in data
        assert "fx" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
