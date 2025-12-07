"""Tests for the macro events module."""

import pytest
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.macro_events import MacroEventFetcher


class TestMacroEventFetcher:
    """Tests for MacroEventFetcher class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fetcher = MacroEventFetcher()

    def test_initialization(self):
        """Test fetcher initializes correctly."""
        assert self.fetcher is not None
        assert self.fetcher.historical_events is not None

    def test_get_events_by_type(self):
        """Test getting events by type."""
        cpi_events = self.fetcher.get_events_by_type("CPI")
        assert isinstance(cpi_events, list)
        assert len(cpi_events) > 0

        for event in cpi_events:
            assert "date" in event
            assert "actual" in event
            assert "forecast" in event

    def test_get_all_events(self):
        """Test getting all events."""
        events_df = self.fetcher.get_all_events()
        assert not events_df.empty
        assert "type" in events_df.columns

    def test_calculate_surprise(self):
        """Test surprise calculation."""
        event = {
            "actual": 3.0,
            "forecast": 2.5,
            "prior": 2.4,
        }
        surprise_pct, classification = self.fetcher.calculate_surprise(event)

        assert surprise_pct == pytest.approx(20.0, rel=0.1)
        assert classification == "huge_beat"

    def test_calculate_surprise_miss(self):
        """Test surprise calculation for miss."""
        event = {
            "actual": 2.0,
            "forecast": 2.5,
            "prior": 2.4,
        }
        surprise_pct, classification = self.fetcher.calculate_surprise(event)

        assert surprise_pct == pytest.approx(-20.0, rel=0.1)
        assert classification == "huge_miss"

    def test_get_event_impact_summary(self):
        """Test event impact summary."""
        summary = self.fetcher.get_event_impact_summary("CPI")
        assert "total_events" in summary
        assert "avg_surprise" in summary
        assert "beat_rate" in summary


class TestEventFiltering:
    """Tests for event filtering functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fetcher = MacroEventFetcher()

    def test_filter_by_date_range(self):
        """Test filtering events by date range."""
        events = self.fetcher.get_all_events(
            start_date="2024-06-01",
            end_date="2024-06-30"
        )
        if not events.empty:
            assert all(events["date"] >= "2024-06-01")
            assert all(events["date"] <= "2024-06-30")

    def test_filter_by_importance(self):
        """Test filtering events by importance."""
        events = self.fetcher.get_all_events(importance="high")
        if not events.empty:
            assert all(events["importance"] == "high")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
