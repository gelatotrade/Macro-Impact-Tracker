"""Tests for the impact analyzer module."""

import pytest
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.macro_events import MacroEventFetcher
from src.data.market_data import MarketDataFetcher
from src.analysis.impact_analyzer import ImpactAnalyzer, EventReaction


class TestImpactAnalyzer:
    """Tests for ImpactAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.macro_fetcher = MacroEventFetcher()
        self.market_fetcher = MarketDataFetcher()
        self.analyzer = ImpactAnalyzer(
            self.macro_fetcher,
            self.market_fetcher
        )

    def test_initialization(self):
        """Test analyzer initializes correctly."""
        assert self.analyzer is not None
        assert self.analyzer.macro_fetcher is not None
        assert self.analyzer.market_fetcher is not None

    def test_analyze_event(self):
        """Test single event analysis."""
        # Get a sample event
        events = self.macro_fetcher.get_events_by_type("CPI")
        if events:
            event = events[-1]
            event["type"] = "CPI"

            reaction = self.analyzer.analyze_event(
                event,
                assets={"equities": ["SPY"]}
            )

            assert isinstance(reaction, EventReaction)
            assert reaction.event_type == "CPI"
            assert reaction.reactions is not None

    def test_analyze_event_history(self):
        """Test historical event analysis."""
        reactions = self.analyzer.analyze_event_history(
            "CPI",
            num_events=3
        )
        assert isinstance(reactions, list)
        assert len(reactions) <= 3

    def test_get_aggregate_stats(self):
        """Test aggregate statistics."""
        stats = self.analyzer.get_aggregate_stats("CPI", num_events=5)
        assert stats.event_type == "CPI"
        assert stats.num_events >= 0

    def test_compare_events(self):
        """Test event comparison."""
        comparison = self.analyzer.compare_events(
            ["CPI", "NFP"],
            timeframe="15min",
            asset="SPY"
        )
        assert comparison is not None

    def test_get_reaction_distribution(self):
        """Test reaction distribution."""
        dist = self.analyzer.get_reaction_distribution(
            "CPI",
            asset="SPY",
            timeframe="15min"
        )
        # May be empty if no data
        assert isinstance(dist, dict)

    def test_generate_event_report(self):
        """Test report generation."""
        events = self.macro_fetcher.get_events_by_type("CPI")
        if events:
            event = events[-1]
            event["type"] = "CPI"
            reaction = self.analyzer.analyze_event(event)
            report = self.analyzer.generate_event_report(event, reaction)

            assert isinstance(report, str)
            assert "CPI" in report
            assert "MARKET REACTIONS" in report


class TestReactionTimeframes:
    """Tests for reaction timeframe handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.macro_fetcher = MacroEventFetcher()
        self.market_fetcher = MarketDataFetcher()
        self.analyzer = ImpactAnalyzer(
            self.macro_fetcher,
            self.market_fetcher
        )

    def test_timeframes_defined(self):
        """Test that all timeframes are defined."""
        expected = ["1min", "5min", "15min", "30min", "1hour", "2hour"]
        for tf in expected:
            assert tf in self.analyzer.timeframes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
