#!/usr/bin/env python3
"""
Basic Usage Example for Macro Event Impact Tracker

This script demonstrates how to use the tracker programmatically
to analyze market reactions to macroeconomic events.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
import pandas as pd

from src.data.macro_events import MacroEventFetcher
from src.data.market_data import MarketDataFetcher
from src.analysis.impact_analyzer import ImpactAnalyzer
from src.visualization.charts import ReactionCharts


def main():
    """Main example function."""
    print("=" * 60)
    print("MACRO EVENT IMPACT TRACKER - BASIC USAGE")
    print("=" * 60)

    # Initialize components
    print("\n1. Initializing components...")
    macro_fetcher = MacroEventFetcher()
    market_fetcher = MarketDataFetcher()
    analyzer = ImpactAnalyzer(macro_fetcher, market_fetcher)
    charts = ReactionCharts()

    # Example 1: Get recent economic events
    print("\n2. Getting recent economic events...")
    events_df = macro_fetcher.get_all_events(
        start_date="2024-01-01",
        end_date="2024-12-31"
    )
    print(f"   Found {len(events_df)} events")

    # Example 2: Analyze a specific CPI release
    print("\n3. Analyzing CPI event...")
    cpi_events = macro_fetcher.get_events_by_type("CPI")

    if cpi_events:
        # Get the most recent CPI event
        latest_cpi = cpi_events[-1]
        latest_cpi["type"] = "CPI"

        print(f"   Date: {latest_cpi['date']}")
        print(f"   Actual: {latest_cpi['actual']}")
        print(f"   Forecast: {latest_cpi['forecast']}")
        print(f"   Prior: {latest_cpi['prior']}")

        # Calculate surprise
        surprise_pct, surprise_class = macro_fetcher.calculate_surprise(latest_cpi)
        print(f"   Surprise: {surprise_pct:.2f}% ({surprise_class})")

        # Analyze market reaction
        print("\n4. Analyzing market reaction...")
        reaction = analyzer.analyze_event(
            latest_cpi,
            assets={
                "equities": ["SPY", "QQQ"],
                "fx": ["EURUSD=X"],
                "rates": ["^TNX"],
            }
        )

        # Print reaction summary
        print("\n   REACTION SUMMARY:")
        for asset_class, class_reactions in reaction.reactions.items():
            print(f"\n   {asset_class.upper()}:")
            for symbol, tf_reactions in class_reactions.items():
                print(f"      {symbol}:")
                for tf, value in tf_reactions.items():
                    direction = "▲" if value > 0 else "▼" if value < 0 else "─"
                    print(f"         {tf}: {direction} {value:+.3f}%")

        # Example 3: Get aggregate statistics
        print("\n5. Getting aggregate statistics for CPI events...")
        stats = analyzer.get_aggregate_stats("CPI", num_events=10)
        print(f"   Number of events analyzed: {stats.num_events}")
        print(f"   Average surprise: {stats.avg_surprise:.2f}%")
        print(f"   Beat rate: {stats.beat_avg_reaction}")

        # Example 4: Compare event types
        print("\n6. Comparing event types...")
        comparison = analyzer.compare_events(
            ["CPI", "NFP", "FOMC"],
            timeframe="15min",
            asset="SPY"
        )
        print(comparison.to_string())

        # Example 5: Generate and save charts
        print("\n7. Generating charts...")

        # Create event datetime
        event_datetime = pd.to_datetime(
            f"{latest_cpi['date']} {latest_cpi['time']}"
        ).tz_localize("America/New_York")

        # Get market data
        market_data = market_fetcher.get_multi_asset_data(
            event_datetime,
            pre_minutes=30,
            post_minutes=120,
            asset_classes=["equities", "fx", "rates"]
        )

        if market_data:
            # Create multi-asset chart
            fig = charts.create_multi_asset_reaction_chart(
                market_data,
                event_datetime,
                latest_cpi
            )

            # Save chart
            output_path = Path(__file__).parent.parent / "output"
            output_path.mkdir(exist_ok=True)
            chart_file = output_path / "cpi_reaction_chart.html"
            charts.save_chart(fig, str(chart_file))
            print(f"   Chart saved to: {chart_file}")

        # Example 6: Get historical reactions
        print("\n8. Getting historical reaction patterns...")
        reactions = analyzer.analyze_event_history("CPI", num_events=5)
        print(f"   Analyzed {len(reactions)} historical CPI events")

        # Create heatmap
        if reactions:
            fig_heatmap = charts.create_reaction_heatmap(reactions, "equities")
            heatmap_file = output_path / "cpi_heatmap.html"
            charts.save_chart(fig_heatmap, str(heatmap_file))
            print(f"   Heatmap saved to: {heatmap_file}")

    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)
    print("\nTo run the interactive dashboard:")
    print("  streamlit run app.py")
    print("\nTo customize analysis, modify the parameters in this script.")


if __name__ == "__main__":
    main()
