#!/usr/bin/env python3
"""
NFP Analysis Example

Demonstrates how to analyze Non-Farm Payroll releases
and their impact on various markets.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
import pandas as pd

from src.data.macro_events import MacroEventFetcher
from src.data.market_data import MarketDataFetcher
from src.analysis.impact_analyzer import ImpactAnalyzer
from src.visualization.charts import ReactionCharts


def analyze_nfp_impact():
    """Analyze NFP impact on markets."""
    print("=" * 60)
    print("NFP (NON-FARM PAYROLLS) IMPACT ANALYSIS")
    print("=" * 60)

    # Initialize
    macro_fetcher = MacroEventFetcher()
    market_fetcher = MarketDataFetcher()
    analyzer = ImpactAnalyzer(macro_fetcher, market_fetcher)
    charts = ReactionCharts()

    # Get NFP events
    nfp_events = macro_fetcher.get_events_by_type("NFP")
    print(f"\nFound {len(nfp_events)} NFP events in database")

    # Analyze each event
    print("\n" + "-" * 40)
    print("INDIVIDUAL EVENT ANALYSIS")
    print("-" * 40)

    for event in nfp_events[-5:]:  # Last 5 events
        event["type"] = "NFP"
        surprise_pct, surprise_class = macro_fetcher.calculate_surprise(event)

        print(f"\n{event['date']}:")
        print(f"  Jobs Added: {event['actual']}K (exp: {event['forecast']}K)")
        print(f"  Surprise: {surprise_pct:+.1f}% ({surprise_class})")

    # Get aggregate statistics
    print("\n" + "-" * 40)
    print("AGGREGATE STATISTICS")
    print("-" * 40)

    stats = analyzer.get_aggregate_stats("NFP", num_events=10)
    print(f"\nEvents analyzed: {stats.num_events}")
    print(f"Average surprise: {stats.avg_surprise:.1f}%")

    # Distribution of reactions
    print("\n" + "-" * 40)
    print("SPY REACTION DISTRIBUTION (15-min)")
    print("-" * 40)

    dist = analyzer.get_reaction_distribution("NFP", asset="SPY", timeframe="15min")
    if dist:
        print(f"  Mean: {dist.get('mean', 0):.3f}%")
        print(f"  Median: {dist.get('median', 0):.3f}%")
        print(f"  Std Dev: {dist.get('std', 0):.3f}%")
        print(f"  Min: {dist.get('min', 0):.3f}%")
        print(f"  Max: {dist.get('max', 0):.3f}%")
        print(f"  Positive Rate: {dist.get('positive_rate', 0):.1f}%")

    # Asset class comparison
    print("\n" + "-" * 40)
    print("CROSS-ASSET ANALYSIS")
    print("-" * 40)

    # Analyze most recent NFP
    if nfp_events:
        latest = nfp_events[-1]
        latest["type"] = "NFP"

        reaction = analyzer.analyze_event(
            latest,
            assets={
                "equities": ["SPY", "QQQ", "IWM"],
                "fx": ["EURUSD=X", "USDJPY=X", "DX-Y.NYB"],
                "rates": ["^TNX", "TLT"],
            }
        )

        print(f"\nMost recent NFP ({latest['date']}):")
        print(f"Volume Spike: {reaction.volume_spike:.1f}x")
        print(f"Post-event Range: {reaction.range_pct:.2f}%")

        # Print detailed reactions
        for asset_class, class_reactions in reaction.reactions.items():
            print(f"\n{asset_class.upper()}:")
            for symbol, tf_reactions in class_reactions.items():
                if "15min" in tf_reactions:
                    val = tf_reactions["15min"]
                    direction = "▲" if val > 0 else "▼"
                    print(f"  {symbol}: {direction} {val:+.3f}% (15-min)")

    # Generate report
    print("\n" + "-" * 40)
    print("FULL EVENT REPORT")
    print("-" * 40)

    if nfp_events:
        latest = nfp_events[-1]
        latest["type"] = "NFP"
        reaction = analyzer.analyze_event(latest)
        report = analyzer.generate_event_report(latest, reaction)
        print(report)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    analyze_nfp_impact()
