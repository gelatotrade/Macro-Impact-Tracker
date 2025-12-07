#!/usr/bin/env python3
"""
Live Event Monitor Example

Demonstrates how to set up real-time monitoring for upcoming economic events.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from datetime import datetime, timedelta
import pytz

from src.data.macro_events import MacroEventFetcher
from src.data.market_data import MarketDataFetcher
from src.analysis.impact_analyzer import ImpactAnalyzer


def monitor_upcoming_events():
    """Monitor upcoming economic events."""
    print("=" * 60)
    print("LIVE EVENT MONITOR")
    print("=" * 60)
    print("\nMonitoring for upcoming economic events...")
    print("Press Ctrl+C to stop\n")

    # Initialize
    macro_fetcher = MacroEventFetcher()
    market_fetcher = MarketDataFetcher()
    analyzer = ImpactAnalyzer(macro_fetcher, market_fetcher)

    et = pytz.timezone("America/New_York")

    try:
        while True:
            current_time = datetime.now(et)
            print(f"\n[{current_time.strftime('%Y-%m-%d %H:%M:%S ET')}]")

            # Get upcoming events (next 24 hours)
            upcoming = macro_fetcher.get_upcoming_events(days=1)

            if not upcoming.empty:
                print(f"\nUpcoming events in next 24 hours:")
                print("-" * 40)

                for _, event in upcoming.iterrows():
                    event_dt = datetime.strptime(
                        f"{event['date']} {event['time']}",
                        "%Y-%m-%d %H:%M"
                    )
                    event_dt = et.localize(event_dt)

                    time_until = event_dt - current_time
                    hours = time_until.total_seconds() / 3600

                    if hours > 0:
                        print(f"  {event['type']:6s} | {event['date']} {event['time']} ET | "
                              f"in {hours:.1f} hours")

                        # If event is within 1 hour, show more details
                        if hours < 1:
                            print(f"         Forecast: {event.get('forecast', 'N/A')}")
                            print(f"         Prior: {event.get('prior', 'N/A')}")

                            # Get current market levels
                            spy_data = market_fetcher.get_intraday_data(
                                "SPY",
                                current_time - timedelta(minutes=5),
                                current_time,
                                interval="1m"
                            )
                            if spy_data is not None and not spy_data.empty:
                                print(f"         SPY: ${spy_data['Close'].iloc[-1]:.2f}")
            else:
                print("No events scheduled in next 24 hours")

            # Recent events (last 2 hours)
            print("\nRecent events (last 2 hours):")
            print("-" * 40)

            recent = macro_fetcher.get_recent_events(days=1)
            if not recent.empty:
                for _, event in recent.iterrows():
                    event_dt = datetime.strptime(
                        f"{event['date']} {event['time']}",
                        "%Y-%m-%d %H:%M"
                    )
                    event_dt = et.localize(event_dt)

                    time_since = current_time - event_dt
                    hours_since = time_since.total_seconds() / 3600

                    if 0 < hours_since < 2:
                        print(f"  {event['type']:6s} | {event['date']} {event['time']} ET | "
                              f"{hours_since:.1f} hours ago")
                        print(f"         Actual: {event.get('actual', 'N/A')} | "
                              f"Forecast: {event.get('forecast', 'N/A')}")

            # Sleep for 5 minutes before next check
            print("\nNext update in 5 minutes...")
            time.sleep(300)

    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")


def simulate_event_release():
    """Simulate what happens when an event is released."""
    print("=" * 60)
    print("SIMULATED EVENT RELEASE")
    print("=" * 60)

    macro_fetcher = MacroEventFetcher()
    market_fetcher = MarketDataFetcher()
    analyzer = ImpactAnalyzer(macro_fetcher, market_fetcher)

    # Simulate a CPI release that beats expectations
    simulated_event = {
        "type": "CPI",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": "08:30",
        "actual": 2.9,
        "forecast": 3.0,
        "prior": 3.1,
    }

    print("\n🔔 SIMULATED EVENT RELEASE!")
    print(f"   Type: {simulated_event['type']}")
    print(f"   Actual: {simulated_event['actual']}")
    print(f"   Forecast: {simulated_event['forecast']}")
    print(f"   Prior: {simulated_event['prior']}")

    surprise_pct, surprise_class = macro_fetcher.calculate_surprise(simulated_event)
    print(f"\n   Surprise: {surprise_pct:+.2f}% ({surprise_class})")

    # Expected market reaction based on historical analysis
    print("\n📊 EXPECTED MARKET REACTION (based on historical analysis):")

    stats = analyzer.get_aggregate_stats("CPI", num_events=10)

    if stats.num_events > 0:
        print(f"\n   Based on {stats.num_events} historical CPI events:")
        print(f"   Average surprise: {stats.avg_surprise:.1f}%")

        # Show typical reactions for beats vs misses
        if "beat" in surprise_class:
            print("\n   This is a BEAT - historically this means:")
            for key, value in stats.beat_avg_reaction.items():
                if "15min" in key:
                    print(f"     {key}: {value:+.3f}%")
        else:
            print("\n   This is a MISS - historically this means:")
            for key, value in stats.miss_avg_reaction.items():
                if "15min" in key:
                    print(f"     {key}: {value:+.3f}%")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Live Event Monitor")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run simulation instead of live monitor"
    )
    args = parser.parse_args()

    if args.simulate:
        simulate_event_release()
    else:
        monitor_upcoming_events()
