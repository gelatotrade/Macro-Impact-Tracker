"""
Impact Analysis Engine

Analyzes how markets react to macroeconomic events by calculating:
- Price reactions at various time horizons
- Volume changes
- Volatility changes
- Cross-asset correlations
- Statistical significance of reactions
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum


class ReactionTimeframe(Enum):
    """Standard timeframes for measuring reactions."""
    IMMEDIATE = 1      # 1 minute
    SHORT = 5          # 5 minutes
    MEDIUM = 15        # 15 minutes
    EXTENDED = 30      # 30 minutes
    HOUR = 60          # 1 hour
    TWO_HOURS = 120    # 2 hours
    END_OF_DAY = 390   # Full trading day


@dataclass
class EventReaction:
    """Represents the market reaction to a single event."""
    event_type: str
    event_date: str
    event_time: str
    actual: float
    forecast: float
    prior: float
    surprise_pct: float
    surprise_class: str

    # Price reactions at different timeframes (in %)
    reactions: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Volume analysis
    volume_spike: float = 0.0
    volume_change_pct: float = 0.0

    # Volatility analysis
    pre_event_vol: float = 0.0
    post_event_vol: float = 0.0
    vol_change: float = 0.0

    # Key levels
    high_after: float = 0.0
    low_after: float = 0.0
    range_pct: float = 0.0


@dataclass
class AggregateStats:
    """Aggregate statistics for event reactions."""
    event_type: str
    num_events: int
    avg_surprise: float
    avg_reaction: Dict[str, Dict[str, float]]  # {timeframe: {asset: avg_reaction}}
    std_reaction: Dict[str, Dict[str, float]]
    beat_avg_reaction: Dict[str, float]
    miss_avg_reaction: Dict[str, float]
    correlation_with_surprise: Dict[str, float]


class ImpactAnalyzer:
    """
    Analyzes the impact of macro events on markets.

    Calculates reactions across multiple asset classes and timeframes,
    and provides statistical analysis of historical patterns.
    """

    def __init__(
        self,
        macro_fetcher,
        market_fetcher,
        default_assets: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize the ImpactAnalyzer.

        Args:
            macro_fetcher: MacroEventFetcher instance
            market_fetcher: MarketDataFetcher instance
            default_assets: Default assets to analyze
        """
        self.macro_fetcher = macro_fetcher
        self.market_fetcher = market_fetcher

        self.default_assets = default_assets or {
            "equities": ["SPY", "QQQ", "^VIX"],
            "fx": ["EURUSD=X", "USDJPY=X", "DX-Y.NYB"],
            "rates": ["^TNX", "TLT"],
        }

        self.timeframes = {
            "1min": 1,
            "5min": 5,
            "15min": 15,
            "30min": 30,
            "1hour": 60,
            "2hour": 120,
        }

    def analyze_event(
        self,
        event: Dict[str, Any],
        assets: Optional[Dict[str, List[str]]] = None,
        pre_minutes: int = 30,
        post_minutes: int = 120
    ) -> EventReaction:
        """
        Analyze market reaction to a single event.

        Args:
            event: Event dictionary with date, time, actual, forecast, prior
            assets: Assets to analyze by class
            pre_minutes: Minutes before event to analyze
            post_minutes: Minutes after event to analyze

        Returns:
            EventReaction dataclass with analysis results
        """
        if assets is None:
            assets = self.default_assets

        # Parse event time
        event_datetime = pd.to_datetime(
            f"{event['date']} {event['time']}"
        ).tz_localize("America/New_York")

        # Calculate surprise
        surprise_pct, surprise_class = self._calculate_surprise(
            event.get("actual", 0),
            event.get("forecast", 0)
        )

        # Initialize reaction object
        reaction = EventReaction(
            event_type=event.get("type", "unknown"),
            event_date=event["date"],
            event_time=event["time"],
            actual=event.get("actual", 0),
            forecast=event.get("forecast", 0),
            prior=event.get("prior", 0),
            surprise_pct=surprise_pct,
            surprise_class=surprise_class,
        )

        # Get market data for each asset class
        all_reactions = {}

        for asset_class, symbols in assets.items():
            class_reactions = {}

            for symbol in symbols:
                data = self.market_fetcher.get_event_window_data(
                    symbol,
                    event_datetime,
                    pre_minutes,
                    post_minutes,
                    interval="1m"
                )

                if data is None or data.empty:
                    continue

                # Find event index
                event_idx = self._find_event_index(data, event_datetime)

                # Calculate reactions at different timeframes
                symbol_reactions = {}
                for tf_name, tf_minutes in self.timeframes.items():
                    if event_idx + tf_minutes < len(data):
                        start_price = data["Close"].iloc[event_idx]
                        end_price = data["Close"].iloc[event_idx + tf_minutes]
                        reaction_pct = ((end_price / start_price) - 1) * 100
                        symbol_reactions[tf_name] = reaction_pct

                class_reactions[symbol] = symbol_reactions

                # Volume analysis for first symbol in class
                if symbol == symbols[0]:
                    volume_profile = self.market_fetcher.get_volume_profile(data, event_idx)
                    if volume_profile.get("available"):
                        reaction.volume_spike = volume_profile.get("volume_spike", 0)
                        if volume_profile.get("pre_event_avg", 0) > 0:
                            reaction.volume_change_pct = (
                                (volume_profile.get("post_event_avg", 0) /
                                 volume_profile.get("pre_event_avg", 1)) - 1
                            ) * 100

                    # Key levels
                    key_levels = self.market_fetcher.get_key_levels(data, event_idx)
                    reaction.high_after = key_levels.get("post_event_high", 0)
                    reaction.low_after = key_levels.get("post_event_low", 0)
                    if key_levels.get("event_price", 0) > 0:
                        reaction.range_pct = (
                            (reaction.high_after - reaction.low_after) /
                            key_levels.get("event_price", 1)
                        ) * 100

            all_reactions[asset_class] = class_reactions

        reaction.reactions = all_reactions
        return reaction

    def analyze_event_history(
        self,
        event_type: str,
        num_events: int = 10,
        assets: Optional[Dict[str, List[str]]] = None
    ) -> List[EventReaction]:
        """
        Analyze historical reactions for an event type.

        Args:
            event_type: Type of event (CPI, NFP, etc.)
            num_events: Number of past events to analyze
            assets: Assets to analyze

        Returns:
            List of EventReaction objects
        """
        events = self.macro_fetcher.get_events_by_type(event_type)
        events = events[-num_events:] if len(events) > num_events else events

        reactions = []
        for event in events:
            event["type"] = event_type
            reaction = self.analyze_event(event, assets)
            reactions.append(reaction)

        return reactions

    def get_aggregate_stats(
        self,
        event_type: str,
        num_events: int = 10
    ) -> AggregateStats:
        """
        Calculate aggregate statistics for event reactions.

        Args:
            event_type: Type of event
            num_events: Number of events to analyze

        Returns:
            AggregateStats dataclass
        """
        reactions = self.analyze_event_history(event_type, num_events)

        if not reactions:
            return AggregateStats(
                event_type=event_type,
                num_events=0,
                avg_surprise=0,
                avg_reaction={},
                std_reaction={},
                beat_avg_reaction={},
                miss_avg_reaction={},
                correlation_with_surprise={}
            )

        # Calculate average surprise
        surprises = [r.surprise_pct for r in reactions]
        avg_surprise = np.mean(surprises)

        # Aggregate reactions by timeframe and asset
        avg_reactions = {}
        std_reactions = {}

        for tf_name in self.timeframes:
            tf_reactions = {}
            tf_stds = {}

            for r in reactions:
                for asset_class, class_reactions in r.reactions.items():
                    for symbol, symbol_reactions in class_reactions.items():
                        key = f"{asset_class}_{symbol}"
                        if key not in tf_reactions:
                            tf_reactions[key] = []

                        if tf_name in symbol_reactions:
                            tf_reactions[key].append(symbol_reactions[tf_name])

            for key, values in tf_reactions.items():
                if values:
                    tf_stds[key] = np.std(values)
                    tf_reactions[key] = np.mean(values)
                else:
                    tf_reactions[key] = 0
                    tf_stds[key] = 0

            avg_reactions[tf_name] = tf_reactions
            std_reactions[tf_name] = tf_stds

        # Separate beats and misses
        beats = [r for r in reactions if "beat" in r.surprise_class]
        misses = [r for r in reactions if "miss" in r.surprise_class]

        beat_avg = self._calculate_group_avg(beats)
        miss_avg = self._calculate_group_avg(misses)

        # Calculate correlation with surprise
        correlations = self._calculate_surprise_correlations(reactions)

        return AggregateStats(
            event_type=event_type,
            num_events=len(reactions),
            avg_surprise=avg_surprise,
            avg_reaction=avg_reactions,
            std_reaction=std_reactions,
            beat_avg_reaction=beat_avg,
            miss_avg_reaction=miss_avg,
            correlation_with_surprise=correlations
        )

    def compare_events(
        self,
        event_types: List[str],
        timeframe: str = "15min",
        asset: str = "SPY"
    ) -> pd.DataFrame:
        """
        Compare reactions across different event types.

        Args:
            event_types: List of event types to compare
            timeframe: Timeframe to compare
            asset: Asset to analyze

        Returns:
            DataFrame with comparison data
        """
        results = []

        for event_type in event_types:
            stats = self.get_aggregate_stats(event_type)

            if stats.num_events > 0:
                # Find the reaction for the specified asset
                reaction_key = None
                for key in stats.avg_reaction.get(timeframe, {}).keys():
                    if asset in key:
                        reaction_key = key
                        break

                avg_react = stats.avg_reaction.get(timeframe, {}).get(reaction_key, 0)
                std_react = stats.std_reaction.get(timeframe, {}).get(reaction_key, 0)

                results.append({
                    "event_type": event_type,
                    "num_events": stats.num_events,
                    "avg_surprise": stats.avg_surprise,
                    f"avg_reaction_{timeframe}": avg_react,
                    f"std_reaction_{timeframe}": std_react,
                    "beat_avg": stats.beat_avg_reaction.get(reaction_key, 0),
                    "miss_avg": stats.miss_avg_reaction.get(reaction_key, 0),
                })

        return pd.DataFrame(results)

    def get_reaction_distribution(
        self,
        event_type: str,
        asset: str = "SPY",
        timeframe: str = "15min"
    ) -> Dict[str, Any]:
        """
        Get distribution statistics for reactions.

        Args:
            event_type: Type of event
            asset: Asset to analyze
            timeframe: Timeframe

        Returns:
            Dictionary with distribution statistics
        """
        reactions = self.analyze_event_history(event_type)

        values = []
        for r in reactions:
            for asset_class, class_reactions in r.reactions.items():
                for symbol, symbol_reactions in class_reactions.items():
                    if asset in symbol and timeframe in symbol_reactions:
                        values.append(symbol_reactions[timeframe])

        if not values:
            return {}

        values = np.array(values)

        return {
            "mean": np.mean(values),
            "median": np.median(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "q25": np.percentile(values, 25),
            "q75": np.percentile(values, 75),
            "skew": self._calculate_skewness(values),
            "kurtosis": self._calculate_kurtosis(values),
            "positive_rate": np.mean(values > 0) * 100,
            "values": values.tolist(),
        }

    def _calculate_surprise(
        self,
        actual: float,
        forecast: float
    ) -> Tuple[float, str]:
        """Calculate surprise percentage and classification."""
        if forecast == 0:
            surprise_pct = 0
        else:
            surprise_pct = ((actual - forecast) / abs(forecast)) * 100

        if surprise_pct > 20:
            classification = "huge_beat"
        elif surprise_pct > 5:
            classification = "beat"
        elif surprise_pct > -5:
            classification = "inline"
        elif surprise_pct > -20:
            classification = "miss"
        else:
            classification = "huge_miss"

        return surprise_pct, classification

    def _find_event_index(
        self,
        data: pd.DataFrame,
        event_time: datetime
    ) -> int:
        """Find the index closest to the event time."""
        if data.index.tz is None:
            event_time = event_time.replace(tzinfo=None)

        # Find closest index
        time_diffs = abs(data.index - event_time)
        return time_diffs.argmin()

    def _calculate_group_avg(
        self,
        reactions: List[EventReaction]
    ) -> Dict[str, float]:
        """Calculate average reactions for a group."""
        if not reactions:
            return {}

        totals = {}
        counts = {}

        for r in reactions:
            for asset_class, class_reactions in r.reactions.items():
                for symbol, symbol_reactions in class_reactions.items():
                    for tf, value in symbol_reactions.items():
                        key = f"{asset_class}_{symbol}_{tf}"
                        if key not in totals:
                            totals[key] = 0
                            counts[key] = 0
                        totals[key] += value
                        counts[key] += 1

        return {k: totals[k] / counts[k] for k in totals}

    def _calculate_surprise_correlations(
        self,
        reactions: List[EventReaction]
    ) -> Dict[str, float]:
        """Calculate correlation between surprise and reaction."""
        correlations = {}

        surprises = [r.surprise_pct for r in reactions]

        for tf_name in self.timeframes:
            for r in reactions:
                for asset_class, class_reactions in r.reactions.items():
                    for symbol, symbol_reactions in class_reactions.items():
                        key = f"{asset_class}_{symbol}_{tf_name}"
                        if key not in correlations:
                            correlations[key] = {"surprises": [], "reactions": []}

                        if tf_name in symbol_reactions:
                            correlations[key]["surprises"].append(r.surprise_pct)
                            correlations[key]["reactions"].append(
                                symbol_reactions[tf_name]
                            )

        result = {}
        for key, data in correlations.items():
            if len(data["surprises"]) > 2:
                corr = np.corrcoef(data["surprises"], data["reactions"])[0, 1]
                result[key] = corr if not np.isnan(corr) else 0

        return result

    def _calculate_skewness(self, values: np.ndarray) -> float:
        """Calculate skewness of distribution."""
        n = len(values)
        if n < 3:
            return 0
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0
        return np.mean(((values - mean) / std) ** 3)

    def _calculate_kurtosis(self, values: np.ndarray) -> float:
        """Calculate kurtosis of distribution."""
        n = len(values)
        if n < 4:
            return 0
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0
        return np.mean(((values - mean) / std) ** 4) - 3

    def generate_event_report(
        self,
        event: Dict[str, Any],
        reaction: EventReaction
    ) -> str:
        """
        Generate a text report for an event.

        Args:
            event: Event dictionary
            reaction: EventReaction object

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append(f"EVENT IMPACT REPORT: {reaction.event_type}")
        report.append("=" * 60)
        report.append(f"\nDate: {reaction.event_date} {reaction.event_time} ET")
        report.append(f"Actual: {reaction.actual}")
        report.append(f"Forecast: {reaction.forecast}")
        report.append(f"Prior: {reaction.prior}")
        report.append(f"Surprise: {reaction.surprise_pct:.2f}% ({reaction.surprise_class})")

        report.append("\n" + "-" * 40)
        report.append("MARKET REACTIONS")
        report.append("-" * 40)

        for asset_class, class_reactions in reaction.reactions.items():
            report.append(f"\n{asset_class.upper()}:")
            for symbol, tf_reactions in class_reactions.items():
                report.append(f"  {symbol}:")
                for tf, value in tf_reactions.items():
                    direction = "▲" if value > 0 else "▼" if value < 0 else "─"
                    report.append(f"    {tf}: {direction} {value:+.3f}%")

        report.append("\n" + "-" * 40)
        report.append("VOLUME & VOLATILITY")
        report.append("-" * 40)
        report.append(f"Volume Spike: {reaction.volume_spike:.1f}x")
        report.append(f"Volume Change: {reaction.volume_change_pct:+.1f}%")
        report.append(f"Post-event Range: {reaction.range_pct:.2f}%")

        report.append("\n" + "=" * 60)

        return "\n".join(report)
