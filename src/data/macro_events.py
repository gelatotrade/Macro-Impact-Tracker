"""
Macro Event Data Fetcher

Fetches macroeconomic event data from various sources:
- FRED API (Federal Reserve Economic Data)
- TradingEconomics API (if available)
- EconDB API (if available)
- Built-in historical event database
"""

import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False


class MacroEventFetcher:
    """
    Fetches and manages macroeconomic event data.

    Supports multiple data sources and maintains a local cache
    for historical events.
    """

    # FRED Series IDs for key economic indicators
    FRED_SERIES = {
        "CPI": "CPIAUCSL",           # CPI for All Urban Consumers
        "CPI_CORE": "CPILFESL",      # CPI Less Food and Energy
        "NFP": "PAYEMS",             # Total Nonfarm Payrolls
        "UNEMPLOYMENT": "UNRATE",     # Unemployment Rate
        "PMI_MFG": "MANEMP",         # Manufacturing Employment (proxy)
        "FED_FUNDS": "FEDFUNDS",     # Federal Funds Rate
        "GDP": "GDP",                 # Gross Domestic Product
        "RETAIL_SALES": "RSAFS",     # Retail Sales
        "INDUSTRIAL_PROD": "INDPRO", # Industrial Production
        "HOUSING_STARTS": "HOUST",   # Housing Starts
        "CONSUMER_CONF": "UMCSENT",  # Consumer Sentiment
        "PPI": "PPIACO",             # Producer Price Index
        "TRADE_BALANCE": "BOPGSTB",  # Trade Balance
        "PCE": "PCE",                # Personal Consumption Expenditures
    }

    # Known release times (Eastern Time) for major US economic releases
    RELEASE_TIMES = {
        "CPI": "08:30",
        "NFP": "08:30",
        "PMI": "10:00",
        "FED_FUNDS": "14:00",  # FOMC announcement
        "GDP": "08:30",
        "RETAIL_SALES": "08:30",
        "UNEMPLOYMENT": "08:30",
    }

    # Historical FOMC meeting dates (sample - expand as needed)
    FOMC_DATES_2024 = [
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
        "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18"
    ]

    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        trading_economics_key: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the MacroEventFetcher.

        Args:
            fred_api_key: FRED API key (or set FRED_API_KEY env var)
            trading_economics_key: TradingEconomics API key
            cache_dir: Directory for caching data
        """
        self.fred_api_key = fred_api_key or os.getenv("FRED_API_KEY")
        self.te_api_key = trading_economics_key or os.getenv("TRADING_ECONOMICS_KEY")

        self.fred = None
        if FRED_AVAILABLE and self.fred_api_key:
            self.fred = Fred(api_key=self.fred_api_key)

        # Setup cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(__file__).parent.parent.parent / "data" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load historical events database
        self._load_historical_events()

    def _load_historical_events(self):
        """Load or create historical events database."""
        self.events_file = self.cache_dir / "historical_events.json"

        if self.events_file.exists():
            with open(self.events_file, "r") as f:
                self.historical_events = json.load(f)
        else:
            # Initialize with sample historical events
            self.historical_events = self._create_sample_events()
            self._save_historical_events()

    def _save_historical_events(self):
        """Save historical events to file."""
        with open(self.events_file, "w") as f:
            json.dump(self.historical_events, f, indent=2, default=str)

    def _create_sample_events(self) -> Dict[str, List[Dict]]:
        """Create sample historical events database."""
        events = {
            "CPI": [],
            "NFP": [],
            "FOMC": [],
            "PMI": [],
            "GDP": [],
        }

        # Sample CPI releases (2024)
        cpi_dates = [
            ("2024-01-11", 3.4, 3.2, 3.1),
            ("2024-02-13", 3.1, 2.9, 3.4),
            ("2024-03-12", 3.2, 3.1, 3.1),
            ("2024-04-10", 3.5, 3.4, 3.2),
            ("2024-05-15", 3.4, 3.4, 3.5),
            ("2024-06-12", 3.3, 3.4, 3.4),
            ("2024-07-11", 3.0, 3.1, 3.3),
            ("2024-08-14", 2.9, 3.0, 3.0),
            ("2024-09-11", 2.5, 2.6, 2.9),
            ("2024-10-10", 2.4, 2.3, 2.5),
            ("2024-11-13", 2.6, 2.6, 2.4),
        ]

        for date, actual, forecast, prior in cpi_dates:
            events["CPI"].append({
                "date": date,
                "time": "08:30",
                "timezone": "America/New_York",
                "actual": actual,
                "forecast": forecast,
                "prior": prior,
                "unit": "% YoY",
                "importance": "high",
            })

        # Sample NFP releases (2024)
        nfp_dates = [
            ("2024-01-05", 216, 175, 173),
            ("2024-02-02", 353, 185, 333),
            ("2024-03-08", 275, 200, 229),
            ("2024-04-05", 303, 200, 270),
            ("2024-05-03", 175, 240, 315),
            ("2024-06-07", 272, 180, 165),
            ("2024-07-05", 206, 190, 218),
            ("2024-08-02", 114, 175, 179),
            ("2024-09-06", 142, 160, 89),
            ("2024-10-04", 254, 140, 159),
            ("2024-11-01", 12, 100, 223),
        ]

        for date, actual, forecast, prior in nfp_dates:
            events["NFP"].append({
                "date": date,
                "time": "08:30",
                "timezone": "America/New_York",
                "actual": actual,
                "forecast": forecast,
                "prior": prior,
                "unit": "K jobs",
                "importance": "high",
            })

        # Sample FOMC decisions (2024)
        fomc_dates = [
            ("2024-01-31", 5.50, 5.50, 5.50, "hold"),
            ("2024-03-20", 5.50, 5.50, 5.50, "hold"),
            ("2024-05-01", 5.50, 5.50, 5.50, "hold"),
            ("2024-06-12", 5.50, 5.50, 5.50, "hold"),
            ("2024-07-31", 5.50, 5.50, 5.50, "hold"),
            ("2024-09-18", 5.00, 5.25, 5.50, "cut"),
            ("2024-11-07", 4.75, 4.75, 5.00, "cut"),
        ]

        for date, actual, forecast, prior, action in fomc_dates:
            events["FOMC"].append({
                "date": date,
                "time": "14:00",
                "timezone": "America/New_York",
                "actual": actual,
                "forecast": forecast,
                "prior": prior,
                "action": action,
                "unit": "%",
                "importance": "high",
            })

        # Sample PMI releases (2024)
        pmi_dates = [
            ("2024-01-03", 47.4, 47.1, 46.7),
            ("2024-02-01", 49.1, 47.0, 47.4),
            ("2024-03-01", 47.8, 49.5, 49.1),
            ("2024-04-01", 50.3, 48.5, 47.8),
            ("2024-05-01", 49.2, 50.0, 50.3),
            ("2024-06-03", 48.7, 49.5, 49.2),
            ("2024-07-01", 46.8, 49.0, 48.7),
            ("2024-08-01", 47.2, 48.0, 46.8),
            ("2024-09-03", 47.2, 47.5, 47.2),
            ("2024-10-01", 47.2, 47.5, 47.2),
            ("2024-11-01", 46.5, 47.5, 47.2),
        ]

        for date, actual, forecast, prior in pmi_dates:
            events["PMI"].append({
                "date": date,
                "time": "10:00",
                "timezone": "America/New_York",
                "actual": actual,
                "forecast": forecast,
                "prior": prior,
                "unit": "index",
                "importance": "high",
            })

        return events

    def get_fred_data(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.Series]:
        """
        Fetch data from FRED API.

        Args:
            series_id: FRED series identifier
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Pandas Series with the data or None if unavailable
        """
        if not self.fred:
            print("FRED API not available. Set FRED_API_KEY environment variable.")
            return None

        try:
            data = self.fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date
            )
            return data
        except Exception as e:
            print(f"Error fetching FRED data for {series_id}: {e}")
            return None

    def get_indicator_history(
        self,
        indicator: str,
        periods: int = 12
    ) -> Optional[pd.DataFrame]:
        """
        Get historical values for an indicator.

        Args:
            indicator: Indicator name (CPI, NFP, etc.)
            periods: Number of periods to fetch

        Returns:
            DataFrame with historical data
        """
        # First try FRED
        if indicator.upper() in self.FRED_SERIES and self.fred:
            series_id = self.FRED_SERIES[indicator.upper()]
            data = self.get_fred_data(series_id)
            if data is not None:
                return pd.DataFrame({
                    "date": data.index,
                    "value": data.values
                }).tail(periods)

        # Fall back to historical events
        indicator_key = indicator.upper()
        if indicator_key in self.historical_events:
            events = self.historical_events[indicator_key][-periods:]
            return pd.DataFrame(events)

        return None

    def get_events_by_type(
        self,
        event_type: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get events of a specific type within a date range.

        Args:
            event_type: Type of event (CPI, NFP, FOMC, PMI)
            start_date: Start date filter
            end_date: End date filter

        Returns:
            List of event dictionaries
        """
        event_type = event_type.upper()
        if event_type not in self.historical_events:
            return []

        events = self.historical_events[event_type]

        if start_date:
            events = [e for e in events if e["date"] >= start_date]
        if end_date:
            events = [e for e in events if e["date"] <= end_date]

        return events

    def get_all_events(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        importance: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get all events within a date range.

        Args:
            start_date: Start date filter
            end_date: End date filter
            importance: Filter by importance (high, medium, low)

        Returns:
            DataFrame with all events
        """
        all_events = []

        for event_type, events in self.historical_events.items():
            for event in events:
                event_copy = event.copy()
                event_copy["type"] = event_type
                all_events.append(event_copy)

        df = pd.DataFrame(all_events)

        if df.empty:
            return df

        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
        df = df.sort_values("datetime")

        if start_date:
            df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]
        if importance:
            df = df[df["importance"] == importance]

        return df

    def get_upcoming_events(self, days: int = 7) -> pd.DataFrame:
        """
        Get upcoming economic events.

        Args:
            days: Number of days to look ahead

        Returns:
            DataFrame with upcoming events
        """
        today = datetime.now().strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")

        return self.get_all_events(start_date=today, end_date=end_date)

    def get_recent_events(self, days: int = 30) -> pd.DataFrame:
        """
        Get recent economic events.

        Args:
            days: Number of days to look back

        Returns:
            DataFrame with recent events
        """
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        today = datetime.now().strftime("%Y-%m-%d")

        return self.get_all_events(start_date=start_date, end_date=today)

    def calculate_surprise(self, event: Dict) -> Tuple[float, str]:
        """
        Calculate the surprise factor for an event.

        Args:
            event: Event dictionary with actual, forecast, prior

        Returns:
            Tuple of (surprise_percentage, classification)
        """
        actual = event.get("actual", 0)
        forecast = event.get("forecast", 0)

        if forecast == 0:
            surprise_pct = 0
        else:
            surprise_pct = ((actual - forecast) / abs(forecast)) * 100

        # Classify the surprise
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

    def add_event(
        self,
        event_type: str,
        date: str,
        time: str,
        actual: float,
        forecast: float,
        prior: float,
        **kwargs
    ):
        """
        Add a new event to the historical database.

        Args:
            event_type: Type of event
            date: Date string (YYYY-MM-DD)
            time: Time string (HH:MM)
            actual: Actual value
            forecast: Forecast value
            prior: Prior value
            **kwargs: Additional event properties
        """
        event_type = event_type.upper()

        if event_type not in self.historical_events:
            self.historical_events[event_type] = []

        event = {
            "date": date,
            "time": time,
            "timezone": kwargs.get("timezone", "America/New_York"),
            "actual": actual,
            "forecast": forecast,
            "prior": prior,
            "unit": kwargs.get("unit", ""),
            "importance": kwargs.get("importance", "high"),
        }
        event.update(kwargs)

        self.historical_events[event_type].append(event)
        self._save_historical_events()

    def fetch_trading_economics(
        self,
        indicator: str,
        country: str = "united states"
    ) -> Optional[List[Dict]]:
        """
        Fetch data from TradingEconomics API.

        Args:
            indicator: Indicator name
            country: Country name

        Returns:
            List of data points or None
        """
        if not self.te_api_key:
            return None

        try:
            base_url = "https://api.tradingeconomics.com"
            endpoint = f"/historical/country/{country}/indicator/{indicator}"
            params = {"c": self.te_api_key, "format": "json"}

            response = requests.get(base_url + endpoint, params=params)
            response.raise_for_status()

            return response.json()
        except Exception as e:
            print(f"Error fetching TradingEconomics data: {e}")
            return None

    def get_event_impact_summary(self, event_type: str) -> Dict[str, Any]:
        """
        Get summary statistics for an event type.

        Args:
            event_type: Type of event

        Returns:
            Dictionary with summary statistics
        """
        events = self.get_events_by_type(event_type)

        if not events:
            return {}

        surprises = []
        beats = 0
        misses = 0
        inline = 0

        for event in events:
            surprise_pct, classification = self.calculate_surprise(event)
            surprises.append(surprise_pct)

            if "beat" in classification:
                beats += 1
            elif "miss" in classification:
                misses += 1
            else:
                inline += 1

        return {
            "total_events": len(events),
            "avg_surprise": np.mean(surprises),
            "max_surprise": np.max(surprises),
            "min_surprise": np.min(surprises),
            "std_surprise": np.std(surprises),
            "beats": beats,
            "misses": misses,
            "inline": inline,
            "beat_rate": beats / len(events) * 100,
        }
