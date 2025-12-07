# 📊 Macro Event Impact Tracker

A real-time system for tracking how financial markets react to macroeconomic events like CPI releases, Non-Farm Payrolls (NFP), FOMC decisions, PMI data, and more.

## Features

- **📅 Macro Event Database**: Track CPI, NFP, PMI, FOMC decisions, GDP, and other key economic releases
- **📈 Multi-Asset Analysis**: Analyze reactions across equities, FX, rates, and commodities
- **⏱️ Multiple Timeframes**: Track reactions at 1min, 5min, 15min, 30min, 1hour, and 2hour intervals
- **📊 Interactive Visualizations**: Beautiful Plotly charts including candlesticks, heatmaps, and scatter plots
- **🔍 Statistical Analysis**: Calculate surprise factors, correlations, and distribution statistics
- **🖥️ Web Dashboard**: Full-featured Streamlit dashboard for interactive exploration
- **🔌 API Integration**: Connect to FRED, TradingEconomics, and yfinance for real data

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Macro-Impact-Tracker.git
cd Macro-Impact-Tracker

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Add your API keys (optional but recommended):
```bash
# Edit .env file
FRED_API_KEY=your_fred_api_key_here
```

Get a free FRED API key at: https://fred.stlouisfed.org/docs/api/api_key.html

### Running the Dashboard

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

### Using the Python API

```python
from src.data.macro_events import MacroEventFetcher
from src.data.market_data import MarketDataFetcher
from src.analysis.impact_analyzer import ImpactAnalyzer
from src.visualization.charts import ReactionCharts

# Initialize components
macro_fetcher = MacroEventFetcher()
market_fetcher = MarketDataFetcher()
analyzer = ImpactAnalyzer(macro_fetcher, market_fetcher)
charts = ReactionCharts()

# Get recent CPI events
cpi_events = macro_fetcher.get_events_by_type("CPI")

# Analyze the latest event
latest_cpi = cpi_events[-1]
latest_cpi["type"] = "CPI"
reaction = analyzer.analyze_event(latest_cpi)

# Print the report
report = analyzer.generate_event_report(latest_cpi, reaction)
print(report)
```

## Project Structure

```
Macro-Impact-Tracker/
├── app.py                      # Streamlit dashboard
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── README.md                  # This file
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── macro_events.py    # Macro event data fetcher
│   │   └── market_data.py     # Market data fetcher (yfinance)
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── impact_analyzer.py # Impact analysis engine
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── charts.py          # Plotly visualization charts
│   │
│   └── utils/
│       ├── __init__.py
│       └── helpers.py         # Utility functions
│
├── config/
│   ├── __init__.py
│   └── settings.py            # Configuration settings
│
├── data/
│   ├── cache/                 # Cached data
│   └── events/                # Historical events
│
├── examples/
│   ├── basic_usage.py         # Basic usage example
│   ├── analyze_nfp.py         # NFP analysis example
│   └── live_monitor.py        # Live monitoring example
│
└── tests/                     # Unit tests
```

## Supported Economic Events

| Event | Description | Release Time (ET) |
|-------|-------------|-------------------|
| CPI | Consumer Price Index | 8:30 AM |
| NFP | Non-Farm Payrolls | 8:30 AM (1st Friday) |
| FOMC | Federal Reserve Decision | 2:00 PM |
| PMI | Purchasing Managers Index | 10:00 AM |
| GDP | Gross Domestic Product | 8:30 AM |

## Supported Assets

### Equities
- SPY (S&P 500 ETF)
- QQQ (Nasdaq 100 ETF)
- DIA (Dow Jones ETF)
- IWM (Russell 2000 ETF)
- VIX (Volatility Index)

### Foreign Exchange
- EUR/USD
- GBP/USD
- USD/JPY
- USD/CHF
- AUD/USD
- USD/CAD
- DXY (Dollar Index)

### Rates
- 10-Year Treasury Yield (^TNX)
- 5-Year Treasury Yield (^FVX)
- 30-Year Treasury Yield (^TYX)
- TLT (20+ Year Treasury ETF)
- IEF (7-10 Year Treasury ETF)

### Commodities
- Gold (GC=F)
- Silver (SI=F)
- Crude Oil (CL=F)
- Natural Gas (NG=F)

## Examples

### Run Basic Example
```bash
python examples/basic_usage.py
```

### Analyze NFP Events
```bash
python examples/analyze_nfp.py
```

### Live Event Monitor
```bash
python examples/live_monitor.py
```

### Simulate Event Release
```bash
python examples/live_monitor.py --simulate
```

## Dashboard Features

### 📅 Recent Events Tab
- View all recent economic events
- Filter by event type
- See surprise calculations
- Track beats vs misses

### 📈 Event Analysis Tab
- Select specific events to analyze
- View multi-asset reaction charts
- See normalized price movements
- Compare reactions across asset classes

### 🔥 Heatmaps Tab
- Visualize reactions across time
- Color-coded by direction and magnitude
- Surprise vs reaction scatter plots
- Identify patterns across events

### 📊 Statistics Tab
- Compare event types
- Distribution analysis
- Box plots of historical reactions
- Statistical significance metrics

### 🔍 Deep Dive Tab
- Single asset detailed analysis
- Candlestick charts with volume
- Sector rotation analysis
- Key price levels

## API Reference

### MacroEventFetcher

```python
fetcher = MacroEventFetcher(fred_api_key="your_key")

# Get events by type
events = fetcher.get_events_by_type("CPI")

# Get all events in date range
all_events = fetcher.get_all_events(
    start_date="2024-01-01",
    end_date="2024-12-31"
)

# Calculate surprise
surprise_pct, classification = fetcher.calculate_surprise(event)
```

### MarketDataFetcher

```python
fetcher = MarketDataFetcher()

# Get intraday data
data = fetcher.get_intraday_data(
    symbol="SPY",
    start="2024-01-01 08:00",
    end="2024-01-01 12:00",
    interval="1m"
)

# Get data around an event
event_data = fetcher.get_event_window_data(
    symbol="SPY",
    event_time=datetime(2024, 1, 11, 8, 30),
    pre_minutes=30,
    post_minutes=120
)

# Get multi-asset data
multi_data = fetcher.get_multi_asset_data(
    event_time=event_datetime,
    asset_classes=["equities", "fx", "rates"]
)
```

### ImpactAnalyzer

```python
analyzer = ImpactAnalyzer(macro_fetcher, market_fetcher)

# Analyze single event
reaction = analyzer.analyze_event(event)

# Get aggregate statistics
stats = analyzer.get_aggregate_stats("CPI", num_events=10)

# Compare event types
comparison = analyzer.compare_events(
    ["CPI", "NFP", "FOMC"],
    timeframe="15min",
    asset="SPY"
)

# Generate report
report = analyzer.generate_event_report(event, reaction)
```

### ReactionCharts

```python
charts = ReactionCharts()

# Price reaction chart
fig = charts.create_price_reaction_chart(data, event_time, "SPY")

# Multi-asset comparison
fig = charts.create_multi_asset_reaction_chart(market_data, event_time)

# Reaction heatmap
fig = charts.create_reaction_heatmap(reactions, "equities")

# Save chart
charts.save_chart(fig, "output.html")
```

## Data Sources

1. **FRED (Federal Reserve Economic Data)**: Historical economic indicators
2. **yfinance**: Market data (equities, FX, rates, commodities)
3. **Built-in Database**: Curated historical event data with actuals/forecasts

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Disclaimer

⚠️ **This tool is for educational and research purposes only. It is not financial advice. Always do your own research before making any investment decisions.**

## Support

For issues and feature requests, please open an issue on GitHub.
