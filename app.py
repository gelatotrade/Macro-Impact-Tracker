"""
Macro Event Impact Tracker - Streamlit Dashboard

A real-time dashboard for tracking how markets react to macroeconomic events.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import pytz

# Import our modules
from src.data.macro_events import MacroEventFetcher
from src.data.market_data import MarketDataFetcher
from src.analysis.impact_analyzer import ImpactAnalyzer
from src.visualization.charts import ReactionCharts

# Page configuration
st.set_page_config(
    page_title="Macro Event Impact Tracker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
    }
    .positive {
        color: #26a69a;
    }
    .negative {
        color: #ef5350;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_data_fetchers():
    """Initialize data fetchers (cached)."""
    macro_fetcher = MacroEventFetcher()
    market_fetcher = MarketDataFetcher()
    return macro_fetcher, market_fetcher


@st.cache_resource
def get_analyzer(_macro_fetcher, _market_fetcher):
    """Initialize analyzer (cached)."""
    return ImpactAnalyzer(_macro_fetcher, _market_fetcher)


@st.cache_resource
def get_charts():
    """Initialize charts (cached)."""
    return ReactionCharts()


def main():
    """Main application."""
    # Header
    st.markdown('<p class="main-header">📊 Macro Event Impact Tracker</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Initialize components
    macro_fetcher, market_fetcher = get_data_fetchers()
    analyzer = get_analyzer(macro_fetcher, market_fetcher)
    charts = get_charts()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Event type selection
        event_type = st.selectbox(
            "Event Type",
            ["CPI", "NFP", "FOMC", "PMI", "GDP"],
            index=0
        )

        # Date range
        st.subheader("Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start",
                datetime(2024, 1, 1),
                min_value=datetime(2020, 1, 1),
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "End",
                datetime.now(),
                min_value=datetime(2020, 1, 1),
                max_value=datetime.now()
            )

        # Asset selection
        st.subheader("Assets to Analyze")
        asset_classes = st.multiselect(
            "Asset Classes",
            ["equities", "fx", "rates", "commodities"],
            default=["equities", "fx", "rates"]
        )

        # Timeframe
        timeframe = st.selectbox(
            "Primary Timeframe",
            ["1min", "5min", "15min", "30min", "1hour", "2hour"],
            index=2
        )

        st.markdown("---")

        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Main content - Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📅 Recent Events",
        "📈 Event Analysis",
        "🔥 Heatmaps",
        "📊 Statistics",
        "🔍 Deep Dive"
    ])

    # Tab 1: Recent Events
    with tab1:
        st.header("Recent Economic Events")

        # Get recent events
        events_df = macro_fetcher.get_all_events(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )

        if not events_df.empty:
            # Filter by event type if selected
            if event_type:
                events_df = events_df[events_df["type"] == event_type]

            # Calculate surprises
            events_df["surprise"] = events_df.apply(
                lambda x: ((x["actual"] - x["forecast"]) / abs(x["forecast"]) * 100)
                if x["forecast"] != 0 else 0,
                axis=1
            )

            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Events", len(events_df))
            with col2:
                avg_surprise = events_df["surprise"].mean()
                st.metric("Avg Surprise", f"{avg_surprise:.1f}%")
            with col3:
                beats = len(events_df[events_df["surprise"] > 2])
                st.metric("Beats", beats)
            with col4:
                misses = len(events_df[events_df["surprise"] < -2])
                st.metric("Misses", misses)

            # Events table
            st.subheader("Events List")
            display_cols = ["date", "time", "type", "actual", "forecast", "prior", "surprise"]
            available_cols = [c for c in display_cols if c in events_df.columns]
            st.dataframe(
                events_df[available_cols].sort_values("date", ascending=False),
                use_container_width=True,
                height=400
            )
        else:
            st.info("No events found for the selected criteria.")

    # Tab 2: Event Analysis
    with tab2:
        st.header(f"{event_type} Event Analysis")

        # Get events for the selected type
        events = macro_fetcher.get_events_by_type(event_type)

        if events:
            # Event selector
            event_options = [f"{e['date']} - Actual: {e['actual']}" for e in events]
            selected_event_idx = st.selectbox(
                "Select Event",
                range(len(event_options)),
                format_func=lambda x: event_options[x]
            )

            selected_event = events[selected_event_idx]
            selected_event["type"] = event_type

            # Event details
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Actual", selected_event["actual"])
            with col2:
                st.metric("Forecast", selected_event["forecast"])
            with col3:
                st.metric("Prior", selected_event["prior"])
            with col4:
                surprise = ((selected_event["actual"] - selected_event["forecast"]) /
                           abs(selected_event["forecast"]) * 100) if selected_event["forecast"] != 0 else 0
                st.metric("Surprise", f"{surprise:.1f}%")

            # Analyze the event
            with st.spinner("Analyzing event impact..."):
                # Build assets dict
                assets_to_analyze = {}
                all_symbols = market_fetcher.get_all_symbols()
                for ac in asset_classes:
                    if ac in all_symbols:
                        assets_to_analyze[ac] = list(all_symbols[ac].keys())[:3]

                reaction = analyzer.analyze_event(
                    selected_event,
                    assets=assets_to_analyze
                )

                # Create event datetime
                event_datetime = pd.to_datetime(
                    f"{selected_event['date']} {selected_event['time']}"
                ).tz_localize("America/New_York")

                # Get market data for visualization
                market_data = market_fetcher.get_multi_asset_data(
                    event_datetime,
                    pre_minutes=30,
                    post_minutes=120,
                    asset_classes=asset_classes
                )

            # Display charts
            if market_data:
                st.subheader("Multi-Asset Reaction")
                fig = charts.create_multi_asset_reaction_chart(
                    market_data,
                    event_datetime,
                    selected_event
                )
                st.plotly_chart(fig, use_container_width=True)

            # Reaction summary
            st.subheader("Reaction Summary")
            reaction_data = []
            for asset_class, class_reactions in reaction.reactions.items():
                for symbol, tf_reactions in class_reactions.items():
                    row = {"Asset Class": asset_class, "Symbol": symbol}
                    row.update(tf_reactions)
                    reaction_data.append(row)

            if reaction_data:
                reaction_df = pd.DataFrame(reaction_data)
                st.dataframe(reaction_df, use_container_width=True)

        else:
            st.warning(f"No {event_type} events found.")

    # Tab 3: Heatmaps
    with tab3:
        st.header("Reaction Heatmaps")

        # Get historical reactions
        with st.spinner("Loading historical reactions..."):
            reactions = analyzer.analyze_event_history(event_type, num_events=10)

        if reactions:
            # Asset class selector for heatmap
            heatmap_class = st.selectbox(
                "Select Asset Class for Heatmap",
                asset_classes,
                index=0 if asset_classes else 0
            )

            # Create heatmap
            fig = charts.create_reaction_heatmap(reactions, heatmap_class)
            st.plotly_chart(fig, use_container_width=True)

            # Surprise vs Reaction scatter
            st.subheader("Surprise vs Reaction")
            primary_asset = st.selectbox(
                "Primary Asset",
                ["SPY", "QQQ", "EURUSD=X", "^TNX"],
                index=0
            )
            fig_scatter = charts.create_surprise_vs_reaction_scatter(
                reactions,
                asset=primary_asset,
                timeframe=timeframe
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("No historical data available for heatmap.")

    # Tab 4: Statistics
    with tab4:
        st.header("Statistical Analysis")

        # Compare event types
        st.subheader("Event Type Comparison")

        event_types_to_compare = st.multiselect(
            "Select Event Types to Compare",
            ["CPI", "NFP", "FOMC", "PMI", "GDP"],
            default=["CPI", "NFP", "FOMC"]
        )

        if event_types_to_compare:
            comparison_df = analyzer.compare_events(
                event_types_to_compare,
                timeframe=timeframe,
                asset="SPY"
            )

            if not comparison_df.empty:
                st.dataframe(comparison_df, use_container_width=True)

                # Bar chart of average reactions
                import plotly.express as px
                fig = px.bar(
                    comparison_df,
                    x="event_type",
                    y=f"avg_reaction_{timeframe}",
                    color=f"avg_reaction_{timeframe}",
                    color_continuous_scale=["#ef5350", "#78909c", "#26a69a"],
                    title=f"Average {timeframe} Reaction by Event Type"
                )
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

        # Distribution analysis
        st.subheader("Reaction Distribution")

        dist_stats = analyzer.get_reaction_distribution(
            event_type,
            asset="SPY",
            timeframe=timeframe
        )

        if dist_stats:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{dist_stats.get('mean', 0):.3f}%")
            with col2:
                st.metric("Median", f"{dist_stats.get('median', 0):.3f}%")
            with col3:
                st.metric("Std Dev", f"{dist_stats.get('std', 0):.3f}%")
            with col4:
                st.metric("Positive Rate", f"{dist_stats.get('positive_rate', 0):.1f}%")

            # Historical reactions box plot
            if reactions:
                fig_hist = charts.create_historical_reactions_chart(
                    reactions,
                    event_type,
                    asset="SPY"
                )
                st.plotly_chart(fig_hist, use_container_width=True)

    # Tab 5: Deep Dive
    with tab5:
        st.header("Deep Dive Analysis")

        # Single asset detailed analysis
        st.subheader("Single Asset Analysis")

        col1, col2 = st.columns(2)
        with col1:
            deep_dive_asset = st.selectbox(
                "Select Asset",
                ["SPY", "QQQ", "IWM", "DIA", "EURUSD=X", "USDJPY=X", "^TNX", "TLT", "GC=F"],
                index=0
            )
        with col2:
            if events:
                deep_dive_event_idx = st.selectbox(
                    "Select Event for Deep Dive",
                    range(len(events)),
                    format_func=lambda x: f"{events[x]['date']} - {event_type}",
                    key="deep_dive_event"
                )
                deep_dive_event = events[deep_dive_event_idx]

        if events and deep_dive_event:
            event_datetime = pd.to_datetime(
                f"{deep_dive_event['date']} {deep_dive_event['time']}"
            ).tz_localize("America/New_York")

            # Get detailed data
            asset_data = market_fetcher.get_event_window_data(
                deep_dive_asset,
                event_datetime,
                pre_minutes=60,
                post_minutes=180
            )

            if asset_data is not None and not asset_data.empty:
                # Price chart
                st.subheader(f"{deep_dive_asset} Price Action")
                fig_price = charts.create_price_reaction_chart(
                    asset_data,
                    event_datetime,
                    deep_dive_asset,
                    title=f"{deep_dive_asset} Around {event_type} Release ({deep_dive_event['date']})"
                )
                st.plotly_chart(fig_price, use_container_width=True)

                # Volume analysis
                st.subheader("Volume Analysis")
                fig_volume = charts.create_volume_analysis_chart(
                    asset_data,
                    event_datetime,
                    deep_dive_asset
                )
                st.plotly_chart(fig_volume, use_container_width=True)

                # Key statistics
                st.subheader("Key Statistics")
                event_idx = (asset_data.index - event_datetime).argmin()
                volume_profile = market_fetcher.get_volume_profile(asset_data, event_idx)
                key_levels = market_fetcher.get_key_levels(asset_data, event_idx)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Volume Spike", f"{volume_profile.get('volume_spike', 0):.1f}x")
                    st.metric("Pre-Event High", f"${key_levels.get('pre_event_high', 0):.2f}")
                with col2:
                    st.metric("Event Price", f"${key_levels.get('event_price', 0):.2f}")
                    st.metric("Post-Event High", f"${key_levels.get('post_event_high', 0):.2f}")
                with col3:
                    st.metric("Range Expansion", f"{key_levels.get('range_expansion', 0):.2f}x")
                    st.metric("Post-Event Low", f"${key_levels.get('post_event_low', 0):.2f}")

        # Sector analysis
        st.subheader("Sector Analysis")
        if events and st.button("Run Sector Analysis"):
            with st.spinner("Analyzing sector reactions..."):
                sector_data = market_fetcher.get_sector_data(
                    event_datetime,
                    pre_minutes=30,
                    post_minutes=60
                )

                if sector_data:
                    fig_sector = charts.create_sector_reaction_chart(
                        sector_data,
                        event_datetime,
                        market_fetcher.SECTOR_ETFS
                    )
                    st.plotly_chart(fig_sector, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Macro Event Impact Tracker | Data updates in real-time when markets are open</p>
            <p>⚠️ For educational purposes only. Not financial advice.</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
