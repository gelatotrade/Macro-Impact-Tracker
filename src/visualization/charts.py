"""
Visualization Module

Creates interactive charts for visualizing market reactions to macro events.
Uses Plotly for interactive web-based charts.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class ReactionCharts:
    """
    Creates interactive visualizations for macro event impact analysis.
    """

    # Color schemes
    COLORS = {
        "positive": "#26a69a",  # Green
        "negative": "#ef5350",  # Red
        "neutral": "#78909c",   # Gray
        "event_line": "#ff9800",  # Orange
        "volume": "#42a5f5",    # Blue
        "equities": "#2196f3",
        "fx": "#9c27b0",
        "rates": "#ff5722",
        "commodities": "#ffc107",
    }

    ASSET_CLASS_COLORS = {
        "equities": ["#2196f3", "#1976d2", "#0d47a1", "#42a5f5"],
        "fx": ["#9c27b0", "#7b1fa2", "#4a148c", "#ba68c8"],
        "rates": ["#ff5722", "#e64a19", "#bf360c", "#ff8a65"],
        "commodities": ["#ffc107", "#ffa000", "#ff6f00", "#ffca28"],
    }

    def __init__(self):
        """Initialize the ReactionCharts."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization. Install with: pip install plotly")

    def create_price_reaction_chart(
        self,
        data: pd.DataFrame,
        event_time: datetime,
        symbol: str,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create a price chart showing reaction to an event.

        Args:
            data: DataFrame with OHLCV data
            event_time: Event timestamp
            symbol: Asset symbol
            title: Chart title

        Returns:
            Plotly Figure object
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=("Price", "Volume")
        )

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="Price",
                increasing_line_color=self.COLORS["positive"],
                decreasing_line_color=self.COLORS["negative"],
            ),
            row=1, col=1
        )

        # Add event line
        fig.add_vline(
            x=event_time,
            line_dash="dash",
            line_color=self.COLORS["event_line"],
            line_width=2,
            annotation_text="Event Release",
            annotation_position="top",
            row=1, col=1
        )

        # Volume bars
        colors = [
            self.COLORS["positive"] if data["Close"].iloc[i] >= data["Open"].iloc[i]
            else self.COLORS["negative"]
            for i in range(len(data))
        ]

        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data["Volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.7,
            ),
            row=2, col=1
        )

        fig.add_vline(
            x=event_time,
            line_dash="dash",
            line_color=self.COLORS["event_line"],
            line_width=2,
            row=2, col=1
        )

        fig.update_layout(
            title=title or f"{symbol} Reaction to Event",
            xaxis_rangeslider_visible=False,
            height=600,
            showlegend=False,
            template="plotly_dark",
        )

        return fig

    def create_normalized_comparison_chart(
        self,
        data_dict: Dict[str, pd.DataFrame],
        event_time: datetime,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create a chart comparing multiple assets normalized to event time.

        Args:
            data_dict: Dictionary of {symbol: DataFrame}
            event_time: Event timestamp
            title: Chart title

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        colors = self.COLORS["equities"]
        color_idx = 0

        for symbol, data in data_dict.items():
            if data.empty:
                continue

            # Find event index and normalize
            event_idx = self._find_event_index(data, event_time)
            base_price = data["Close"].iloc[event_idx]
            normalized = ((data["Close"] / base_price) - 1) * 100

            # Get color
            color = self.ASSET_CLASS_COLORS["equities"][color_idx % 4]
            color_idx += 1

            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=normalized,
                    mode="lines",
                    name=symbol,
                    line=dict(color=color, width=2),
                )
            )

        # Add event line
        fig.add_vline(
            x=event_time,
            line_dash="dash",
            line_color=self.COLORS["event_line"],
            line_width=2,
            annotation_text="Event",
            annotation_position="top right"
        )

        # Add zero line
        fig.add_hline(
            y=0,
            line_dash="dot",
            line_color="white",
            opacity=0.5
        )

        fig.update_layout(
            title=title or "Normalized Price Reaction Comparison",
            xaxis_title="Time",
            yaxis_title="Return (%)",
            height=500,
            template="plotly_dark",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
        )

        return fig

    def create_multi_asset_reaction_chart(
        self,
        market_data: Dict[str, Dict[str, pd.DataFrame]],
        event_time: datetime,
        event_info: Optional[Dict] = None
    ) -> go.Figure:
        """
        Create a comprehensive multi-asset reaction chart.

        Args:
            market_data: Nested dict of {asset_class: {symbol: DataFrame}}
            event_time: Event timestamp
            event_info: Optional event information

        Returns:
            Plotly Figure object
        """
        num_classes = len(market_data)
        fig = make_subplots(
            rows=num_classes, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[cls.upper() for cls in market_data.keys()]
        )

        row = 1
        for asset_class, symbols_data in market_data.items():
            colors = self.ASSET_CLASS_COLORS.get(
                asset_class,
                self.ASSET_CLASS_COLORS["equities"]
            )
            color_idx = 0

            for symbol, data in symbols_data.items():
                if data.empty:
                    continue

                # Normalize to event time
                event_idx = self._find_event_index(data, event_time)
                base_price = data["Close"].iloc[event_idx]
                normalized = ((data["Close"] / base_price) - 1) * 100

                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=normalized,
                        mode="lines",
                        name=f"{symbol}",
                        line=dict(color=colors[color_idx % 4], width=2),
                        legendgroup=asset_class,
                    ),
                    row=row, col=1
                )
                color_idx += 1

            # Add event line for each subplot
            fig.add_vline(
                x=event_time,
                line_dash="dash",
                line_color=self.COLORS["event_line"],
                line_width=1.5,
                row=row, col=1
            )

            row += 1

        # Build title
        title = "Multi-Asset Reaction"
        if event_info:
            title = f"{event_info.get('type', 'Event')} Impact - "
            title += f"Actual: {event_info.get('actual', 'N/A')} "
            title += f"vs Forecast: {event_info.get('forecast', 'N/A')}"

        fig.update_layout(
            title=title,
            height=300 * num_classes,
            template="plotly_dark",
            showlegend=True,
        )

        # Update y-axis labels
        for i in range(1, num_classes + 1):
            fig.update_yaxes(title_text="Return (%)", row=i, col=1)

        return fig

    def create_reaction_heatmap(
        self,
        reactions: List[Any],  # List of EventReaction objects
        asset_class: str = "equities"
    ) -> go.Figure:
        """
        Create a heatmap of reactions across events and timeframes.

        Args:
            reactions: List of EventReaction objects
            asset_class: Asset class to display

        Returns:
            Plotly Figure object
        """
        # Extract data
        events = []
        timeframes = ["1min", "5min", "15min", "30min", "1hour"]

        for r in reactions:
            if asset_class not in r.reactions:
                continue

            # Get first symbol in the class
            symbols = list(r.reactions[asset_class].keys())
            if not symbols:
                continue

            symbol = symbols[0]
            symbol_reactions = r.reactions[asset_class][symbol]

            event_label = f"{r.event_date}"
            events.append({
                "label": event_label,
                "surprise": r.surprise_pct,
                **{tf: symbol_reactions.get(tf, 0) for tf in timeframes}
            })

        if not events:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5)
            return fig

        df = pd.DataFrame(events)

        # Create heatmap data
        z_data = df[timeframes].values

        fig = go.Figure(
            data=go.Heatmap(
                z=z_data,
                x=timeframes,
                y=df["label"],
                colorscale=[
                    [0, self.COLORS["negative"]],
                    [0.5, "white"],
                    [1, self.COLORS["positive"]]
                ],
                zmid=0,
                text=np.round(z_data, 2),
                texttemplate="%{text}%",
                textfont={"size": 10},
                colorbar=dict(title="Return %"),
            )
        )

        fig.update_layout(
            title=f"{asset_class.title()} Reactions Across Events",
            xaxis_title="Timeframe",
            yaxis_title="Event Date",
            height=max(400, len(events) * 40),
            template="plotly_dark",
        )

        return fig

    def create_surprise_vs_reaction_scatter(
        self,
        reactions: List[Any],
        asset: str = "SPY",
        timeframe: str = "15min"
    ) -> go.Figure:
        """
        Create a scatter plot of surprise vs reaction.

        Args:
            reactions: List of EventReaction objects
            asset: Asset to analyze
            timeframe: Timeframe for reaction

        Returns:
            Plotly Figure object
        """
        data_points = []

        for r in reactions:
            for asset_class, class_reactions in r.reactions.items():
                for symbol, symbol_reactions in class_reactions.items():
                    if asset in symbol and timeframe in symbol_reactions:
                        data_points.append({
                            "date": r.event_date,
                            "surprise": r.surprise_pct,
                            "reaction": symbol_reactions[timeframe],
                            "class": r.surprise_class,
                        })

        if not data_points:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5)
            return fig

        df = pd.DataFrame(data_points)

        # Color by classification
        color_map = {
            "huge_beat": "#00c853",
            "beat": "#69f0ae",
            "inline": "#78909c",
            "miss": "#ff8a80",
            "huge_miss": "#ff1744",
        }
        colors = [color_map.get(c, "#78909c") for c in df["class"]]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df["surprise"],
                y=df["reaction"],
                mode="markers+text",
                text=df["date"],
                textposition="top center",
                marker=dict(
                    size=12,
                    color=colors,
                    line=dict(width=1, color="white")
                ),
                hovertemplate=(
                    "Date: %{text}<br>" +
                    "Surprise: %{x:.2f}%<br>" +
                    "Reaction: %{y:.2f}%<br>" +
                    "<extra></extra>"
                ),
            )
        )

        # Add trend line
        if len(df) > 2:
            z = np.polyfit(df["surprise"], df["reaction"], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df["surprise"].min(), df["surprise"].max(), 100)

            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=p(x_line),
                    mode="lines",
                    name="Trend",
                    line=dict(dash="dash", color="white", width=1),
                )
            )

        # Add quadrant lines
        fig.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.3)
        fig.add_vline(x=0, line_dash="dot", line_color="white", opacity=0.3)

        fig.update_layout(
            title=f"Surprise vs {asset} Reaction ({timeframe})",
            xaxis_title="Surprise (%)",
            yaxis_title=f"Reaction ({timeframe}) %",
            height=500,
            template="plotly_dark",
            showlegend=False,
        )

        return fig

    def create_volume_analysis_chart(
        self,
        data: pd.DataFrame,
        event_time: datetime,
        symbol: str
    ) -> go.Figure:
        """
        Create detailed volume analysis around event.

        Args:
            data: DataFrame with OHLCV data
            event_time: Event timestamp
            symbol: Asset symbol

        Returns:
            Plotly Figure object
        """
        event_idx = self._find_event_index(data, event_time)

        # Calculate volume metrics
        pre_vol = data["Volume"].iloc[:event_idx]
        post_vol = data["Volume"].iloc[event_idx:]

        pre_avg = pre_vol.mean() if len(pre_vol) > 0 else 0
        volume_normalized = data["Volume"] / pre_avg if pre_avg > 0 else data["Volume"]

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Normalized Volume", "Cumulative Volume")
        )

        # Normalized volume bars
        colors = [
            self.COLORS["positive"] if v > 1 else self.COLORS["negative"]
            for v in volume_normalized
        ]

        fig.add_trace(
            go.Bar(
                x=data.index,
                y=volume_normalized,
                name="Normalized Volume",
                marker_color=colors,
            ),
            row=1, col=1
        )

        # Add average line
        fig.add_hline(
            y=1, line_dash="dash", line_color="white",
            annotation_text="Pre-event Avg",
            row=1, col=1
        )

        # Cumulative volume
        cumulative = data["Volume"].cumsum()
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=cumulative,
                name="Cumulative Volume",
                fill="tozeroy",
                line=dict(color=self.COLORS["volume"]),
            ),
            row=2, col=1
        )

        # Event lines
        for row in [1, 2]:
            fig.add_vline(
                x=event_time,
                line_dash="dash",
                line_color=self.COLORS["event_line"],
                row=row, col=1
            )

        fig.update_layout(
            title=f"{symbol} Volume Analysis",
            height=500,
            template="plotly_dark",
            showlegend=False,
        )

        return fig

    def create_historical_reactions_chart(
        self,
        reactions: List[Any],
        event_type: str,
        asset: str = "SPY"
    ) -> go.Figure:
        """
        Create a chart showing historical reaction patterns.

        Args:
            reactions: List of EventReaction objects
            event_type: Type of event
            asset: Asset to display

        Returns:
            Plotly Figure object
        """
        timeframes = ["1min", "5min", "15min", "30min", "1hour", "2hour"]

        data = []
        for r in reactions:
            row = {"date": r.event_date, "surprise": r.surprise_pct}
            for asset_class, class_reactions in r.reactions.items():
                for symbol, symbol_reactions in class_reactions.items():
                    if asset in symbol:
                        for tf in timeframes:
                            row[tf] = symbol_reactions.get(tf, None)
            data.append(row)

        df = pd.DataFrame(data)

        fig = go.Figure()

        # Add box plots for each timeframe
        for tf in timeframes:
            if tf in df.columns:
                values = df[tf].dropna()
                if len(values) > 0:
                    fig.add_trace(
                        go.Box(
                            y=values,
                            name=tf,
                            boxpoints="all",
                            jitter=0.3,
                            pointpos=-1.5,
                            marker_color=self.COLORS["equities"],
                        )
                    )

        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)

        fig.update_layout(
            title=f"{event_type} Historical Reactions - {asset}",
            xaxis_title="Timeframe",
            yaxis_title="Reaction (%)",
            height=500,
            template="plotly_dark",
        )

        return fig

    def create_sector_reaction_chart(
        self,
        sector_data: Dict[str, pd.DataFrame],
        event_time: datetime,
        sector_names: Dict[str, str]
    ) -> go.Figure:
        """
        Create a chart showing sector reactions.

        Args:
            sector_data: Dictionary of {symbol: DataFrame}
            event_time: Event timestamp
            sector_names: Dictionary mapping symbols to names

        Returns:
            Plotly Figure object
        """
        reactions = {}

        for symbol, data in sector_data.items():
            if data.empty:
                continue

            event_idx = self._find_event_index(data, event_time)
            if event_idx + 15 >= len(data):
                continue

            base = data["Close"].iloc[event_idx]
            end = data["Close"].iloc[event_idx + 15]  # 15-min reaction
            reaction = ((end / base) - 1) * 100
            reactions[sector_names.get(symbol, symbol)] = reaction

        if not reactions:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5)
            return fig

        # Sort by reaction
        sorted_sectors = sorted(reactions.items(), key=lambda x: x[1])
        names = [s[0] for s in sorted_sectors]
        values = [s[1] for s in sorted_sectors]

        colors = [
            self.COLORS["positive"] if v >= 0 else self.COLORS["negative"]
            for v in values
        ]

        fig = go.Figure(
            go.Bar(
                x=values,
                y=names,
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.2f}%" for v in values],
                textposition="outside",
            )
        )

        fig.add_vline(x=0, line_color="white", line_width=1)

        fig.update_layout(
            title="Sector Reactions (15-minute)",
            xaxis_title="Return (%)",
            yaxis_title="Sector",
            height=max(400, len(reactions) * 35),
            template="plotly_dark",
        )

        return fig

    def create_dashboard_summary(
        self,
        event_info: Dict,
        reaction: Any,  # EventReaction
        key_assets: List[str] = None
    ) -> go.Figure:
        """
        Create a summary dashboard figure.

        Args:
            event_info: Event information dictionary
            reaction: EventReaction object
            key_assets: Key assets to highlight

        Returns:
            Plotly Figure object
        """
        if key_assets is None:
            key_assets = ["SPY", "EURUSD=X", "^TNX"]

        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}],
                [{"type": "table", "colspan": 2}, None]
            ],
            subplot_titles=("Event Surprise", "Volume Impact", "Reaction Summary")
        )

        # Surprise indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=reaction.surprise_pct,
                title={"text": "Surprise %"},
                delta={"reference": 0},
                gauge={
                    "axis": {"range": [-30, 30]},
                    "bar": {"color": "orange"},
                    "steps": [
                        {"range": [-30, -5], "color": "#ef5350"},
                        {"range": [-5, 5], "color": "#78909c"},
                        {"range": [5, 30], "color": "#26a69a"}
                    ],
                }
            ),
            row=1, col=1
        )

        # Volume spike indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=reaction.volume_spike,
                title={"text": "Volume Spike (x)"},
                gauge={
                    "axis": {"range": [0, 10]},
                    "bar": {"color": "#42a5f5"},
                    "steps": [
                        {"range": [0, 2], "color": "#78909c"},
                        {"range": [2, 5], "color": "#64b5f6"},
                        {"range": [5, 10], "color": "#1976d2"}
                    ],
                }
            ),
            row=1, col=2
        )

        # Reaction table
        table_data = []
        headers = ["Asset", "1min", "5min", "15min", "30min", "1hour"]

        for asset_class, class_reactions in reaction.reactions.items():
            for symbol, tf_reactions in class_reactions.items():
                row = [symbol]
                for tf in ["1min", "5min", "15min", "30min", "1hour"]:
                    val = tf_reactions.get(tf, 0)
                    row.append(f"{val:+.2f}%")
                table_data.append(row)

        if table_data:
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=headers,
                        fill_color="#424242",
                        align="center"
                    ),
                    cells=dict(
                        values=list(zip(*table_data)) if table_data else [[]],
                        fill_color="#212121",
                        align="center"
                    )
                ),
                row=2, col=1
            )

        fig.update_layout(
            title=f"{reaction.event_type} Impact Summary - {reaction.event_date}",
            height=600,
            template="plotly_dark",
        )

        return fig

    def _find_event_index(
        self,
        data: pd.DataFrame,
        event_time: datetime
    ) -> int:
        """Find the index closest to the event time."""
        if data.index.tz is None and event_time.tzinfo is not None:
            event_time = event_time.replace(tzinfo=None)
        elif data.index.tz is not None and event_time.tzinfo is None:
            event_time = event_time.tz_localize(data.index.tz)

        time_diffs = abs(data.index - event_time)
        return time_diffs.argmin()

    def save_chart(
        self,
        fig: go.Figure,
        filename: str,
        format: str = "html"
    ):
        """
        Save a chart to file.

        Args:
            fig: Plotly Figure object
            filename: Output filename
            format: Output format (html, png, pdf, svg)
        """
        if format == "html":
            fig.write_html(filename)
        else:
            fig.write_image(filename, format=format)
