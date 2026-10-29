"""
Presentation and theme helpers for the Sales Forecasting AI Dashboard.
Light Glass / iDraft aesthetic with polished chart styling and alignment.
Contains NO data processing, model inference, or evaluation logic.
"""

import os
import streamlit as st

# ──────────────────────────────────────────────────────────────────────
# Restrained Chart Palette (Preserved & Polished)
# ──────────────────────────────────────────────────────────────────────
COLOR_ACTUAL = "#292929"       # Charcoal for actual historical series
COLOR_PREDICTED = "#7C8490"    # Muted slate for predicted series
COLOR_FORECAST = "#5F8F89"     # Muted teal for future recursive forecast
COLOR_MA = "#8A8F98"           # Muted slate for 30-day moving average
COLOR_GRID = "rgba(30, 30, 30, 0.08)"
COLOR_AXIS_TEXT = "#737373"


def load_css():
    """Load and inject styles.css along with ambient background elements."""
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

    # Ambient background blurred blobs
    st.markdown(
        """
        <div class="blob b1"></div>
        <div class="blob b2"></div>
        <div class="blob b3"></div>
        """,
        unsafe_allow_html=True,
    )


def get_plotly_layout(height=260):
    """Return consistent, clutter-free Plotly layout settings for Light Glass theme."""
    return dict(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="'DM Sans', Inter, sans-serif", color="#1A1A1A", size=12),
        margin=dict(l=40, r=15, t=10, b=30),
        showlegend=False,  # Legend is displayed cleanly outside in the card header
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            tickfont=dict(family="'DM Sans', sans-serif", color=COLOR_AXIS_TEXT, size=11),
        ),
        yaxis=dict(
            gridcolor=COLOR_GRID,
            gridwidth=1,
            zeroline=False,
            showline=False,
            tickfont=dict(family="'DM Sans', sans-serif", color=COLOR_AXIS_TEXT, size=11),
        ),
        hoverlabel=dict(
            bgcolor="#292929",
            font_size=12,
            font_family="'DM Sans', sans-serif",
            font_color="#ffffff",
            bordercolor="#292929",
        ),
        height=height,
    )


def render_sidebar_header():
    """Render the SalesIQ brand and navigation pills in the sidebar."""
    st.sidebar.markdown(
        """
        <div class="brand">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#232323" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M3 17l6-6 4 4 8-8"></path>
                <path d="M15 7h6v6"></path>
            </svg>
            <span>SalesIQ</span>
        </div>
        <div class="nav-group">
            <a class="nav on" href="#kpi">
                <span>Dashboard</span>
            </a>
            <a class="nav" href="#historical-sales">
                <span>Historical Sales</span>
            </a>
            <a class="nav" href="#model-performance">
                <span>Model Performance</span>
            </a>
            <a class="nav" href="#seven-day-forecast">
                <span>7-Day Forecast</span>
            </a>
            <a class="nav" href="#dataset-explorer">
                <span>Dataset Explorer</span>
            </a>
        </div>
        <div class="grp">FILTERS</div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_footer():
    """Render the model info and project attribution in the sidebar."""
    st.sidebar.markdown(
        """
        <div class="grp" style="margin-top:10px;">MODEL</div>
        <div class="lbl" style="margin-top:0; font-weight:500; color:#3A3A3D;">Linear Regression</div>
        <div class="lbl" style="margin-top:0; color:#7A7A80;">Lag-based time-series features</div>
        <div class="foot">
            <b>Sales Forecast AI</b><br>
            B.Tech AI &amp; ML Mini-Project<br>
            Small Business Intelligence
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_header(start_date_str: str, end_date_str: str):
    """Render the main header banner with the active date range pill."""
    st.markdown(
        f"""
        <div class="top-header">
            <div>
                <h1>Sales Forecasting &amp; Business Intelligence</h1>
                <div class="sub">Historical sales analytics and machine-learning-based forecasts</div>
            </div>
            <div class="dpill">
                Selected range: <b>{start_date_str} &rarr; {end_date_str}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero_card(total_sales: int, avg_daily_sales: float, total_profit: float, avg_inventory: int):
    """Render the Overall Information dark charcoal hero card with inset light tiles."""
    st.markdown(
        f"""
        <div class="black">
            <div>
                <div class="sh">
                    <div class="st">Overall Information</div>
                    <span class="cap" style="margin:0;">Selected range</span>
                </div>
                <div class="big">
                    <div>
                        <div class="bv">{total_sales:,}</div>
                        <div class="bl">Total sales (units)</div>
                    </div>
                    <div>
                        <div class="bv">{avg_daily_sales:,}</div>
                        <div class="bl">Average daily sales</div>
                    </div>
                </div>
            </div>
            <div class="tiles">
                <div class="tile">
                    <div class="tv">&#8377;{total_profit:,.2f}</div>
                    <div class="tl">Total Recorded Profit</div>
                </div>
                <div class="tile">
                    <div class="tv">{avg_inventory:,}</div>
                    <div class="tl">Average Inventory</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_metric_tiles(mae: float, rmse: float):
    """Render the two dark charcoal metric tiles for MAE and RMSE."""
    st.markdown(
        f"""
        <div class="metric">
            <div class="m">
                <div class="tv">{mae:.2f}</div>
                <div class="tl">MAE</div>
            </div>
            <div class="m">
                <div class="tv">{rmse:.2f}</div>
                <div class="tl">RMSE</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_forecast_cards(forecast_rows: list):
    """Render the 7 daily forecast cards in a responsive grid."""
    cards_html = ""
    for r in forecast_rows:
        cards_html += f"""
        <div class="day">
            <div class="dl">{r['Date']}<br><span style="opacity:0.75; font-size:10.5px;">{r['Weekday'][:3]}</span></div>
            <div class="dv">{r['Forecasted Sales']}</div>
        </div>
        """
    st.markdown(
        f"""
        <div class="days">
            {cards_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
