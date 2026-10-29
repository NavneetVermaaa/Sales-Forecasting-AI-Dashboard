"""
AI-Based Sales Forecasting & Business Intelligence Dashboard
=============================================================
B.Tech AI & ML Mini-Project — Third Year

Loads a pre-trained Linear Regression time-series model (sales_model.pkl)
and the historical sales dataset (sales_data.csv) to present interactive
KPIs, trend analysis, model evaluation, and a 7-day recursive forecast.

Redesigned & polished with the Light Glass / iDraft aesthetic.
"""

import warnings
warnings.filterwarnings("ignore")

import os
from datetime import timedelta
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

import theme

# ──────────────────────────────────────────────────────────────────────
# Page configuration & Theme Initialization
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Forecast AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS & ambient background effects
theme.load_css()

# ──────────────────────────────────────────────────────────────────────
# Data loading & feature engineering (cached)
# ──────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "Sales_Lag_1",
    "Sales_Lag_7",
    "Rolling_Mean_7",
    "Weekday",
    "Month",
    "Day_Of_Year",
]


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """Load and validate the sales CSV."""
    if not os.path.exists(path):
        st.error(f"Dataset file not found: `{path}`")
        st.stop()

    df = pd.read_csv(path)

    # Validate required columns
    required = {"Date", "Sales", "Profit", "inventory"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Missing required columns in dataset: {missing}")
        st.stop()

    # Parse dates safely
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    invalid_dates = df["Date"].isna().sum()
    if invalid_dates > 0:
        st.warning(f"⚠️ {invalid_dates} rows with invalid dates were dropped.")
        df = df.dropna(subset=["Date"])

    df = df.sort_values("Date").reset_index(drop=True)
    return df


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """Load the pre-trained model from a joblib-pickled file."""
    if not os.path.exists(path):
        st.error(f"Model file not found: `{path}`")
        st.stop()
    try:
        artifact = joblib.load(path)
        if isinstance(artifact, dict):
            model = artifact.get("model", artifact)
        else:
            model = artifact
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the time-series features expected by the model."""
    df = df.copy()
    df["Sales_Lag_1"] = df["Sales"].shift(1)
    df["Sales_Lag_7"] = df["Sales"].shift(7)
    df["Rolling_Mean_7"] = df["Sales"].rolling(window=7).mean()
    df["Weekday"] = df["Date"].dt.weekday
    df["Month"] = df["Date"].dt.month
    df["Day_Of_Year"] = df["Date"].dt.dayofyear
    return df


# ──────────────────────────────────────────────────────────────────────
# Load assets
# ──────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "sales_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "sales_model.pkl")

df_raw = load_data(DATA_PATH)
model = load_model(MODEL_PATH)

# Engineer features on the full dataset
df_feat = engineer_features(df_raw)
df_feat = df_feat.dropna(subset=FEATURE_COLS).reset_index(drop=True)

# ──────────────────────────────────────────────────────────────────────
# Sidebar — branding & filters
# ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    theme.render_sidebar_header()

    min_date = df_raw["Date"].min().date()
    max_date = df_raw["Date"].max().date()

    start_date = st.date_input(
        "Start date",
        value=min_date,
        min_value=min_date,
        max_value=max_date,
        key="start_date",
    )
    end_date = st.date_input(
        "End date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        key="end_date",
    )

    if start_date > end_date:
        st.error("Start date must be before end date.")
        st.stop()

    theme.render_sidebar_footer()

# Apply date filter to raw data (for KPIs / trend)
mask = (df_raw["Date"].dt.date >= start_date) & (df_raw["Date"].dt.date <= end_date)
df_filtered = df_raw.loc[mask].copy()

if df_filtered.empty:
    st.warning("No data in the selected date range. Please adjust the filters.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────
# Main Header
# ──────────────────────────────────────────────────────────────────────
theme.render_top_header(
    start_date.strftime("%d %b %Y"),
    end_date.strftime("%d %b %Y")
)

# ──────────────────────────────────────────────────────────────────────
# Row 1: Overall Information (Hero Card) + Historical Sales Card
# ──────────────────────────────────────────────────────────────────────
total_sales = int(df_filtered["Sales"].sum())
avg_daily_sales = round(df_filtered["Sales"].mean(), 1)
total_profit = round(df_filtered["Profit"].sum(), 2)
avg_inventory = round(df_filtered["inventory"].mean(), 0)

col_hero, col_hist = st.columns([5, 7], gap="medium")

with col_hero:
    st.markdown('<div class="panel-hero"></div><div id="kpi"></div>', unsafe_allow_html=True)
    theme.render_hero_card(
        total_sales=total_sales,
        avg_daily_sales=avg_daily_sales,
        total_profit=total_profit,
        avg_inventory=int(avg_inventory),
    )

with col_hist:
    st.markdown('<div class="panel-glass"></div><div id="historical-sales"></div>', unsafe_allow_html=True)
    has_ma = len(df_filtered) >= 30
    legend_html = (
        f'<span><i></i>Actual sales</span><span><i class="ma"></i>30D Moving Avg</span>'
        if has_ma
        else '<span><i></i>Actual sales</span>'
    )

    st.markdown(
        f"""
        <div class="sh">
            <div class="st">Historical Sales</div>
            <div class="legend">{legend_html}</div>
        </div>
        <div class="cap">Daily sales across the selected date range</div>
        """,
        unsafe_allow_html=True,
    )

    fig_trend = go.Figure()
    fig_trend.add_trace(
        go.Scatter(
            x=df_filtered["Date"],
            y=df_filtered["Sales"],
            mode="lines",
            name="Daily Sales",
            line=dict(color=theme.COLOR_ACTUAL, width=1.8),
            fill="tozeroy",
            fillcolor="rgba(41, 41, 41, 0.04)",
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Sales: %{y:,}<extra></extra>",
        )
    )

    # 30-day moving average overlay
    if has_ma:
        df_filtered["Rolling_30"] = df_filtered["Sales"].rolling(30).mean()
        fig_trend.add_trace(
            go.Scatter(
                x=df_filtered["Date"],
                y=df_filtered["Rolling_30"],
                mode="lines",
                name="30-Day Moving Avg",
                line=dict(color=theme.COLOR_MA, width=1.6, dash="dash"),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>30-Day Avg: %{y:.1f}<extra></extra>",
            )
        )

    layout_trend = theme.get_plotly_layout(height=245)
    fig_trend.update_layout(**layout_trend, hovermode="x unified")
    st.plotly_chart(fig_trend, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────
# Row 2: Model Performance (Actual vs Predicted) + 7-Day Forecast Chart
# ──────────────────────────────────────────────────────────────────────
col_eval, col_fc = st.columns([1, 1], gap="medium")

with col_eval:
    st.markdown('<div class="panel-glass"></div><div id="model-performance"></div>', unsafe_allow_html=True)
    split_idx = int(len(df_feat) * 0.8)
    df_test = df_feat.iloc[split_idx:].copy()

    X_test = df_test[FEATURE_COLS]
    y_test = df_test["Sales"]
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.markdown(
        """
        <div class="sh">
            <div class="st">Model Performance</div>
            <div class="legend">
                <span><i></i>Actual</span>
                <span><i class="pred"></i>Predicted</span>
            </div>
        </div>
        <div class="cap">Error in sales units; lower is better.</div>
        """,
        unsafe_allow_html=True,
    )

    theme.render_model_metric_tiles(mae=mae, rmse=rmse)

    fig_eval = go.Figure()
    fig_eval.add_trace(
        go.Scatter(
            x=df_test["Date"],
            y=y_test,
            mode="lines",
            name="Actual",
            line=dict(color=theme.COLOR_ACTUAL, width=1.8),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Actual: %{y:,}<extra></extra>",
        )
    )
    fig_eval.add_trace(
        go.Scatter(
            x=df_test["Date"],
            y=y_pred,
            mode="lines",
            name="Predicted",
            line=dict(color=theme.COLOR_PREDICTED, width=1.8, dash="dash"),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Predicted: %{y:.1f}<extra></extra>",
        )
    )

    layout_eval = theme.get_plotly_layout(height=210)
    fig_eval.update_layout(**layout_eval, hovermode="x unified")
    st.plotly_chart(fig_eval, use_container_width=True)

    st.caption(
        "📌 **Evaluation Type**: One-step-ahead walk-forward testing on chronological 80/20 holdout. "
        "Features use true observed past sales, measuring single-day precision."
    )

with col_fc:
    st.markdown('<div class="panel-white"></div><div id="seven-day-forecast"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sh">
            <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                <span class="st">Seven-Day Sales Forecast</span>
                <span class="badge">FORECAST</span>
            </div>
            <div class="legend">
                <span><i></i>Recent</span>
                <span><i class="fc"></i>Forecast</span>
            </div>
        </div>
        <div class="cap">Recursive 7-day forecast; not independently validated.</div>
        """,
        unsafe_allow_html=True,
    )

    # Build history for recursive rollout
    last_date = df_raw["Date"].max()
    recent_sales = df_raw["Sales"].values[-7:].tolist()

    forecast_rows = []
    for day in range(1, 8):
        forecast_date = last_date + timedelta(days=day)

        sales_lag_1 = recent_sales[-1]
        sales_lag_7 = recent_sales[-7]
        rolling_mean_7 = np.mean(recent_sales[-7:])
        weekday = forecast_date.weekday()
        month = forecast_date.month
        day_of_year = forecast_date.timetuple().tm_yday

        features = pd.DataFrame(
            [[sales_lag_1, sales_lag_7, rolling_mean_7, weekday, month, day_of_year]],
            columns=FEATURE_COLS,
        )

        pred = float(model.predict(features)[0])
        pred = max(pred, 0)  # physical non-negativity constraint

        forecast_rows.append(
            {
                "Date": forecast_date.strftime("%Y-%m-%d"),
                "Forecasted Sales": round(pred, 1),
                "Weekday": forecast_date.strftime("%A"),
            }
        )
        recent_sales.append(pred)

    df_forecast = pd.DataFrame(forecast_rows)

    fig_fc = go.Figure()
    tail_actual = df_raw.tail(14)[["Date", "Sales"]]
    fig_fc.add_trace(
        go.Scatter(
            x=tail_actual["Date"],
            y=tail_actual["Sales"],
            mode="lines+markers",
            name="Actual (recent)",
            line=dict(color=theme.COLOR_ACTUAL, width=1.8),
            marker=dict(size=4, color=theme.COLOR_ACTUAL),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Actual: %{y:,}<extra></extra>",
        )
    )
    fig_fc.add_trace(
        go.Scatter(
            x=df_forecast["Date"],
            y=df_forecast["Forecasted Sales"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color=theme.COLOR_FORECAST, width=1.8, dash="dash"),
            marker=dict(size=5, symbol="circle", color=theme.COLOR_FORECAST),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Forecast: %{y:.1f}<extra></extra>",
        )
    )

    fig_fc.add_vline(
        x=last_date.timestamp() * 1000,
        line_dash="dot",
        line_color="rgba(115, 115, 115, 0.4)",
        annotation_text="Forecast begins",
        annotation_position="top left",
        annotation_font_size=11,
        annotation_font_color="#737373",
    )

    layout_fc = theme.get_plotly_layout(height=280)
    layout_fc["margin"] = dict(l=40, r=15, t=28, b=30)
    fig_fc.update_layout(**layout_fc, hovermode="x unified")
    st.plotly_chart(fig_fc, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────
# Row 3: Forecast Values & CSV Export (Unified Glass Card)
# ──────────────────────────────────────────────────────────────────────
col_fc_full = st.columns(1)[0]
with col_fc_full:
    st.markdown('<div class="panel-glass"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sh">
            <div>
                <div class="st">Forecast Values</div>
                <div class="cap" style="margin-bottom: 0;">Predicted daily sales, next 7 days</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    theme.render_forecast_cards(forecast_rows)

    col_tbl, col_dl = st.columns([3, 1], gap="medium")
    with col_tbl:
        st.dataframe(
            df_forecast.set_index("Date").style.format({"Forecasted Sales": "{:.1f}"}),
            use_container_width=True,
            height=215,
        )

    with col_dl:
        st.markdown(
            """
            <div style="padding: 4px 0;">
                <div style="font-size: 13.5px; font-weight: 600; color: #1A1A1A; margin-bottom: 4px;">
                    Export Predictions
                </div>
                <div style="font-size: 12px; color: #737373; margin-bottom: 12px; line-height: 1.4;">
                    Download the complete 7-day predicted values in standard CSV format for downstream planning.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        csv_bytes = df_forecast.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Forecast CSV",
            data=csv_bytes,
            file_name="sales_forecast_7day.csv",
            mime="text/csv",
        )

# ──────────────────────────────────────────────────────────────────────
# Row 4: Dataset Explorer
# ──────────────────────────────────────────────────────────────────────
st.markdown('<div id="dataset-explorer"></div>', unsafe_allow_html=True)

with st.expander("Dataset Explorer — Filtered Records", expanded=False):
    st.caption(f"Showing **{len(df_filtered):,}** records from {start_date} to {end_date}.")
    st.dataframe(
        df_filtered,
        use_container_width=True,
        height=380,
    )
