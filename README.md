# 📊 AI-Based Sales Forecasting and Business Intelligence Dashboard for Small Businesses

> **Third-Year B.Tech Mini-Project in Artificial Intelligence & Machine Learning**  
> **Universal Skilltech University**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Project Overview

This project provides an interactive, machine-learning-powered business intelligence (BI) dashboard tailored for small-to-medium retail and e-commerce enterprises. By leveraging historical daily sales records and an autoregressive time-series **Linear Regression** model, the dashboard delivers real-time key performance indicators (KPIs), historical trend exploration, rigorous model evaluation, and recursive multi-step 7-day sales forecasts.

Designed with a modern **Light Glass / iDraft** aesthetic, the dashboard presents financial analytics with high visual clarity, tabular numerals, translucent glassmorphism cards, and responsive layouts.

---

## 🎯 Problem Statement & Objectives

### Problem Statement
Small business owners frequently struggle with inventory misallocation, stockouts, and unpredictable cash flow due to reliance on static spreadsheets or intuition. Enterprise ERP/forecasting solutions are often excessively complex, opaque, and cost-prohibitive.

### Project Objectives
1. **Accurate Short-Term Forecasting**: Predict daily sales up to 7 days into the future using autoregressive momentum and calendar seasonality.
2. **Transparent Business Intelligence**: Present core executive KPIs—total volume, daily averages, profit metrics, and inventory levels—in a unified dashboard.
3. **Faculty & Examiner Auditability**: Clearly distinguish one-step-ahead historical evaluation from multi-step recursive forecasting to prevent time-series data leakage.
4. **Accessible Deployment**: Enable zero-friction deployment to Streamlit Community Cloud for stakeholder and academic demonstrations.

---

## ✨ Key Features

- **Executive KPI Cards**: Real-time totals and averages for Sales Volume, Daily Sales, Recorded Profit, and Inventory.
- **Dynamic Date Filtering**: Sidebar date-range controls that update KPIs, moving averages, and data views dynamically.
- **Historical Trend Visualizer**: Interactive Plotly line charts displaying daily sales alongside an optional 30-day moving average trend line.
- **Leak-Free Model Evaluation**: Chronological 80/20 train/test holdout evaluation displaying Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) badges.
- **7-Day Recursive Forecast**: Day-by-day predictive rollout that iteratively feeds forecasted values into subsequent lag features, complemented by daily prediction tiles.
- **Actionable Data Export**: Instant CSV download of 7-day forecasted figures (`sales_forecast_7day.csv`) for operational planning.
- **Filtered Dataset Explorer**: Expandable, sortable view of historical records matching the active filter criteria.

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.9–3.12 | Core programming environment |
| **Frontend Framework** | Streamlit (>= 1.28.0) | Reactive web UI and dashboard routing |
| **Data Manipulation** | Pandas (>= 1.5.0), NumPy (>= 1.23.0) | Feature engineering, rolling windows, date operations |
| **Machine Learning** | Scikit-learn (>= 1.2.0), Joblib (>= 1.2.0) | Pre-trained Linear Regression model inference & metrics |
| **Data Visualization** | Plotly (>= 5.15.0) | Interactive, responsive time-series charts |
| **Design System** | Vanilla CSS (`styles.css`), DM Sans | Glassmorphism, ambient gradients, tabular numerals |

---

## 🗂️ Project Directory Structure

```text
Sales_Forecasting_Project/
├── .streamlit/
│   └── config.toml          # Streamlit light theme and server configuration
├── app.py                   # Main Streamlit application and layout
├── theme.py                 # Modular presentation helpers and Plotly styling
├── styles.css               # Light Glass / iDraft CSS design tokens
├── sales_data.csv           # Historical e-commerce sales dataset (1,784 records)
├── sales_model.pkl          # Pre-trained Scikit-Learn Linear Regression model artifact
├── requirements.txt         # Minimal, production-tested Python dependencies
├── .gitignore               # Exclusions for virtualenvs, caches, and secrets
└── README.md                # Project documentation and reproduction guide
```

---

## 📊 Dataset Description & Source

- **Original Data Source**: [John Mars E-commerce Sales Dataset](https://raw.githubusercontent.com/johnmars-prog/E-commerce-Sales-Data/main/ecommerce_sales_data.csv)
- **Local File**: `sales_data.csv`
- **Temporal Coverage**: January 1, 2013 to November 19, 2017 (1,784 continuous daily observations)
- **Key Fields**:
  - `Date`: Timestamp formatted as `YYYY-MM-DD`
  - `Sales`: Daily units sold (target variable)
  - `Profit`: Daily net profit generated
  - `inventory`: Daily warehouse inventory count
  - `Clicks`, `Views`, `Click-Through Rate`: Digital marketing engagement metrics

> **Academic Disclosure**: The real-world business provenance of this open dataset has not been independently verified. It is utilized here as an academic benchmarking and demonstration resource.

---

## ⚙️ Machine Learning Methodology

### 1. Feature Engineering
Time-series dependencies and seasonal patterns are engineered through 6 autoregressive and calendar features:

| # | Feature Name | Computation | Intuition Captured |
|---| :--- | :--- | :--- |
| 1 | `Sales_Lag_1` | $y_{t-1}$ (`Sales.shift(1)`) | Immediate previous-day momentum |
| 2 | `Sales_Lag_7` | $y_{t-7}$ (`Sales.shift(7)`) | Same-day weekly recurring seasonality |
| 3 | `Rolling_Mean_7` | $\frac{1}{7} \sum_{i=1}^{7} y_{t-i}$ | Short-term smoothed moving average |
| 4 | `Weekday` | `Date.dt.weekday` (0=Mon … 6=Sun) | Intra-week demand cyclicality |
| 5 | `Month` | `Date.dt.month` (1–12) | Macro monthly and quarterly patterns |
| 6 | `Day_Of_Year` | `Date.dt.dayofyear` (1–366) | Annual seasonality and holiday progression |

*Note: Initial rows with NaN values resulting from the 7-day lag/rolling window (7 rows) are dropped, leaving 1,777 feature-engineered observations.*

### 2. Model Architecture
- **Algorithm**: Ordinary Least Squares Linear Regression (`sklearn.linear_model.LinearRegression`)
- **Trained Artifact**: Serialized dictionary saved via Joblib containing:
  - `model`: Fitted `LinearRegression` estimator
  - `features`: Exact ordered list of 6 input column names

### 3. Chronological Train/Test Split
To prevent temporal data leakage, an **80/20 chronological holdout** is enforced (no random shuffling):
- **Training Set (80%)**: 1,421 records (2013-01-08 to 2016-11-28)
- **Testing Set (20%)**: 356 records (2016-11-29 to 2017-11-19)
- **Condition**: $\max(\text{Train Date}) < \min(\text{Test Date})$ is strictly preserved.

### 4. Evaluation vs. Multi-Step Forecasting
- **One-Step-Ahead Walk-Forward Evaluation**: Evaluated across all 356 test days using true observed historical lags ($y_{t-1}, y_{t-7}$) to establish baseline single-day precision:
  - **Mean Absolute Error (MAE)**: $\approx 15.35$ units
  - **Root Mean Squared Error (RMSE)**: $\approx 19.23$ units
  *(Metrics reflect the academic prototype baseline.)*
- **Recursive 7-Day Forecasting**: For unobserved future dates beyond November 19, 2017, predictions are generated iteratively: day 1’s forecast is appended to the rolling history to compute $y_{t-1}, y_{t-7}$ for day 2, and so forth. A non-negativity constraint ($\max(\hat{y}, 0)$) ensures physical plausibility.

---

## 🚀 Getting Started Locally

### Prerequisites
- Python 3.9, 3.10, 3.11, or 3.12
- Git

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

5. **Access the Dashboard:**
   Open your browser and navigate to `http://localhost:8501`.

---

## ☁️ Deployment on Streamlit Community Cloud

This repository is pre-configured for direct, one-click deployment:

1. Push this repository to your public GitHub account.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with your GitHub credentials.
3. Click **New app**.
4. Select your repository, branch (`main`), and set the **Main file path** to `app.py`.
5. Click **Deploy!**

The `.streamlit/config.toml` file automatically configures the server and light-glass theme.

---

## ⚠️ Current Limitations & Future Improvements

### Current Limitations
- **Linearity Assumption**: Standard Linear Regression does not model non-linear promotional spikes or complex macro-economic shocks.
- **Error Compounding**: Recursive multi-step forecasts inherently accumulate variance as the prediction horizon extends past 7 days.
- **Single Aggregated Series**: Forecasts are computed on total daily volume rather than granular SKU-level hierarchies.

### Future Roadmap
- [ ] Integration of tree-based ensembles (LightGBM / XGBoost) and Prophet for non-linear holiday handling.
- [ ] Automated continuous retraining pipelines via GitHub Actions.
- [ ] Dynamic safety-stock and reorder-point calculations integrated directly with the inventory KPI.
- [ ] SKU-level breakdown with multi-select category filtering.

---

## 👨‍🎓 Academic & Student Information

- **Student Author**: Navneet Verma
- **Degree**: Bachelor of Technology (B.Tech) in Artificial Intelligence & Machine Learning
- **Roll Number**: `UST24R0101031`
- **Project Supervisor**: Professor Kiran Dhanaji Kale
- **Institution**: Universal Skilltech University

---

## 📜 License

This project is licensed under the [MIT License](LICENSE)—free for academic, educational, and research use.
