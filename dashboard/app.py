import numpy as np
import streamlit as st
import requests
from datetime import date

# Dashboard configuration
st.set_page_config(
    page_title="Enterprise Demand Forecasting",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Enterprise Demand Forecasting Platform")

st.info(
    "This platform uses the M5 Walmart historical sales dataset "
    "(2011–2016) to demonstrate demand forecasting, historical "
    "validation, and model performance evaluation."
)

st.write(
    "Machine Learning-powered retail demand forecasting "
    "with FastAPI historical predictions and seven-day forecasts."
)

st.divider()

# FastAPI backend
API_URL = "http://127.0.0.1:8000"
selected_product = None

st.subheader("📦 Select Product")

try:
    # Fetch products from FastAPI
    response = requests.get(
        f"{API_URL}/products",
        timeout=10
    )

    response.raise_for_status()
    products = response.json()["products"]

    # Product selection
    selected_product = st.selectbox(
        "Choose a product",
        products
    )

    st.info(f"Selected Product: {selected_product}")

    # Date selection
    selected_date = st.date_input(
        "Select  Historical Prediction Date",
        value=date(2015, 7, 17),
        min_value=date(2011, 2, 26),
        max_value=date(2016, 4, 24)
    )

    # Predict button
    if st.button("🔮 Predict Sales", type="primary"):

        request_data = {
            "item_id": selected_product,
            "date": selected_date.isoformat()
        }

        try:
            prediction_response = requests.post(
                f"{API_URL}/predict-historical",
                json=request_data,
                timeout=30
            )

            prediction_response.raise_for_status()
            result = prediction_response.json()

            st.success("Prediction generated successfully!")

            st.metric(
                label="Predicted Daily Sales",
                value=f"{result['predicted_sales']:.2f} items"
            )

            st.write("**Product:**", result["item_id"])
            st.write("**Date:**", result["date"])
            st.write("**Model:**", result["model"])

        except requests.exceptions.HTTPError as error:
            st.error(f"Prediction request failed: {error}")

        except requests.exceptions.RequestException:
            st.error("Cannot connect to the prediction API.")

except requests.exceptions.RequestException:
    st.error(
        "Unable to connect to FastAPI. "
        "Please make sure your backend is running."
    )
from pathlib import Path
import pandas as pd

st.divider()
st.subheader("📈 Actual vs Predicted Sales")

# Locate saved test predictions
PROJECT_DIR = Path(__file__).resolve().parent.parent

RESULTS_PATH = (
    PROJECT_DIR / "data" / "processed" / "test_predictions.csv"
)

# Load test predictions
results_df = pd.read_csv(
    RESULTS_PATH,
    parse_dates=["date"]
)

# Display chart for selected product
if selected_product is not None:

    product_results = results_df[
        results_df["item_id"] == selected_product
    ].copy()

    product_results = product_results.sort_values("date")

    # Prepare chart data
    chart_data = product_results.set_index("date")[
        ["sales", "predicted_sales"]
    ]

    chart_data.columns = [
        "Actual Sales",
        "Predicted Sales"
    ]

    st.line_chart(chart_data)

        # Calculate product-level performance

    actual = product_results["sales"]
    predicted = product_results["predicted_sales"]

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    st.subheader("📊 Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Mean Absolute Error",
            f"{mae:.2f} items"
        )

    with col2:
        st.metric(
            "Root Mean Squared Error",
            f"{rmse:.2f} items"
        )

    with col3:
        st.metric(
            "Test Records",
            len(product_results)
        )

    st.caption(
        "Historical test-period predictions. "
        "Each prediction uses observed prior sales."
    )
st.divider()
st.subheader("🏆 Product Performance Comparison")

# Calculate MAE for each product
product_performance = (
    results_df.groupby("item_id")["absolute_error"]
    .mean()
    .sort_values()
    .reset_index()
)

product_performance.columns = [
    "Product",
    "MAE"
]

# Display product comparison chart
st.bar_chart(
    product_performance,
    x="Product",
    y="MAE"
)

st.caption(
    "Lower MAE indicates better prediction accuracy. "
    "Results are based on the historical test dataset."
)
st.divider()
st.subheader("📦 Retail Demand Analytics")

# Calculate sales analytics for the selected product
if selected_product is not None:

    selected_sales = results_df[
        results_df["item_id"] == selected_product
    ].copy()

    total_sales = selected_sales["sales"].sum()
    avg_daily_sales = selected_sales["sales"].mean()
    peak_daily_sales = selected_sales["sales"].max()

    # Display business KPI cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Units Sold",
            f"{total_sales:,.0f}"
        )

    with col2:
        st.metric(
            "Average Daily Sales",
            f"{avg_daily_sales:.2f}"
        )

    with col3:
        st.metric(
            "Peak Daily Sales",
            f"{peak_daily_sales:,.0f}"
        )

    st.caption(
        "Sales statistics are calculated from the "
        "historical test period (July 2015 – April 2016)."
    )
st.divider()
st.subheader("📅 Monthly Demand Trends")

if selected_product is not None:

    monthly_data = results_df[
        results_df["item_id"] == selected_product
    ].copy()

    # Group daily sales by month
    monthly_data["month"] = (
        monthly_data["date"].dt.to_period("M").dt.to_timestamp()
    )

    monthly_sales = (
        monthly_data.groupby("month", as_index=False)["sales"]
        .sum()
    )

    monthly_sales = monthly_sales.set_index("month")

    # Display monthly sales chart
    st.bar_chart(
        monthly_sales,
        y="sales"
    )

    st.caption(
        "Monthly units sold for the selected product. "
        "July 2015 and April 2016 are partial months."
    )
# -----------------------------------
# 7-Day Demand Forecast (Historical Dataset)
# -----------------------------------

st.subheader("🔮 7-Day Demand Forecast (Historical Dataset)")

# Load saved future predictions
future_forecast_path = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "future_forecasts.csv"
)

future_df = pd.read_csv(
    future_forecast_path,
    parse_dates=["date"]
)

# Filter predictions for selected product
selected_future = future_df[
    future_df["item_id"] == selected_product
].copy()

selected_future = selected_future.sort_values("date")

# Display total forecasted demand
if selected_product is None:
    st.caption("Connect to the prediction API to select a product for the forecast chart.")
else:
    total_future_demand = selected_future["predicted_sales"].sum()

    st.metric("Next 7 Days Predicted Demand", f"{total_future_demand:,.0f} units")
    st.line_chart(selected_future.set_index("date")[["predicted_sales"]])
    first, last = selected_future["date"].min(), selected_future["date"].max()
    st.caption(f"Forecast dates {first:%d %b %Y} through {last:%d %b %Y}.")
# ----------------------------------------
# MODEL PERFORMANCE & BACKTESTING
# ----------------------------------------

st.divider()
st.header("Model Performance & Backtesting")

# Load saved rolling-backtest results
backtest_file = (
    PROJECT_DIR / "data" / "processed"
    / "rolling_backtest_results.csv"
)

backtest_df = pd.read_csv(backtest_file)

# Calculate model performance
model_mae = (
    backtest_df["sales"]
    - backtest_df["predicted_sales"]
).abs().mean()

baseline_mae = (
    backtest_df["sales"]
    - backtest_df["baseline_sales"]
).abs().mean()

improvement = 100 * (1 - model_mae / baseline_mae) if baseline_mae else np.nan
sales_volume = backtest_df["sales"].abs().sum()
accuracy = 100 * (1 - backtest_df["absolute_error"].sum() / sales_volume) if sales_volume else np.nan

# Display KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Forecast MAE", f"{model_mae:.2f}")
col2.metric("Seasonal Naive MAE", f"{baseline_mae:.2f}")
col3.metric("MAE Improvement", f"{improvement:.2f}%" if np.isfinite(improvement) else "N/A")
col4.metric("Forecast Accuracy", f"{accuracy:.2f}%" if np.isfinite(accuracy) else "N/A")

st.caption(
    f"Seven-day forecasts across {backtest_df['window_start'].nunique()} held-out windows "
    f"and {backtest_df['item_id'].nunique()} products ({len(backtest_df):,} predictions). "
    "Accuracy is 100% minus WAPE. The backtest uses recorded calendar and selling prices."
)
# ----------------------------------------
# FORECAST ACCURACY BY HORIZON
# ----------------------------------------

st.subheader("Forecast Accuracy by Horizon")

horizon_file = (
    PROJECT_DIR / "data" / "processed"
    / "forecast_horizon_metrics.csv"
)

horizon_df = pd.read_csv(horizon_file).rename(columns={"RF_MAE": "Model_MAE"})

# Create chart-friendly labels
horizon_df["Forecast Day"] = (
    "Day " + horizon_df["horizon"].astype(str)
)

# Compare model and baseline
chart_data = horizon_df.set_index("Forecast Day")[
    ["Model_MAE", "Baseline_MAE"]
]

chart_data = chart_data.rename(columns={
    "Model_MAE": "Forecast",
    "Baseline_MAE": "Seasonal Naive"
})

st.line_chart(chart_data)

st.caption(
    "Lower MAE indicates better forecasting accuracy. "
    f"Predictions per horizon range from {horizon_df['Predictions'].min()} to {horizon_df['Predictions'].max()}."
)
# ----------------------------------------
# PRODUCT-LEVEL FORECAST PERFORMANCE
# ----------------------------------------

st.subheader("Product-Level Forecast Performance")

# Load product-level evaluation metrics
product_metrics_file = (
    PROJECT_DIR / "data" / "processed"
    / "product_forecast_metrics.csv"
)

product_metrics_df = pd.read_csv(
    product_metrics_file
)

# Rename columns for readability
product_metrics_df = product_metrics_df.rename(
    columns={
        "item_id": "Product",
        "MAE": "Mean Absolute Error",
        "RMSE": "Root Mean Squared Error"
    }
)

# Display product-level comparison
st.bar_chart(
    product_metrics_df.set_index("Product")[
        ["Mean Absolute Error", "Root Mean Squared Error"]
    ]
)

# Display detailed metrics
st.dataframe(
    product_metrics_df,
    use_container_width=True,
    hide_index=True
)
# ----------------------------------------
# ACTUAL VS PREDICTED: ROLLING BACKTEST
# ----------------------------------------

st.subheader("Actual vs Predicted Demand")

# Select a product
backtest_product = st.selectbox(
    "Select Product for Backtest Analysis",
    sorted(backtest_df["item_id"].unique()),
    key="backtest_product"
)

# ----------------------------------------
# FILTER BY PRODUCT AND FORECAST WINDOW
# ----------------------------------------

# Get available backtesting windows
available_windows = sorted(
    backtest_df["window_start"].unique()
)

# Allow users to select a forecast window
selected_window = st.selectbox(
    "Select Forecast Window",
    available_windows,
    format_func=lambda x: pd.to_datetime(x).strftime(
        "%d %B %Y"
    ),
    key="selected_backtest_window"
)

# Filter by selected product and window
product_backtest = backtest_df[
    (backtest_df["item_id"] == backtest_product)
    & (backtest_df["window_start"] == selected_window)
].copy()
# Convert dates
product_backtest["date"] = pd.to_datetime(
    product_backtest["date"]
)

# Sort chronologically
product_backtest = product_backtest.sort_values("date")

# Prepare comparison chart
comparison_chart = product_backtest.set_index("date")[
    ["sales", "predicted_sales", "baseline_sales"]
]

comparison_chart = comparison_chart.rename(columns={
    "sales": "Actual Demand",
    "predicted_sales": "Model Forecast",
    "baseline_sales": "Seasonal Naive Forecast"
})

# Display daily dates instead of time labels
comparison_chart.index = (
    comparison_chart.index.strftime("%d %b")
)

comparison_chart.index.name = "Forecast Date"

st.line_chart(
    comparison_chart,
    x_label="Forecast Date",
    y_label="Units Sold"
)
# ----------------------------------------
# SELECTED WINDOW PERFORMANCE
# ----------------------------------------

# Calculate MAE for the selected product and week
selected_model_mae = (
    product_backtest["sales"]
    - product_backtest["predicted_sales"]
).abs().mean()

selected_baseline_mae = (
    product_backtest["sales"]
    - product_backtest["baseline_sales"]
).abs().mean()

# Calculate improvement percentage
selected_improvement = (
    (selected_baseline_mae - selected_model_mae)
    / selected_baseline_mae
) * 100 if selected_baseline_mae > 0 else np.nan

# Display metrics
metric1, metric2, metric3 = st.columns(3)

metric1.metric(
    "Forecast MAE",
    f"{selected_model_mae:.2f}"
)

metric2.metric(
    "Baseline MAE",
    f"{selected_baseline_mae:.2f}"
)

metric3.metric(
    "MAE Improvement",
    f"{selected_improvement:.2f}%" if np.isfinite(selected_improvement) else "N/A"
)
