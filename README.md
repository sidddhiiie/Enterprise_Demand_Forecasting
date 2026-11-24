# Enterprise Demand Forecasting & Predictive Analytics Platform

An end-to-end machine learning project designed to forecast retail product demand using historical sales, calendar events, pricing information, and time-series feature engineering.

The platform integrates a **seven-day forecasting ensemble, FastAPI backend, and interactive Streamlit dashboard** to support demand analysis and forecasting.

## Project Overview

Accurate demand forecasting helps businesses make better inventory planning and operational decisions.

This project demonstrates how historical retail sales data can be transformed into actionable demand forecasts through machine learning.

### Key Features

- Exploratory data analysis of retail sales patterns
- Time-series feature engineering with lag and rolling statistics
- Random Forest regression for demand prediction
- Direct and recursive models blended for seven-day forecasting
- REST API developed using FastAPI
- Interactive analytics dashboard built using Streamlit
- Product-level forecasting performance analysis
- Rolling-origin backtesting against a seasonal naive baseline

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, Random Forest |
| Visualization | Matplotlib, Seaborn, Streamlit |
| API Development | FastAPI, Uvicorn |
| Model Persistence | Joblib |
| Development | VS Code, Jupyter Notebook |
| Version Control | Git, GitHub |

## Dataset

**Source:** Kaggle M5 Forecasting dataset

The project uses three source files:

- `sales_train_validation.csv` — Historical daily product sales
- `calendar.csv` — Calendar information, holidays, and events
- `sell_prices.csv` — Historical product selling prices

The forecasting prototype focuses on five selected products from store `CA_1`.

## Machine Learning Approach

The forecasting model uses a Random Forest Regressor with features including:

- Calendar features: weekday, month, year, and weekend indicator
- Lagged demand: 1, 7, and 28 days
- Rolling demand averages: 7 and 28 days
- Rolling standard deviations: 7 and 28 days
- Calendar event indicators
- SNAP indicator
- Selling price information
- Product identification features

The dataset was split chronologically into training, validation, and testing periods.

The FastAPI historical prediction endpoints use the one-day Random Forest.
The seven-day forecaster blends recursive Random Forest predictions with direct,
horizon-aware gradient boosting. The direct model uses sales available before the
forecast origin. The blend weight is chosen on the validation period.

## Model Performance

Run `python -m src.forecast` or run `notebooks/02_future_forecasting.ipynb`
to regenerate the seven-day forecast and dashboard metrics. The command reports
MAE, RMSE, WAPE, and forecast accuracy for the ensemble, original Random Forest,
and seasonal naive baseline on all complete held-out weeks and on the original
ten sampled weeks.

| Held-out evaluation | Ensemble accuracy | Random Forest accuracy | Seasonal naive accuracy |
|---|---:|---:|---:|
| 40 complete weeks, 1,400 predictions | 70.12% | 69.41% | 61.60% |
| Original 10 weeks, 350 predictions | 68.60% | 68.38% | 61.09% |

On the 40-week evaluation, MAE was 11.28 units for the ensemble, 11.54 for the
Random Forest, and 14.49 for the seasonal naive baseline. The ensemble weight
selected on validation was 25% direct gradient boosting and 75% Random Forest.

Forecast accuracy is `100 × (1 − WAPE)`, where WAPE divides total absolute
forecast error by total actual sales. The held-out backtest uses recorded target-date
calendar and selling prices. The future forecast refits both models through the last
observed date before predicting the following seven days.

### XGBoost benchmark

The standalone XGBoost pipeline predicts each of the seven days directly.
The earlier validation period selected a depth-5 model trained on five products
across ten stores. Evaluation uses the same 1,400 predictions for the five CA_1
products as the current ensemble.

| Model | Forecast accuracy | WAPE | MAE (units) | RMSE (units) |
|---|---:|---:|---:|---:|
| XGBoost | 71.18% | 28.82% | 10.88 | 17.12 |
| Current ensemble | 70.12% | 29.88% | 11.28 | 17.56 |

The API and dashboard continue to use the existing forecasts. To reproduce the
XGBoost results, place the M5 source CSVs in `data/raw/` and run these commands
from the repository root:

```bash
python -m pip install xgboost-cpu==3.4.1
python -m src.xgboost_forecast
```

## Interactive Dashboard

The Streamlit dashboard provides:

- Historical sales and prediction analysis
- Product-level demand trends
- Seven-day future demand forecasts
- Forecasting performance KPIs
- Model-versus-baseline comparisons
- Forecast accuracy by prediction horizon
- Product-level MAE and RMSE
- Interactive actual-versus-predicted demand charts
- Product and backtesting-window filters

## Project Structure

```text
Enterprise_Demand_Forecasting/
├── api/
│   └── main.py
├── dashboard/
│   └── app.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── models/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_future_forecasting.ipynb
├── src/
├── tests/
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

Clone the repository and navigate into the project directory.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the M5 dataset and place its source CSV files inside `data/raw/`.

The application also requires the trained model artifacts and processed datasets generated by the project notebooks.

## Running the Application

Start the FastAPI backend:

```bash
python -m uvicorn api.main:app --reload
```

API documentation is available at:

`http://127.0.0.1:8000/docs`

Start the Streamlit dashboard in a separate terminal:

```bash
python -m streamlit run dashboard/app.py
```

Open:

`http://localhost:8501`

## Project Outcomes

- Developed a machine learning workflow for retail demand forecasting.
- Implemented lag-based and rolling-window feature engineering.
- Built recursive seven-day forecasting functionality.
- Evaluated seven-day forecasts on every complete held-out week and retained the original ten-window comparison.
- Integrated forecasting outputs into an interactive Streamlit dashboard and FastAPI backend.

## Future Improvements

- Extend forecasting to more products and stores
- Introduce additional forecasting algorithms
- Add automated data ingestion and model retraining
- Deploy the API and dashboard to a cloud environment
- Incorporate inventory planning and replenishment recommendations

## Author

Siddhi Lokare

B.Tech — Computer Science and Engineering (Artificial Intelligence & Machine Learning)

**Focus:** Data Science, Machine Learning, Data Analytics, and Predictive Analytics
