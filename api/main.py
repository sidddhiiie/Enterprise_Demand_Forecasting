from fastapi import FastAPI

# Initialize FastAPI application
app = FastAPI(
    title="Enterprise Demand Forecasting API",
    description="Machine Learning API for Retail Demand Forecasting",
    version="1.0.0"
)

# Create our first API endpoint
@app.get("/")
def home():
    return {
        "message": "Enterprise Demand Forecasting API is running",
        "status": "success"
    }
import joblib
from pathlib import Path

# Define project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Define saved model paths
MODEL_PATH = PROJECT_DIR / "models" / "demand_forecasting_rf_v2.joblib"
FEATURE_PATH = PROJECT_DIR / "models" / "feature_columns_v2.joblib"

# Load trained model and feature names
model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)

# Endpoint to verify model loading
@app.get("/model-info")
def model_info():
    return {
        "model": "Random Forest V2",
        "status": "loaded",
        "feature_count": len(feature_columns),
        "features": feature_columns
    }
from pydantic import BaseModel
import pandas as pd

# Define the prediction request format
class PredictionInput(BaseModel):
    features: dict[str, float]


@app.post("/predict")
def predict_sales(data: PredictionInput):

    # Check for missing or unexpected features
    received = set(data.features)
    expected = set(feature_columns)

    missing = expected - received
    extra = received - expected

    if missing or extra:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=422,
            detail={
                "missing_features": sorted(missing),
                "unexpected_features": sorted(extra)
            }
        )

    # Arrange features in the same order as training
    input_df = pd.DataFrame(
        [[data.features[col] for col in feature_columns]],
        columns=feature_columns
    )

    # Generate prediction
    prediction = model.predict(input_df)[0]

    return {
        "predicted_sales": round(max(0, float(prediction)), 2),
        "unit": "items",
        "model": "Random Forest V2"
    }
from fastapi import HTTPException

# Load processed historical data
DATA_PATH = (
    PROJECT_DIR / "data" / "processed" / "forecasting_features.csv"
)

historical_data = pd.read_csv(DATA_PATH)

# Define a simpler request format
class HistoricalPredictionInput(BaseModel):
    item_id: str
    date: str


@app.post("/predict-historical")
def predict_historical(data: HistoricalPredictionInput):

    # Find the requested product and date
    matching_rows = historical_data[
        (historical_data["item_id"] == data.item_id)
        & (historical_data["date"] == data.date)
    ]

    if matching_rows.empty:
        raise HTTPException(
            status_code=404,
            detail="No historical record found for this product and date"
        )

    # Retrieve prepared features
    row = matching_rows.iloc[0]

    input_features = {
        col: float(row[col])
        for col in feature_columns
        if col in historical_data.columns
    }

    # Create product identity features
    for col in feature_columns:
        if col.startswith("product_"):
            product_id = col.replace("product_", "", 1)
            input_features[col] = float(data.item_id == product_id)

    # Arrange all 23 features
    input_df = pd.DataFrame(
        [[input_features[col] for col in feature_columns]],
        columns=feature_columns
    )

    # Generate prediction
    prediction = model.predict(input_df)[0]

    return {
        "item_id": data.item_id,
        "date": data.date,
        "predicted_sales": round(max(0, float(prediction)), 2),
        "unit": "items",
        "model": "Random Forest V2"
    }
@app.get("/products")
def get_products():

    products = sorted(
        historical_data["item_id"].unique().tolist()
    )

    return {
        "total_products": len(products),
        "products": products
    }