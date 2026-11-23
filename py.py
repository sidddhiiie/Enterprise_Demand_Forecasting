
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"

prices = pd.read_csv(DATA_DIR / "sell_prices.csv")

print("Sell Prices Dataset Shape:")
print(prices.shape)

print("\nFirst 5 Rows:")
print(prices.head())

print("\nColumn Names:")
print(prices.columns.tolist())

print("\nMissing Values:")
print(prices.isnull().sum())

print("\nBasic Price Statistics:")
print(prices["sell_price"].describe())

