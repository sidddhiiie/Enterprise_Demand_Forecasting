"""Train and backtest direct seven-day forecasts for the existing CA_1 products."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
TEST_START = pd.Timestamp("2015-07-17")
HORIZON = 7
KEYS = ["item_id", "date"]
NON_FEATURES = [*KEYS, "sales", "window_start"]


RF_FEATURES = [
    "day_of_week", "day_of_month", "week_of_year", "month", "year", "is_weekend",
    "lag_1", "lag_7", "lag_28", "rolling_mean_7", "rolling_mean_28",
    "rolling_std_7", "rolling_std_28", "was_zero_yesterday", "is_event", "snap_CA",
    "sell_price", "price_available",
]


def build_features(history, future=None):
    """For a target h days after the origin, use sales at lags h or greater."""
    if future is not None:
        history = pd.concat([history, future], ignore_index=True)
    rows = []
    for product, group in history.groupby("item_id", sort=True):
        daily = group.set_index("date").sort_index().asfreq("D")
        sales = daily["sales"]
        for horizon in range(1, HORIZON + 1):
            recent = sales.shift(horizon)
            frame = daily[["sales", "snap_CA", "is_event", "sell_price"]].copy()
            frame["item_id"] = product
            frame["day_of_week"] = daily.index.day_of_week
            frame["month"] = daily.index.month
            frame["horizon"] = horizon
            frame["window_start"] = daily.index - pd.Timedelta(days=horizon - 1)
            frame["baseline_sales"] = sales.shift(7)
            for length in (7, 14, 28, 56, 84):
                frame[f"mean_{length}"] = recent.rolling(length).mean()
            for lag in (14, 21, 28, 35, 42, 49, 56):
                frame[f"lag_{lag}"] = sales.shift(lag)
            weekdays = [sales.shift(lag) for lag in (7, 14, 21, 28, 35, 42)]
            frame["weekday_median"] = np.median(weekdays, axis=0)
            frame["seasonal_level"] = (
                frame["mean_14"] * frame["weekday_median"]
                / frame["mean_56"].clip(lower=1)
            )
            frame["relative_price"] = (
                frame["sell_price"] / daily["sell_price"].shift(horizon).rolling(56).median()
            )
            rows.append(frame.reset_index())
    features = pd.concat(rows, ignore_index=True)
    for product in sorted(history["item_id"].unique()):
        features[f"product_{product}"] = (features["item_id"] == product).astype(int)
    return features


def fit_model(rows, cutoff):
    train = rows.loc[(rows["date"] < cutoff) & rows["sales"].notna()]
    columns = train.columns.difference(NON_FEATURES)
    weights = 0.08 + np.exp(-(cutoff - train["date"]).dt.days / 365)
    model = HistGradientBoostingRegressor(
        loss="absolute_error", max_depth=3, max_iter=160,
        learning_rate=0.055, min_samples_leaf=45, early_stopping=False,
        random_state=42,
    )
    model.fit(train[columns], train["sales"], sample_weight=weights)
    return model, columns


def predict(model, columns, rows):
    return model.predict(rows[columns]).clip(min=0)


def fit_recursive(history, cutoff):
    train = history.loc[history["date"] < cutoff]
    columns = [*RF_FEATURES, *[
        f"product_{item}" for item in sorted(history["item_id"].unique())
    ]]
    x = train.assign(**{
        column: (train["item_id"] == column.removeprefix("product_")).astype(int)
        for column in columns if column.startswith("product_")
    })
    model = RandomForestRegressor(
        n_estimators=100, max_depth=15, min_samples_leaf=3, random_state=42, n_jobs=4,
    )
    model.fit(x[columns], train["sales"])
    return model, columns


def predict_recursive(model, columns, targets, history):
    daily = history.pivot(index="date", columns="item_id", values="sales")
    first = targets.loc[targets["horizon"] == 1, ["window_start", "item_id"]].reset_index(drop=True)
    dates = first["window_start"].to_numpy()
    products = daily.columns.get_indexer(first["item_id"])
    prior = np.column_stack([
        daily.reindex(dates - np.timedelta64(lag, "D")).to_numpy()[np.arange(len(first)), products]
        for lag in range(28, 0, -1)
    ])
    inputs = targets.set_index(["window_start", "item_id", "horizon"])
    forecasts = []
    for h in range(1, HORIZON + 1):
        keys = pd.MultiIndex.from_frame(first.assign(horizon=h))
        rows = inputs.loc[keys].reset_index()
        date = rows["date"].dt
        x = rows.assign(
            day_of_month=date.day, week_of_year=date.isocalendar().week,
            year=date.year, is_weekend=rows["day_of_week"] >= 5,
            lag_1=prior[:, -1], lag_7=prior[:, -7], lag_28=prior[:, -28],
            rolling_mean_7=prior[:, -7:].mean(axis=1), rolling_mean_28=prior.mean(axis=1),
            rolling_std_7=prior[:, -7:].std(axis=1, ddof=1),
            rolling_std_28=prior.std(axis=1, ddof=1),
            was_zero_yesterday=prior[:, -1] == 0, price_available=rows["sell_price"].notna(),
        )
        for name in columns:
            if name.startswith("product_"):
                x[name] = rows["item_id"] == name.removeprefix("product_")
        prediction = predict(model, columns, x)
        forecasts.append(pd.Series(prediction, index=keys))
        prior = np.column_stack((prior[:, 1:], prediction))
    return pd.concat(forecasts).reindex(inputs.index).to_numpy()


def score(rows):
    total_sales = rows["sales"].abs().sum()
    wape = rows["absolute_error"].sum() / total_sales if total_sales else np.nan
    return pd.Series({
        "MAE": rows["absolute_error"].mean(),
        "RMSE": np.sqrt(rows["squared_error"].mean()),
        "WAPE": 100 * wape,
        "Forecast accuracy": 100 * (1 - wape),
    })


def load_future_inputs(project_dir, history, start):
    raw = project_dir / "data" / "raw"
    dates = pd.date_range(start, periods=HORIZON)
    calendar = pd.read_csv(raw / "calendar.csv", parse_dates=["date"])
    calendar = calendar.loc[
        calendar["date"].isin(dates),
        ["date", "wm_yr_wk", "event_name_1", "snap_CA"],
    ].copy()
    if len(calendar) != HORIZON:
        raise ValueError("Calendar is missing dates for the seven-day forecast.")
    calendar["is_event"] = calendar["event_name_1"].notna()
    products = history[["item_id", "store_id"]].drop_duplicates()
    future = products.merge(calendar, how="cross")
    keys = ["store_id", "item_id", "wm_yr_wk"]
    price_rows = []
    for chunk in pd.read_csv(raw / "sell_prices.csv", chunksize=250_000):
        price_rows.append(chunk.merge(future[keys].drop_duplicates(), on=keys))
    prices = pd.concat(price_rows, ignore_index=True)
    future = future.merge(prices, on=keys, how="left", validate="many_to_one")
    last_prices = history.sort_values("date").groupby("item_id")["sell_price"].last()
    future["sell_price"] = future["sell_price"].fillna(future["item_id"].map(last_prices))
    return future


def backtest_rows(rows, start, end, products):
    starts = pd.date_range(start, end - pd.Timedelta(days=HORIZON - 1), freq="7D")
    results = rows.loc[rows["window_start"].isin(starts) & rows["sales"].notna()].copy()
    expected = len(starts) * HORIZON * products
    if expected == 0 or len(results) != expected or results["baseline_sales"].isna().any():
        raise ValueError("Incomplete product history for the backtest.")
    return results


@threadpool_limits.wrap(limits=4, user_api="openmp")
def run_forecast(project_dir=ROOT, *, write=False):
    processed = project_dir / "data" / "processed"
    history = pd.read_csv(processed / "forecasting_features.csv", parse_dates=["date"])
    rows = build_features(history)
    latest = history["date"].max()
    products = history["item_id"].nunique()
    dates = sorted(history["date"].unique())
    validation_start = dates[int(len(dates) * 0.70)]
    if validation_start >= TEST_START:
        raise ValueError("The validation split must precede the held-out test period.")

    # The saved API model was trained on the first 70% of dates.
    legacy_model = joblib.load(project_dir / "models" / "demand_forecasting_rf_v2.joblib")
    legacy_columns = joblib.load(project_dir / "models" / "feature_columns_v2.joblib")
    validation_model, columns = fit_model(rows, validation_start)
    validation = backtest_rows(rows, validation_start, TEST_START - pd.Timedelta(days=1), products)
    validation_direct = predict(validation_model, columns, validation)
    validation_legacy = predict_recursive(legacy_model, legacy_columns, validation, history)
    weight = min(
        (0, 0.25, 0.5, 0.75, 1),
        key=lambda w: np.abs(
            validation["sales"] - (w * validation_direct + (1 - w) * validation_legacy)
        ).mean(),
    )

    model, columns = fit_model(rows, TEST_START)
    results = backtest_rows(rows, TEST_START, latest, products)
    results["legacy_sales"] = predict_recursive(legacy_model, legacy_columns, results, history)
    results["predicted_sales"] = (
        weight * predict(model, columns, results) + (1 - weight) * results["legacy_sales"]
    )
    error = results["sales"] - results["predicted_sales"]
    results["absolute_error"] = error.abs()
    results["squared_error"] = error.pow(2)
    results["baseline_absolute_error"] = (results["sales"] - results["baseline_sales"]).abs()
    results = results[[*KEYS, "predicted_sales", "window_start", "sales", "absolute_error",
                       "squared_error", "baseline_sales", "horizon", "baseline_absolute_error",
                       "legacy_sales"]]
    results = results.sort_values(["window_start", *KEYS]).reset_index(drop=True)
    results.attrs["direct_weight"] = weight

    future_start = latest + pd.Timedelta(days=1)
    future_inputs = load_future_inputs(project_dir, history, future_start)
    future_rows = build_features(history, future_inputs)
    future_rows = future_rows.loc[future_rows["window_start"] == future_start]
    if len(future_rows) != HORIZON * products:
        raise ValueError("Incomplete product inputs for the future forecasts.")
    future = future_rows[KEYS].copy()
    model, columns = fit_model(rows, future_start)
    recursive_model, recursive_columns = fit_recursive(history, future_start)
    future["predicted_sales"] = (
        weight * predict(model, columns, future_rows)
        + (1 - weight) * predict_recursive(recursive_model, recursive_columns, future_rows, history)
    )
    future = future.sort_values(KEYS).reset_index(drop=True)

    if write:
        by_horizon = results.groupby("horizon").agg(
            Model_MAE=("absolute_error", "mean"),
            Baseline_MAE=("baseline_absolute_error", "mean"),
            Predictions=("sales", "count"),
        )
        by_horizon["Improvement_%"] = 100 * (
            1 - by_horizon["Model_MAE"] / by_horizon["Baseline_MAE"].replace(0, np.nan)
        )
        by_product = results.groupby("item_id").agg(
            MAE=("absolute_error", "mean"), RMSE=("squared_error", "mean"),
        )
        by_product["RMSE"] = np.sqrt(by_product["RMSE"])
        results.to_csv(processed / "rolling_backtest_results.csv", index=False)
        by_horizon.to_csv(processed / "forecast_horizon_metrics.csv")
        by_product.to_csv(processed / "product_forecast_metrics.csv")
        future.to_csv(processed / "future_forecasts.csv", index=False)
        pd.DataFrame({"metric": ["forecast_accuracy"], "value": [
            round(score(results)["Forecast accuracy"], 2)
        ]}).to_csv(processed / "model_accuracy.csv", index=False)
    return results, future


if __name__ == "__main__":
    backtest, forecast = run_forecast(write=True)
    print(f"Direct-model blend weight {backtest.attrs['direct_weight']:.0%}")
    for label, frame in (
        ("All complete weeks", backtest),
        ("Original 10 windows", backtest.loc[
            ((backtest["window_start"] - TEST_START).dt.days % 28) == 0
        ]),
    ):
        comparison = {}
        for name, column in (
            ("Ensemble", "predicted_sales"), ("Random Forest V2", "legacy_sales"),
            ("Seasonal naive", "baseline_sales"),
        ):
            error = frame["sales"] - frame[column]
            comparison[name] = score(frame.assign(absolute_error=error.abs(), squared_error=error.pow(2)))
        print(label)
        print(pd.DataFrame(comparison).round(2).to_string())
    print(f"Backtest rows {len(backtest)}, future forecasts {len(forecast)}")
