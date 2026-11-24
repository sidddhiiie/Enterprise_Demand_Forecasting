"""Run a standalone XGBoost benchmark for the seven-day demand forecast.

Install with: python -m pip install xgboost-cpu==3.4.1
Run from the repository root: python -m src.xgboost_forecast
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.forecast import NON_FEATURES, ROOT, TEST_START, backtest_rows, build_features, score


def load_panel(products):
    raw = ROOT / "data" / "raw"
    sales = pd.concat([
        chunk.loc[chunk["item_id"].isin(products)]
        for chunk in pd.read_csv(raw / "sales_train_validation.csv", chunksize=500)
    ], ignore_index=True)
    day_columns = [col for col in sales if col.startswith("d_")]
    panel = sales.melt(
        id_vars=["item_id", "store_id", "state_id"], value_vars=day_columns,
        var_name="d", value_name="sales",
    )
    calendar = pd.read_csv(raw / "calendar.csv", parse_dates=["date"])
    panel = panel.merge(calendar, on="d", validate="many_to_one")
    panel["snap_CA"] = np.select(
        [panel["state_id"].eq("TX"), panel["state_id"].eq("WI")],
        [panel["snap_TX"], panel["snap_WI"]], default=panel["snap_CA"],
    )
    panel["is_event"] = panel["event_name_1"].notna().astype(int)
    prices = pd.concat([
        chunk.loc[chunk["item_id"].isin(products)]
        for chunk in pd.read_csv(raw / "sell_prices.csv", chunksize=250_000)
    ], ignore_index=True)
    panel = panel.merge(prices, on=["item_id", "store_id", "wm_yr_wk"],
                        how="left", validate="many_to_one")
    panel["item_id"] = panel["item_id"] + "/" + panel["store_id"]
    return panel[["item_id", "date", "sales", "snap_CA", "is_event", "sell_price"]]


def fit_model(train, columns, cutoff, depth):
    weights = 0.08 + np.exp(-(cutoff - train["date"]).dt.days / 365)
    model = XGBRegressor(
        objective="reg:absoluteerror", eval_metric="mae", tree_method="hist",
        max_depth=depth, n_estimators=240, learning_rate=0.055, min_child_weight=20,
        subsample=0.85, colsample_bytree=0.85, n_jobs=4, random_state=42,
    )
    model.fit(train[columns], train["sales"], sample_weight=weights)
    return model


def evaluate(columns, fit_rows, validation, holdout, cutoff):
    model_scores = []
    for scope, candidates in (("Pooled", fit_rows), ("CA_1 only", fit_rows.loc[
        fit_rows["item_id"].str.endswith("/CA_1")
    ])):
        for depth in (3, 5):
            train = candidates.loc[candidates["date"] < cutoff]
            model = fit_model(train, columns, cutoff, depth)
            predictions = model.predict(validation[columns]).clip(min=0)
            mae = (validation["sales"] - predictions).abs().mean()
            model_scores.append((mae, scope, depth))
            print(f"Validation {scope}, depth {depth}: MAE {mae:.2f}", flush=True)

    _, scope, depth = min(model_scores)
    train = fit_rows.loc[fit_rows["date"] < TEST_START]
    if scope == "CA_1 only":
        train = train.loc[train["item_id"].str.endswith("/CA_1")]
    model = fit_model(train, columns, TEST_START, depth)
    error = holdout["sales"] - model.predict(holdout[columns]).clip(min=0)
    return scope, depth, score(holdout.assign(
        absolute_error=error.abs(), squared_error=error.pow(2),
    ))


def main():
    processed = ROOT / "data" / "processed"
    history = pd.read_csv(processed / "forecasting_features.csv", parse_dates=["date"])
    products = history["item_id"].unique()
    rows = build_features(load_panel(products))
    product_store = rows["item_id"].str.split("/", expand=True)
    identifiers = pd.get_dummies(product_store.rename(columns={0: "product", 1: "store"}), dtype=int)
    series_columns = [col for col in rows if col.startswith("product_")]
    rows = pd.concat([rows.drop(columns=series_columns), identifiers], axis=1)
    columns = rows.columns.difference(NON_FEATURES)
    fit_rows = rows.loc[rows["sales"].notna() & rows["sell_price"].notna()]
    target = rows.loc[rows["item_id"].str.endswith("/CA_1")]
    dates = sorted(history["date"].unique())
    cutoff = dates[int(len(dates) * 0.70)]
    validation = backtest_rows(target, cutoff, TEST_START - pd.Timedelta(days=1), len(products))
    holdout = backtest_rows(target, TEST_START, dates[-1], len(products))
    current = pd.read_csv(processed / "rolling_backtest_results.csv", parse_dates=["date", "window_start"])
    keys = ["item_id", "date", "window_start", "sales"]
    expected = holdout.assign(item_id=holdout["item_id"].str.removesuffix("/CA_1"))[keys]
    paired = expected.merge(current[keys], on=keys, validate="one_to_one")
    if len(paired) != len(holdout) or len(paired) != len(current):
        raise ValueError("XGBoost and the current model must use identical holdout rows.")
    scope, depth, metrics = evaluate(columns, fit_rows, validation, holdout, cutoff)
    results = pd.DataFrame({"XGBoost": metrics, "Current ensemble": score(current)})
    print(f"Validation selected {scope}, depth {depth}")
    print(f"Held-out results across {holdout['window_start'].nunique()} weeks and {len(holdout):,} predictions")
    print(results.round(2).to_string())


if __name__ == "__main__":
    main()
