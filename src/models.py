import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .features import make_features, recursive_forecast


def seasonal_naive_forecast(train: pd.DataFrame, horizon: int, season: int = 7):
    values = train["sales"].to_numpy(dtype=float)

    if len(values) < season:
        raise ValueError("Not enough observations for seasonal naive forecasting.")

    repeated = np.resize(values[-season:], horizon)
    dates = pd.date_range(
        train.index.max() + pd.Timedelta(days=1),
        periods=horizon,
        freq="D",
    )
    return pd.DataFrame({"forecast": repeated}, index=dates)


def fit_sarima_forecast(
    train: pd.DataFrame,
    horizon: int,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
):
    print("[MODEL] Fitting SARIMA...")
    print(f"[MODEL] order={order}, seasonal_order={seasonal_order}")

    model = SARIMAX(
        train["sales"],
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    result = model.fit(disp=False)

    forecast = result.forecast(steps=horizon)
    forecast = np.maximum(0.0, np.asarray(forecast, dtype=float))

    dates = pd.date_range(
        train.index.max() + pd.Timedelta(days=1),
        periods=horizon,
        freq="D",
    )

    return pd.DataFrame({"forecast": forecast}, index=dates), result


def fit_gradient_boosting(train: pd.DataFrame):
    print("[MODEL] Building lag/rolling features...")
    features = make_features(train).dropna()

    X = features.drop(columns=["sales"])
    y = features["sales"]

    print(f"[MODEL] Training rows: {len(X):,}")
    print(f"[MODEL] Feature count: {X.shape[1]}")

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )

    print("[MODEL] Fitting HistGradientBoostingRegressor...")
    model.fit(X, y)

    return model


def forecast_gradient_boosting(model, train: pd.DataFrame, horizon: int):
    print(f"[MODEL] Generating recursive {horizon}-day forecast...")
    return recursive_forecast(model, train, horizon)
