import numpy as np
import pandas as pd


LAGS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 28]


def make_features(series: pd.DataFrame) -> pd.DataFrame:
    df = series.copy()
    df["day_of_week"] = df.index.dayofweek
    df["day_of_month"] = df.index.day
    df["month"] = df.index.month
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    for lag in LAGS:
        df[f"lag_{lag}"] = df["sales"].shift(lag)

    shifted = df["sales"].shift(1)
    for window in ROLLING_WINDOWS:
        df[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
        df[f"rolling_std_{window}"] = shifted.rolling(window).std()

    return df


def recursive_forecast(model, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Recursive multi-step forecast for the tabular ML model.

    At each future day, features are generated only from observations that
    would have been available at that point in time.
    """
    history = history.copy()
    predictions = []

    for _ in range(horizon):
        next_date = history.index.max() + pd.Timedelta(days=1)

        temp = pd.concat(
            [history, pd.DataFrame({"sales": [np.nan]}, index=[next_date])]
        )
        features = make_features(temp)

        X_next = features.loc[[next_date]].drop(columns=["sales"])

        if X_next.isna().any().any():
            raise RuntimeError(
                "NaN encountered during recursive forecasting. "
                "Check that enough historical observations are available."
            )

        pred = float(model.predict(X_next)[0])
        pred = max(0.0, pred)

        predictions.append((next_date, pred))
        history.loc[next_date, "sales"] = pred

    return pd.DataFrame(predictions, columns=["date", "forecast"]).set_index("date")
