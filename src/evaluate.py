import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = np.abs(y_true) + np.abs(y_pred)
    numerator = 2.0 * np.abs(y_pred - y_true)

    values = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator != 0,
    )
    return 100.0 * np.mean(values)


def wape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return np.nan

    return 100.0 * np.sum(np.abs(y_true - y_pred)) / denominator


def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "SMAPE": smape(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
    }


def print_metrics(name, metrics):
    print()
    print("-" * 60)
    print(f"EVALUATION: {name}")
    print("-" * 60)

    for metric, value in metrics.items():
        if np.isnan(value):
            print(f"{metric:<10}: N/A")
        elif metric in {"SMAPE", "WAPE"}:
            print(f"{metric:<10}: {value:.2f}%")
        else:
            print(f"{metric:<10}: {value:.4f}")
