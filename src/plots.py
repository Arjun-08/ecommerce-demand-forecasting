from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def save_plot(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved: {path}")


def plot_history(series, path):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(series.index, series["sales"], linewidth=1)
    ax.set_title("Historical Daily Demand")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units sold")
    ax.grid(alpha=0.25)
    save_plot(fig, path)


def plot_weekly_pattern(series, path):
    temp = series.copy()
    temp["day_of_week"] = temp.index.dayofweek

    grouped = temp.groupby("day_of_week")["sales"].mean()

    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, grouped.reindex(range(7)).values)
    ax.set_title("Average Demand by Day of Week")
    ax.set_xlabel("Day")
    ax.set_ylabel("Average units sold")
    ax.grid(axis="y", alpha=0.25)
    save_plot(fig, path)


def plot_validation(actual, predictions, path):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(actual.index, actual.values, label="Actual", linewidth=2)

    for name, pred in predictions.items():
        ax.plot(pred.index, pred.values, label=name, linewidth=1.6)

    ax.set_title("Validation Forecast vs Actual Demand")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units sold")
    ax.legend()
    ax.grid(alpha=0.25)
    save_plot(fig, path)


def plot_future_forecast(history, forecast, path):
    recent = history.tail(120)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(recent.index, recent["sales"], label="Historical", linewidth=2)
    ax.plot(
        forecast.index,
        forecast["forecast"],
        label="Forecast",
        linewidth=2,
    )

    ax.set_title("Historical Demand and Future Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units sold")
    ax.legend()
    ax.grid(alpha=0.25)
    save_plot(fig, path)


def plot_residuals(actual, predicted, path):
    residuals = actual.values - predicted.values

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0, linewidth=1)
    ax.plot(actual.index, residuals, linewidth=1)
    ax.set_title("Validation Residuals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Actual - Forecast")
    ax.grid(alpha=0.25)
    save_plot(fig, path)
