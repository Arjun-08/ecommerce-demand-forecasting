from pathlib import Path
import json
import joblib
import pandas as pd

from src.data import load_transactions, choose_product, build_daily_series
from src.models import (
    seasonal_naive_forecast,
    fit_sarima_forecast,
    fit_gradient_boosting,
    forecast_gradient_boosting,
)
from src.evaluate import evaluate, print_metrics
from src.plots import (
    plot_history,
    plot_weekly_pattern,
    plot_validation,
    plot_future_forecast,
    plot_residuals,
)
from src.llm import generate_explanation


DATA_PATH = Path("data/raw/online_retail.csv")
OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")

VALIDATION_DAYS = 30
FORECAST_HORIZON = 30


def main():
    print("=" * 72)
    print("E-COMMERCE PRODUCT DEMAND FORECASTING")
    print("=" * 72)

    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    print("\n[STEP 1] Loading and cleaning transaction data")
    df = load_transactions(DATA_PATH)

    print("\n[STEP 2] Selecting a forecastable product")
    product_id, description = choose_product(df)

    print("\n[STEP 3] Building daily demand series")
    series = build_daily_series(df, product_id)

    if len(series) <= VALIDATION_DAYS + 60:
        raise RuntimeError(
            "The selected product does not have enough history for the "
            "configured validation window."
        )

    plot_history(series, OUTPUT_DIR / "01_historical_demand.png")
    plot_weekly_pattern(series, OUTPUT_DIR / "02_weekly_pattern.png")

    print("\n[STEP 4] Time-based train/validation split")
    train = series.iloc[:-VALIDATION_DAYS].copy()
    validation = series.iloc[-VALIDATION_DAYS:].copy()

    print(f"[SPLIT] Train: {train.index.min().date()} -> {train.index.max().date()}")
    print(
        f"[SPLIT] Validation: "
        f"{validation.index.min().date()} -> {validation.index.max().date()}"
    )
    print(f"[SPLIT] Validation horizon: {len(validation)} days")

    predictions = {}
    metrics = {}

    print("\n[STEP 5] Seasonal-naive baseline")
    naive = seasonal_naive_forecast(train, len(validation), season=7)
    predictions["Seasonal Naive"] = naive["forecast"]
    metrics["Seasonal Naive"] = evaluate(
        validation["sales"], naive["forecast"]
    )
    print_metrics("Seasonal Naive", metrics["Seasonal Naive"])

    print("\n[STEP 6] SARIMA")
    sarima_forecast, sarima_result = fit_sarima_forecast(
        train,
        len(validation),
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
    )
    predictions["SARIMA"] = sarima_forecast["forecast"]
    metrics["SARIMA"] = evaluate(
        validation["sales"], sarima_forecast["forecast"]
    )
    print_metrics("SARIMA", metrics["SARIMA"])

    print("\n[STEP 7] Gradient boosting with lag features")
    gb_model = fit_gradient_boosting(train)
    gb_forecast = forecast_gradient_boosting(
        gb_model,
        train,
        len(validation),
    )
    predictions["Gradient Boosting"] = gb_forecast["forecast"]
    metrics["Gradient Boosting"] = evaluate(
        validation["sales"], gb_forecast["forecast"]
    )
    print_metrics("Gradient Boosting", metrics["Gradient Boosting"])

    print("\n[STEP 8] Validation plots")
    prediction_frame = pd.DataFrame(predictions, index=validation.index)

    plot_validation(
        validation["sales"],
        predictions,
        OUTPUT_DIR / "03_validation_forecasts.png",
    )

    # Use the model with the lowest WAPE for the final refit.
    selected_model_name = min(
        metrics,
        key=lambda name: metrics[name]["WAPE"]
        if pd.notna(metrics[name]["WAPE"])
        else float("inf"),
    )

    print()
    print("=" * 72)
    print(f"SELECTED MODEL BY VALIDATION WAPE: {selected_model_name}")
    print("=" * 72)

    print("\n[STEP 9] Refit selected model on all available history")

    if selected_model_name == "Seasonal Naive":
        final_forecast = seasonal_naive_forecast(
            series, FORECAST_HORIZON, season=7
        )
        selected_model = None

    elif selected_model_name == "SARIMA":
        final_forecast, final_sarima_result = fit_sarima_forecast(
            series,
            FORECAST_HORIZON,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
        )
        selected_model = None

    else:
        selected_model = fit_gradient_boosting(series)
        final_forecast = forecast_gradient_boosting(
            selected_model,
            series,
            FORECAST_HORIZON,
        )

        joblib.dump(
            selected_model,
            MODEL_DIR / "gradient_boosting_model.joblib",
        )
        print(
            f"[MODEL] Saved: {MODEL_DIR / 'gradient_boosting_model.joblib'}"
        )

    final_forecast.to_csv(OUTPUT_DIR / "forecast.csv")
    prediction_frame.to_csv(OUTPUT_DIR / "validation_predictions.csv")

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_future_forecast(
        series,
        final_forecast,
        OUTPUT_DIR / "04_future_forecast.png",
    )

    plot_residuals(
        validation["sales"],
        predictions[selected_model_name],
        OUTPUT_DIR / "05_validation_residuals.png",
    )

    print("\n[STEP 10] Building LLM analyst context")

    best_metrics = metrics[selected_model_name]

    summary = {
        "product_id": product_id,
        "description": description,
        "history_days": int(len(series)),
        "mean_daily_demand": float(series["sales"].mean()),
        "last_7_day_average": float(series["sales"].tail(7).mean()),
        "last_28_day_average": float(series["sales"].tail(28).mean()),
        "validation_metrics": {
            selected_model_name: {
                k: float(v) for k, v in best_metrics.items()
            }
        },
        "forecast_horizon": FORECAST_HORIZON,
        "forecast_total": float(final_forecast["forecast"].sum()),
        "forecast_average": float(final_forecast["forecast"].mean()),
        "forecast_min": float(final_forecast["forecast"].min()),
        "forecast_max": float(final_forecast["forecast"].max()),
    }

    explanation = generate_explanation(summary)

    with open(
        OUTPUT_DIR / "llm_forecast_explanation.md",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(explanation)

    with open(
        OUTPUT_DIR / "run_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "product_id": product_id,
                "description": description,
                "selected_model": selected_model_name,
                "validation_days": VALIDATION_DAYS,
                "forecast_horizon": FORECAST_HORIZON,
                "metrics": metrics,
                "forecast_total": summary["forecast_total"],
                "forecast_average": summary["forecast_average"],
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 72)
    print("PIPELINE COMPLETE")
    print("=" * 72)
    print(f"Product: {product_id} - {description}")
    print(f"Selected model: {selected_model_name}")
    print(f"Forecast horizon: {FORECAST_HORIZON} days")
    print(f"Forecast total: {summary['forecast_total']:.2f} units")
    print(f"Forecast average/day: {summary['forecast_average']:.2f}")
    print(f"Outputs: {OUTPUT_DIR.resolve()}")
    print()
    print("Generated files:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
