from pathlib import Path
import json
import pandas as pd
import streamlit as st


OUTPUT_DIR = Path("outputs")

st.set_page_config(
    page_title="E-commerce Demand Forecasting",
    layout="wide",
)

st.title("E-commerce Product Demand Forecasting")
st.caption(
    "Time-series demand forecasting with classical and machine-learning "
    "models, plus a local LLM explanation."
)

if not (OUTPUT_DIR / "run_summary.json").exists():
    st.warning(
        "No completed run found. Execute `python main.py` first."
    )
    st.stop()

with open(OUTPUT_DIR / "run_summary.json", encoding="utf-8") as f:
    summary = json.load(f)

st.subheader("Forecast overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Product", summary["product_id"])
c2.metric("Selected model", summary["selected_model"])
c3.metric("Forecast horizon", f'{summary["forecast_horizon"]} days')
c4.metric("Forecast units", f'{summary["forecast_total"]:.0f}')

st.write(f'**Description:** {summary["description"]}')

st.subheader("Validation metrics")

metrics = []
for model_name, values in summary["metrics"].items():
    row = {"Model": model_name}
    row.update(values)
    metrics.append(row)

metrics_df = pd.DataFrame(metrics).set_index("Model")
st.dataframe(metrics_df, use_container_width=True)

forecast_path = OUTPUT_DIR / "forecast.csv"
if forecast_path.exists():
    forecast = pd.read_csv(forecast_path, parse_dates=["date"])
    forecast = forecast.set_index("date")

    st.subheader("Future demand forecast")
    st.line_chart(forecast["forecast"])

explanation_path = OUTPUT_DIR / "llm_forecast_explanation.md"
if explanation_path.exists():
    st.subheader("LLM analyst note")
    st.markdown(explanation_path.read_text(encoding="utf-8"))

st.subheader("Generated plots")

plot_names = [
    "01_historical_demand.png",
    "02_weekly_pattern.png",
    "03_validation_forecasts.png",
    "04_future_forecast.png",
    "05_validation_residuals.png",
]

for plot_name in plot_names:
    path = OUTPUT_DIR / plot_name
    if path.exists():
        st.image(str(path), caption=plot_name)
