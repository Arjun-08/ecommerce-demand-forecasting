import json
from pathlib import Path

import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3:4b"


def build_prompt(summary: dict) -> str:
    return f"""
You are a forecasting analyst helping an e-commerce inventory team.

Use ONLY the numerical information provided below. Do not invent business facts.

Product:
{summary["product_id"]} - {summary["description"]}

Historical demand:
- History length: {summary["history_days"]} days
- Average daily demand: {summary["mean_daily_demand"]:.2f}
- Last 7-day average: {summary["last_7_day_average"]:.2f}
- Last 28-day average: {summary["last_28_day_average"]:.2f}

Validation results:
{json.dumps(summary["validation_metrics"], indent=2)}

Future forecast:
- Horizon: {summary["forecast_horizon"]} days
- Forecast total units: {summary["forecast_total"]:.2f}
- Forecast average per day: {summary["forecast_average"]:.2f}
- Minimum forecast: {summary["forecast_min"]:.2f}
- Maximum forecast: {summary["forecast_max"]:.2f}

Write a concise analyst note with exactly these sections:

1. Forecast summary
2. Observed demand pattern
3. Model performance
4. Inventory planning consideration
5. Caveats

Do not claim that the forecast is certain. Clearly distinguish observed historical behavior from model-generated forecasts.
""".strip()


def generate_explanation(summary: dict, model: str = DEFAULT_MODEL) -> str:
    prompt = build_prompt(summary)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a concise e-commerce forecasting analyst. "
                    "Never fabricate facts or numerical results."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }

    print(f"[LLM] Calling local Ollama model: {model}...")

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

        data = response.json()
        text = data["message"]["content"].strip()

        print("[LLM] Explanation generated successfully.")
        return text

    except requests.exceptions.ConnectionError:
        message = (
            "LLM explanation was skipped because Ollama is not running. "
            "Start Ollama and run `ollama pull qwen3:4b` to enable it."
        )
        print(f"[LLM] {message}")
        return message

    except Exception as exc:
        message = f"LLM explanation unavailable: {exc}"
        print(f"[LLM] {message}")
        return message
