# E-commerce Product Demand Forecasting with Time-Series Models and a Local LLM
## The question

An e-commerce system does not only need to know what sold yesterday.

It needs to estimate what might be sold next.

That estimate can influence inventory planning, replenishment, purchasing, and operational decisions. The difficult part is that product demand is not simply a list of independent numbers. It evolves through time. Yesterday's demand can be related to today's demand, the day of the week can matter, and recurring patterns can appear across weeks.

This project starts with a simple question:

> Can historical product demand be converted into a reproducible forecast, and can a local language model turn that numerical forecast into an understandable analyst note?

The project deliberately separates these two responsibilities.

The forecasting models produce numbers.

The language model explains those numbers.

---

## What the project is solving

The input is transaction-level e-commerce data.

Each transaction contains information such as:

- product identifier
- quantity purchased
- transaction timestamp
- unit price
- invoice information

The raw transactions are converted into a daily demand series for one product:

$$
y_t = \sum_{i \in t} q_i
$$

where:

- $y_t$ is the product demand on day $t$
- $q_i$ is the quantity associated with transaction $i$
- $i \in t$ means that the transaction occurred on day $t$

The forecasting problem then becomes:

$$
\hat{y}_{t+1}, \hat{y}_{t+2}, \ldots, \hat{y}_{t+h} = f(y_1, y_2, \ldots, y_t)
$$

where $h$ is the forecasting horizon.

For this implementation, the default horizon is 30 days.

---

## Dataset

The project uses the **Online Retail** dataset from the UCI Machine Learning Repository.

The dataset contains transaction records from a UK-based online retailer between December 2010 and December 2011. It contains product codes, quantities, timestamps, prices, invoices, customers, and countries.

The dataset is licensed under **CC BY 4.0**.

Source:

Chen, D. (2015). *Online Retail*. UCI Machine Learning Repository.

DOI: `10.24432/C5BW33`

Dataset:

https://archive.ics.uci.edu/dataset/352/online-retail

The raw dataset is deliberately not committed to this repository. The project downloads it locally and keeps it under `data/raw/`, which is gitignored.

This avoids unnecessarily redistributing the source dataset while preserving the required attribution.

---

## A small but important definition

The dataset contains returns and cancelled invoices.

For this project, "demand" means **positive fulfilled quantity**.

Therefore, the preprocessing removes:

- cancelled invoices
- non-positive quantities
- non-positive unit prices
- invalid timestamps
- missing product identifiers

This is a modeling assumption, not a universal definition of demand.

If a business wanted to forecast net demand including returns, the preprocessing definition would need to change.

---

# From transactions to a time series

Suppose a product has these transactions:

| Date | Quantity |
|---|---:|
| Monday | 4 |
| Monday | 7 |
| Tuesday | 3 |
| Wednesday | 0 |


The forecasting series becomes:

$$
[11, 3, 0, \ldots]
$$

Dates without transactions are explicitly represented as zero demand.

This matters because a missing transaction date can represent a day on which the product sold nothing. It should not automatically be interpreted as missing information.

---

# Why time-based validation matters

Random train/test splitting is inappropriate for forecasting.

A forecasting model should never learn from the future when predicting the past.

The project therefore uses chronological validation:

| Historical Observations | Final 30 Days |
|:-----------------------:|:-------------:|
| **Training Period**     | **Validation Period** |
| Used for model training | Held out as unseen data |

The final 30 days are held out as an unseen validation period.

The models never receive those observations during training.

This gives us a much more realistic estimate of how the forecasting pipeline behaves on future data.

---

# The models

Three progressively different approaches are evaluated.

## 1. Seasonal Naive baseline

The first model is intentionally simple.

For daily demand with weekly seasonality:

$$
\hat{y}_t = y_{t-7}
$$

In other words, Monday's forecast starts from the previous Monday, Tuesday from the previous Tuesday, and so on.

This baseline is important because a more complicated model should justify its additional complexity by improving over a simple seasonal rule.

---

## 2. SARIMA

The second approach models the statistical structure of the time series.

SARIMA can be represented as:

$$
\{SARIMA}(p,d,q)\times(P,D,Q)_s
$$

where:

- $p$: autoregressive order
- $d$: non-seasonal differencing order
- $q$: moving-average order
- $P$: seasonal autoregressive order
- $D$: seasonal differencing order
- $Q$: seasonal moving-average order
- $s$: seasonal period

The implementation uses a weekly seasonal period:

$$
s=7
$$

The project uses:

$$
\{SARIMA}(1,1,1)\times(1,1,1)_7
$$

The implementation uses `statsmodels`' SARIMAX interface.

---

## 3. Gradient boosting with lag features

The third approach converts the time series into a supervised-learning problem.

For each day, features include:

$$
X_t = \left[\, y_{t-1},\; y_{t-7},\; y_{t-14},\; y_{t-28},\; \text{rolling mean}_{7},\; \text{rolling mean}_{28},\; \text{day of week},\; \text{month},\; \ldots \right]
$$

The target is:

$$
y_t
$$

The model used is scikit-learn's `HistGradientBoostingRegressor`.

For future prediction, the model forecasts one day at a time and feeds its prediction back into the feature-generation process.

This is called **recursive multi-step forecasting**.

---

# Evaluation

The project reports several complementary metrics.

## Mean Absolute Error

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i-\hat{y}_i|
$$

MAE answers:

> On average, how many units was the forecast away from the actual demand?

---

## Root Mean Squared Error

$$
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i-\hat{y}_i)^2}
$$

RMSE penalizes large errors more strongly than MAE.

---

## Symmetric Mean Absolute Percentage Error

$$
SMAPE = \frac{100}{n} \sum_{i=1}^{n} \frac{2|y_i-\hat{y}_i|}{|y_i|+|\hat{y}_i|}
$$

The implementation defines the contribution as zero when both actual and predicted demand are zero.

---

## Weighted Absolute Percentage Error

$$
WAPE = 100 \frac{\sum_i |y_i-\hat{y}_i|}{\sum_i |y_i|}
$$

WAPE is useful for interpreting total forecast error relative to total observed demand.

The pipeline uses validation WAPE to select the final forecasting approach.

---

# Where the LLM fits

The LLM is intentionally **not** the forecasting model.

That distinction is important.

A language model is given structured forecasting information such as:

```text

Product

Historical demand

Recent demand averages

Validation metrics

Forecast horizon

Forecast total

Forecast range

```

It then produces an analyst-style explanation.

```text

             Historical Transactions

                       |

                       v

              Data Preparation

                       |

                       v

                Daily Demand

                       |

                       v

        +--------------+--------------+

        |              |              |

        v              v              v

 Seasonal Naive     SARIMA      Gradient Boosting

        |              |              |

        +--------------+--------------+

                       |

                       v

                 Validation

                       |

                       v

                Model Selection

                       |

                       v

               Future Forecast

                       |

                       +------------------+

                       |                  |

                       v                  v

                Forecast CSV       Structured Summary

                                          |

                                          v

                                   Local Qwen3 LLM

                                          |

                                          v

                                  Analyst Explanation

```

This architecture keeps numerical prediction deterministic and measurable while using the LLM for communication rather than pretending that free-form text generation is a substitute for forecasting evaluation.

---

# Local LLM

The project uses **Qwen3-4B through Ollama**.

The model can run locally, so the project does not require a paid API key or a cloud LLM provider.

Qwen3-4B is listed with an **Apache License 2.0** on its official model listing.

Ollama provides a local interface for running the model.

Model:

`qwen3:4b`

The LLM is optional.

If Ollama is unavailable, the complete forecasting pipeline still runs and produces the numerical forecast and evaluation results.

This is intentional: the core ML system should not fail just because the explanation layer is unavailable.

---

# Reproducibility

The project fixes the main machine-learning random seed:

```text

random_state = 42

```

The chronological split is deterministic.

The forecasting configuration is stored directly in the code.

The generated outputs include:

```text

outputs/

├── metrics.json

├── validation_predictions.csv

├── forecast.csv

├── run_summary.json

├── llm_forecast_explanation.md

├── 01_historical_demand.png

├── 02_weekly_pattern.png

├── 03_validation_forecasts.png

├── 04_future_forecast.png

└── 05_validation_residuals.png

```

The raw dataset and generated model artifacts are excluded from Git.

---


# Results

The first end-to-end experiment was completed successfully using the selected product:

**WHITE HANGING HEART T-LIGHT HOLDER**

The product contained **373 days** of daily demand history, including **304 active sales days** and **69 zero-demand days**. The mean daily demand was **101.58 units**.

The experiment used a chronological split:

- Training period: **2010-12-01 to 2011-11-08**
- Validation period: **2011-11-09 to 2011-12-08**
- Validation horizon: **30 days**

## Model comparison

| Model | MAE | RMSE | SMAPE | WAPE |
|---|---:|---:|---:|---:|
| Seasonal Naive | 272.5000 | 497.9743 | 83.87% | 206.54% |
| SARIMA | **90.9501** | 125.1483 | **75.94%** | **68.94%** |
| Gradient Boosting | 102.0394 | **121.6686** | 100.20% | 77.34% |


The Seasonal Naive model provides a useful baseline, while both SARIMA and Gradient Boosting substantially improve upon it. Based on the validation WAPE used by the pipeline for model selection, **SARIMA was selected as the final forecasting model**.

## Final forecast

The selected SARIMA model was refitted using the complete available history and used to generate a **30-day future demand forecast**.

- Forecast horizon: **30 days**
- Forecasted total demand: **2,747.09 units**
- Forecasted average daily demand: **91.57 units/day**

The pipeline also generated historical-demand, weekly-pattern, validation-forecast, future-forecast, and residual plots for visual analysis.

## LLM explanation

The numerical forecasting pipeline completed successfully without requiring an external API.

The optional local Qwen3-4B explanation layer was not enabled during this run because Ollama was not running. The forecasting results therefore remain independent of the LLM layer, while the generated analyst explanation can be added by running the local model separately. (will update it after few fixes)

## Visual Analysis

### Historical Demand

Daily product demand exhibits substantial variability, intermittent zero-demand days, and occasional sales spikes, making accurate forecasting challenging.
<img width="1901" height="784" alt="image" src="https://github.com/user-attachments/assets/0d6262ff-6ca2-4dda-b16b-854b7c28c8c2" />

### Model Validation

Models are evaluated on a chronological 30-day validation period. SARIMA achieves the lowest MAE (90.95) and WAPE (68.94%), while Gradient Boosting achieves the lowest RMSE (121.67).
<img width="1896" height="784" alt="image" src="https://github.com/user-attachments/assets/3eabfb40-c154-4a2a-b188-29968f5f633b" />


### Future Forecast

The selected SARIMA model is refitted on the available historical data to generate a 30-day forecast, with a predicted total of approximately 2,747 units.

<img width="1901" height="784" alt="image" src="https://github.com/user-attachments/assets/af2058a1-a2da-4940-a5fd-cec731784f1d" />

---

# Dataset attribution

Chen, D. (2015). Online Retail. UCI Machine Learning Repository.

DOI: `10.24432/C5BW33`

License: **CC BY 4.0**

https://archive.ics.uci.edu/dataset/352/online-retail

The raw dataset is not included in this repository.

---

# Model attribution

This project uses Qwen3-4B through Ollama.

Model:

`Qwen3-4B`

License shown on the official model listing: **Apache License 2.0**

The model is downloaded separately by the user and is not bundled with this repository.


> **Note:** Run the Streamlit UI using the following command:
>
> ```bash
> streamlit run app.py
> ```
