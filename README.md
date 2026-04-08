# RuralWatch — Rural Hospital Financial Distress Early Warning System

[![Python](https://img.shields.io/badge/Python-3.9-blue)](https://python.org)
[![PySpark](https://img.shields.io/badge/PySpark-3.5.1-orange)](https://spark.apache.org)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)](https://xgboost.readthedocs.io)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue)](https://mlflow.org)

## What This Project Does

RuralWatch is an end-to-end ML pipeline that predicts which rural US hospitals are at risk of financial distress or permanent closure within the next 1–3 years, using publicly available CMS cost report data. Every prediction is accompanied by a SHAP waterfall explanation showing which financial ratios drove the score — making the system transparent enough for policy use.

## Why It Was Built

Over 140 rural hospitals have permanently closed since 2010, leaving millions of Americans without access to emergency care within a reasonable distance. The closures follow predictable financial patterns, but no systematic early warning system exists. A December 2025 systematic review in *BMC Health Services Research* examined every published study of rural hospital closures from 2013–2024 and found that no study had yet applied ML to predict closures before they happen — and explicitly called for an AI-driven early warning system. RuralWatch is a direct response to that gap.

## Architecture

```
CMS Cost Reports (2011–2022) + USDA Rural Codes + UNC Sheps Closure Registry
                                    │
                                    ▼
                     [BRONZE LAYER] — Raw Delta Lake tables
                                    │
                    PySpark: RUCC join, label engineering, imputation
                                    │
                                    ▼
                [SILVER LAYER] — Rural hospitals only, distress labels attached
                                    │
                    PySpark: Star schema construction, ratio engineering
                                    │
                                    ▼
          [GOLD LAYER] — fact_hospital_financials + 4 dimension tables
                                    │
                    XGBoost, SMOTE, temporal train/test split
                                    │
                                    ▼
                    [MLFLOW] — Experiment tracking, model registry
                                    │
                    Kafka producer/consumer (quarterly CMS updates)
                                    │
                                    ▼
              [DASHBOARD] — FastAPI endpoint + Streamlit risk monitor
```

## Tech Stack

| Tool | Purpose |
|---|---|
| PySpark 3.5.1 | ELT across all Medallion layers + star schema construction |
| Delta Lake | ACID storage for Bronze / Silver / Gold layers |
| Star Schema | Fact table + 4 dimension tables in Gold layer |
| XGBoost | Financial distress classifier (best model) |
| SMOTE | Class imbalance handling (~3–5% positive rate) |
| MLflow | Experiment tracking across 3 model runs |
| SHAP | Per-hospital explainability via waterfall plots |
| Kafka | Simulated streaming ingestion of quarterly CMS updates |
| FastAPI | REST scoring endpoint (`/predict`) |
| Streamlit | Interactive distress assessment dashboard |

## Data Sources

All data is fully public — no HIPAA concerns, no IRB requirements.

| Source | What It Provides |
|---|---|
| [CMS Hospital Provider Cost Report](https://data.cms.gov/provider-compliance/cost-report/hospital-provider-cost-report) | Annual financial filings for every Medicare-certified US hospital (2011–2022) |
| [USDA Rural-Urban Continuum Codes](https://www.ers.usda.gov/data-products/rural-urban-continuum-codes/) | County-level rurality scores (1 = metro, 9 = most rural) |
| [UNC Sheps Center Closure Registry](https://www.shepscenter.unc.edu/programs-projects/rural-health/rural-hospital-closures/) | Ground-truth labels: every rural hospital closure since 2005 |

## Target Variable

A hospital-year is labeled **distress = 1** if the hospital permanently closed within 3 years of the fiscal year filing. The 3-year window reflects the realistic planning horizon for state health department intervention. Only permanent full closures are labeled positive — conversions to other facility types are excluded.

## Features (10 engineered financial ratios)

| Feature | What It Measures |
|---|---|
| `operating_margin` | Profitability from core patient care operations |
| `total_margin` | Overall profitability including non-operating income |
| `cost_to_revenue` | Whether the hospital spends more than it earns |
| `medicaid_pct` | Medicaid dependency (pays below cost for most services) |
| `medicare_day_pct` | Medicare inpatient day share (government payer pressure) |
| `occupancy_proxy` | Bed utilization — low occupancy = fixed costs unsustainable |
| `days_cash_on_hand` | How many days of expenses covered by available cash |
| `current_ratio` | Short-term solvency (< 1.0 = cannot cover current obligations) |
| `uncompensated_care_pct` | Charity care + bad debt as share of revenue |
| `revenue_yoy_change` | Year-over-year revenue trajectory |

## Model Results

| Model | Val ROC-AUC | Val PR-AUC |
|---|---|---|
| Logistic Regression | — | — |
| Random Forest | — | — |
| **XGBoost** | **0.8789** | **0.0726** |

Temporal split: Train 2011–2018 · Validation 2019–2020 · Test 2021–2022

PR-AUC is the primary metric. With a ~3–5% positive rate, ROC-AUC is optimistic by construction — PR-AUC evaluates performance specifically on the rare positive class, which is what matters for policy targeting.

## Key Design Decisions

**Temporal train/test split** — Random splitting would leak future data into training. The model is trained only on data available at prediction time, simulating real production conditions.

**3-year closure window** — 1 year provides insufficient lead time for intervention. 5 years introduces closures caused by factors (e.g. COVID) that didn't exist at prediction time. 3 years reflects published rural hospital distress literature.

**State-year median imputation** — Rural hospital economics vary enormously by state. Global median imputation would systematically misrepresent hospitals in high-cost or low-cost states.

**Star schema in Gold layer** — The Medallion architecture describes data quality stages. The star schema organizes the Gold layer for efficient analytical querying — fact table in the center, dimension tables (hospital, geography, time, payer) radiating outward.

## How to Run

```bash
# 1. Clone and set up environment
git clone https://github.com/pillaiarjun/ruralwatch.git
cd ruralwatch
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Start Kafka (requires Docker Desktop)
docker-compose up -d

# 3. Run notebooks in order
# notebooks/01_eda.ipynb through 05_modeling.ipynb

# 4. Start the API
uvicorn dashboard.model_api:app --reload

# 5. Start the dashboard (separate terminal)
streamlit run dashboard/app.py
```

## Project Structure

```
ruralwatch/
├── data/
│   ├── bronze/          # Raw Delta tables (CMS, USDA, Sheps)
│   ├── silver/          # Cleaned, labeled, rural-filtered
│   └── gold/
│       ├── fact/        # fact_hospital_financials
│       └── dims/        # dim_hospital, dim_geography, dim_time, dim_payer
├── notebooks/           # Phase-by-phase build notebooks
├── src/
│   ├── utils/           # Spark session config
│   └── streaming/       # Kafka producer + consumer
├── dashboard/
│   ├── model_api.py     # FastAPI endpoint
│   └── app.py           # Streamlit dashboard
├── docs/                # SHAP plots, EDA charts
├── docker-compose.yml   # Local Kafka setup
└── requirements.txt
```

---

*Arjun Pillai · UC Berkeley Data Science · github.com/pillaiarjun/ruralwatch*