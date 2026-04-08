# ---- dashboard/model_api.py ----

from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np

app = FastAPI(title='RuralWatch Distress API', version='1.0')

# Load the model once at startup — not on every request
# This path points to the XGBoost artifact saved by MLflow during Phase 5
MODEL_URI = 'file:///Users/arjunpillai/Desktop/ruralwatch/mlruns/130426424173145298/models/m-f9b7c03a44fe4d90b15c42e461610c23/artifacts'

model = mlflow.sklearn.load_model(MODEL_URI)

# Pydantic model defines the expected request body shape
# FastAPI uses this for automatic input validation and API docs
class HospitalFinancials(BaseModel):
    operating_margin: float
    total_margin: float
    cost_to_revenue: float
    medicaid_pct: float
    medicare_day_pct: float
    occupancy_proxy: float
    days_cash_on_hand: float
    current_ratio: float
    uncompensated_care_pct: float
    revenue_yoy_change: float

@app.get('/')
def root():
    return {'status': 'RuralWatch API is running'}

@app.post('/predict')
def predict(f: HospitalFinancials):
    # Assemble features in the exact order used during model training
    feature_array = np.array([[
        f.operating_margin,
        f.total_margin,
        f.cost_to_revenue,
        f.medicaid_pct,
        f.medicare_day_pct,
        f.occupancy_proxy,
        f.days_cash_on_hand,
        f.current_ratio,
        f.uncompensated_care_pct,
        f.revenue_yoy_change
    ]])

    # predict_proba returns [[prob_class_0, prob_class_1]]
    # Index [0][1] gives the distress probability
    risk_score = float(model.predict_proba(feature_array)[0][1])

    # Risk tier thresholds — tuned to match the 0.4 threshold used in Phase 5 fairness analysis
    if risk_score >= 0.6:
        risk_tier = 'HIGH'
    elif risk_score >= 0.35:
        risk_tier = 'ELEVATED'
    else:
        risk_tier = 'LOW'

    return {
        'risk_score': round(risk_score, 4),
        'risk_tier': risk_tier
    }