# ---- dashboard/app.py ----

import streamlit as st
import pandas as pd
import numpy as np
import shap
import mlflow.sklearn
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title='RuralWatch — Rural Hospital Distress Monitor',
    page_icon='🏥',
    layout='wide'
)

MODEL_URI = 'file:///Users/arjunpillai/Desktop/ruralwatch/mlruns/130426424173145298/models/m-f9b7c03a44fe4d90b15c42e461610c23/artifacts'

# @st.cache_resource loads the model once and reuses it across all user interactions
# Without this, Streamlit would reload the model on every slider move
@st.cache_resource
def load_model():
    return mlflow.sklearn.load_model(MODEL_URI)

model = load_model()

FEATURE_COLS = [
    'operating_margin', 'total_margin', 'cost_to_revenue',
    'medicaid_pct', 'medicare_day_pct', 'occupancy_proxy',
    'days_cash_on_hand', 'current_ratio',
    'uncompensated_care_pct', 'revenue_yoy_change'
]

st.title('🏥 RuralWatch — Rural Hospital Financial Distress Monitor')
st.caption(
    'Predicts 3-year closure risk for rural US hospitals using CMS cost report data. '
    'Built on publicly available CMS HCRIS data. For research purposes only.'
)

left, right = st.columns([1, 1.5])

with left:
    st.subheader('Hospital Financial Profile')

    operating_margin     = st.slider('Operating Margin',        -0.5,  0.3,  -0.05, 0.01)
    total_margin         = st.slider('Total Margin',            -0.5,  0.3,  -0.02, 0.01)
    cost_to_revenue      = st.slider('Cost-to-Revenue Ratio',    0.5,  2.0,   1.05, 0.01)
    medicaid_pct         = st.slider('Medicaid Payer Mix %',     0.0,  1.0,   0.25, 0.01)
    medicare_day_pct     = st.slider('Medicare Day Mix %',       0.0,  1.0,   0.45, 0.01)
    occupancy_proxy      = st.slider('Occupancy Proxy',          0.0,  1.0,   0.45, 0.01)
    days_cash_on_hand    = st.slider('Days Cash on Hand',        0.0, 200.0,  45.0, 1.0)
    current_ratio        = st.slider('Current Ratio',            0.0,  5.0,   1.5,  0.05)
    uncompensated_care   = st.slider('Uncompensated Care %',     0.0,  0.5,   0.05, 0.01)
    revenue_yoy_change   = st.slider('YoY Revenue Change',      -0.3,  0.3,  -0.02, 0.01)

    run_button = st.button('Run Distress Assessment', type='primary')

with right:
    if run_button:
        features = np.array([[
            operating_margin, total_margin, cost_to_revenue,
            medicaid_pct, medicare_day_pct, occupancy_proxy,
            days_cash_on_hand, current_ratio,
            uncompensated_care, revenue_yoy_change
        ]])

        risk_score = float(model.predict_proba(features)[0][1])

        if risk_score >= 0.6:
            risk_tier = 'HIGH'
        elif risk_score >= 0.35:
            risk_tier = 'ELEVATED'
        else:
            risk_tier = 'LOW'

        st.subheader('Distress Assessment Results')

        if risk_tier == 'HIGH':
            st.error(f'⚠️ HIGH RISK — 3-Year Closure Probability: {risk_score:.1%}')
        elif risk_tier == 'ELEVATED':
            st.warning(f'⚡ ELEVATED RISK — 3-Year Closure Probability: {risk_score:.1%}')
        else:
            st.success(f'✅ LOW RISK — 3-Year Closure Probability: {risk_score:.1%}')

        st.metric('Risk Score', f'{risk_score:.4f}')
        st.progress(float(min(risk_score * 5, 1.0)))  # Scale for visibility given model range

        st.subheader('Why This Score? (SHAP Explanation)')

        explainer = shap.TreeExplainer(model)
        features_df = pd.DataFrame(features, columns=FEATURE_COLS)
        shap_values = explainer.shap_values(features_df)

        # shap_values may be a list (binary classification) or a single array
        # For binary XGBoost via sklearn wrapper, it's typically a single array
        sv = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
        ev = (explainer.expected_value[1]
              if isinstance(explainer.expected_value, (list, np.ndarray))
              else explainer.expected_value)

        fig, ax = plt.subplots(figsize=(8, 4))
        shap.waterfall_plot(
            shap.Explanation(
                values=sv,
                base_values=ev,
                data=features[0],
                feature_names=FEATURE_COLS
            ),
            show=False
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    else:
        st.info('Set the financial parameters on the left and click **Run Distress Assessment**.')