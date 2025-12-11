# src/app_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os

# Configure page
st.set_page_config(
    page_title="Solar Power Forecasting",
    page_icon="☀️",
    layout="wide"
)

# === Helper Functions ===

def load_model_direct(plant_num, target_type):
    """Load model directly from models folder"""
    target_lower = target_type.lower().replace("_power", "")
    
    # Map available models
    model_files = {
        ("1", "ac"): "plant1_ac_mlp.pkl",
        ("1", "dc"): "plant1_dc_mlp_transformed.pkl",
        ("2", "ac"): "plant2_ac_lasso.pkl",
        ("2", "dc"): "plant2_dc_lasso.pkl"
    }
    
    filename = model_files.get((plant_num, target_lower))
    if not filename:
        return None, None
    
    path = os.path.join(filename)
    
    if not os.path.exists(path):
        return None, filename
    
    try:
        model = joblib.load(path)
        model_name = "MLP (Neural Net)" if "mlp" in filename else "Lasso"
        return model, model_name
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
        return None, filename

def preprocess_input(irradiation, amb_temp, mod_temp, hour, source_key):
    """Prepare input data for model prediction"""
    # Cyclic encoding for hour
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    data = {
        "IRRADIATION": [irradiation],
        "AMBIENT_TEMPERATURE": [amb_temp],
        "MODULE_TEMPERATURE": [mod_temp],
        "Hour_sin": [hour_sin],  # Capital H to match model's expected columns
        "Hour_cos": [hour_cos],  # Capital H to match model's expected columns
        "SOURCE_KEY": [source_key]
    }
    
    df = pd.DataFrame(data)
    return df

def make_prediction(model, input_df):
    """Make prediction and ensure non-negative result"""
    try:
        prediction = model.predict(input_df)[0]
        return max(0.0, prediction)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0.0

# === Main Dashboard ===

st.title("☀️ Solar Power Dashboard")
st.markdown("### Predict power generation based on environmental conditions")

st.divider()

# === Configuration Section ===
config_col1, config_col2 = st.columns(2)

with config_col1:
    st.subheader("🏭 Plant Configuration")
    plant_num = st.radio(
        "Select Plant",
        ["1", "2"],
        format_func=lambda x: f"Plant {x}",
        horizontal=True
    )
    
    target_type = st.radio(
        "Power Type",
        ["DC_POWER", "AC_POWER"],
        horizontal=True
    )
    
    # Default inverter keys
    default_inverters = {
        "1": [
            "1BY6WEcLGh8j5v7", "1IF53ai7Xc0U56Y", "3PZuoBAID5Wc2HD", "4QWxYZaBcD9EfGhI", 
            "5MNoPqRsTuVwXyZ1", "6AaBbCcDdEeFfGg1", "7HhIiJjKkLlMmNn1", "8OoPpQqRrSsTtUu1",
            "9VvWwXxYyZzAaBb1", "0CcDdEeFfGgHhIi1", "1JjKkLlMmNnOoPp1", "2QqRrSsTtUuVvWw1",
            "3XxYyZzAaBbCcDd1", "4EeFfGgHhIiJjKk1", "5LlMmNnOoPpQqRr1", "6SsTtUuVvWwXxYy1",
            "7ZzAaBbCcDdEeFf1", "8GgHhIiJjKkLlMm1", "9NnOoPpQqRrSsTt1", "0UuVvWwXxYyZzAa1"
        ],
        "2": [
            "7JYdWkrLSPkdwr4", "McdE0feGCSoy8RC", "VHMLBKoKgIrUVDU", "6LkJhGfDsAqWePo2", 
            "8NbVcXzQwErTyUiO", "9PaSdFgHjKlZxCv2", "0QwErTyUiOpAsDfG2", "1HjKlZxCvBnMqWe2",
            "2RtYuIoPaSdFgHj2", "3KlZxCvBnMqWeRtY2", "4UiOpAsDfGhJkLzX2", "5CvBnMqWeRtYuIoP2",
            "6AsDfGhJkLzXcVbN2", "7MqWeRtYuIoPaSdF2", "8GhJkLzXcVbNmQwE2", "9RtYuIoPaSdFgHjK2",
            "0LzXcVbNmQwErTyU2", "1IoPaSdFgHjKlZxC2", "2VbNmQwErTyUiOpA2", "3SdFgHjKlZxCvBnM2"
        ]
    }
    
    source_key = st.selectbox(
        "⚡ Inverter ID",
        default_inverters[plant_num]
    )

with config_col2:
    st.subheader("⚙️ Environmental Parameters")
    
    irradiation = st.slider(
        "☀️ Solar Irradiation (kW/m²)",
        min_value=0.0,
        max_value=1.5,
        value=0.6,
        step=0.01,
        help="Solar radiation intensity"
    )
    
    amb_temp = st.slider(
        "🌡️ Ambient Temperature (°C)",
        min_value=10.0,
        max_value=40.0,
        value=25.0,
        step=0.5
    )
    
    mod_temp = st.slider(
        "📟 Module Temperature (°C)",
        min_value=10.0,
        max_value=60.0,
        value=35.0,
        step=0.5
    )
    
    hour = st.slider(
        "🕐 Hour of Day",
        min_value=5,
        max_value=19,
        value=12,
        step=1
    )

st.divider()

# === Load Model and Make Prediction ===
model, model_name = load_model_direct(plant_num, target_type)

if model is None:
    st.error(f"⚠️ Model not found in models folder")
    st.info("Available models: plant1_ac_mlp.pkl, plant1_dc_mlp_transformed.pkl, plant2_ac_lasso.pkl, plant2_dc_lasso.pkl")
    st.stop()

# Prepare input and predict
input_df = preprocess_input(irradiation, amb_temp, mod_temp, hour, source_key)
prediction = make_prediction(model, input_df)

irr_values = np.linspace(0.0, 1.5, 100)
predictions = []

for irr in irr_values:
    temp_input = preprocess_input(irr, amb_temp, mod_temp, hour, source_key)
    pred = make_prediction(model, temp_input)
    predictions.append(pred)


st.subheader("🔮 Prediction Results")
left, right = st.columns([1,2])

import streamlit as st

with left:
    st.markdown("#### Current Prediction")
    # st.subheader("Current Prediction")
    st.metric(
        label=f"Predicted {target_type}",
        value=f"{prediction:,.2f} W",
        delta=None
    )
    st.info(f"**Model Used:** {model_name} | **Plant:** {plant_num} | **Inverter:** {source_key}")

    st.metric(
        label="☀️ Irradiation",
        value=f"{irradiation:.2f} kW/m²"
    )
    st.metric(
        label="🌡️ Ambient Temp",
        value=f"{amb_temp:.1f} °C"
    )
       
    st.metric(
        label="📟 Module Temp",
        value=f"{mod_temp:.1f} °C"
    )
    
    st.metric(
        label="🕐 Time",
        value=f"{hour:02d}:00"
    )

with right:
    st.markdown("#### 📈 Power Prediction vs. Irradiation")
        
        # Generate sensitivity analysis data
    irr_range = np.linspace(0.0, 1.5, 50)
    sim_preds = []

    for irr in irr_values:
        temp_input = preprocess_input(irr, amb_temp, mod_temp, hour, source_key)
        pred = make_prediction(model, temp_input)
        predictions.append(pred)

    # Create interactive plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=irr_values,
        y=predictions,
        mode='lines',
        name='Predicted Power',
        line=dict(color='blue', width=3),
        hovertemplate='Irradiation: %{x:.2f} kW/m²<br>Power: %{y:.2f} W<extra></extra>'
    ))

    # Add current point
    fig.add_trace(go.Scatter(
        x=[irradiation],
        y=[prediction],
        mode='markers',
        name='Current Setting',
        marker=dict(size=15, color='red', symbol='star'),
        hovertemplate=f'<b>Current</b><br>Irradiation: {irradiation:.2f} kW/m²<br>Power: {prediction:.2f} W<extra></extra>'
    ))

    fig.update_layout(
        xaxis_title="Solar Irradiation (kW/m²)",
        yaxis_title=f"Predicted {target_type} (W)",
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    st.plotly_chart(fig, use_container_width=True)
# === Additional Info ===
with st.expander("ℹ️ How to Use This Dashboard"):
    st.markdown(f"""
    ### Quick Guide
    
    **Step 1:** Select your plant (Plant 1 or Plant 2) and power type (DC or AC)
    
    **Step 2:** Choose the inverter ID from the dropdown
    
    **Step 3:** Adjust the environmental parameters:
    - **Solar Irradiation:** Amount of solar energy hitting the panels
    - **Ambient Temperature:** Outside air temperature
    - **Module Temperature:** Solar panel surface temperature
    - **Hour of Day:** Time of day (5 AM to 7 PM)
    
    **Step 4:** View the prediction and see how power generation changes with irradiation
    
    ### Available Models
    - **Plant 1:** Multi-Layer Perceptron (MLP) Neural Network
    - **Plant 2:** Lasso Regression
    
    ### Notes
    - Predictions cannot be negative (minimum value is 0 W)
    - The chart shows how power generation varies with irradiation while keeping other parameters constant
    - Red star indicates your current parameter settings
    """)

