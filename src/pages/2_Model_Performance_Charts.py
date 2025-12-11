# src/app_rmse_analysis.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Configure page
st.set_page_config(
    page_title="Model RMSE Analysis",
    page_icon="📊",
    layout="wide"
)

# === Load RMSE Data ===
@st.cache_data
def load_rmse_data():
    """Load and process RMSE data from CSV"""
    csv_path = os.path.join("models", "RMSE_Plant.csv")
    
    if not os.path.exists(csv_path):
        return None
    
    # Read the CSV with multi-level header
    df = pd.read_csv(csv_path, header=[0, 1])
    
    # Reset column names for easier access
    df.columns = ['Model', 'Plant1_DC', 'Plant1_AC', 'Plant2_DC', 'Plant2_AC']
    
    return df

# === Main Dashboard ===
st.title("📊 Model Performance Comparison")
st.markdown("### Compare Root Mean Square Error (RMSE) across different models")

st.divider()

# Load data
df_rmse = load_rmse_data()

if df_rmse is None:
    st.error("⚠️ RMSE_Plant.csv not found in models folder")
    st.stop()

# === Configuration ===
st.subheader("🔧 Select Configuration")

col1, col2 = st.columns(2)

with col1:
    plant_select = st.radio(
        "🏭 Select Plant",
        ["Plant 1", "Plant 2"],
        horizontal=True
    )

with col2:
    power_select = st.radio(
        "⚡ Select Power Type",
        ["DC", "AC"],
        horizontal=True
    )

st.divider()

# === Get data for selected configuration ===
plant_num = "1" if plant_select == "Plant 1" else "2"
column_name = f"Plant{plant_num}_{power_select}"

# Prepare data for plotting
plot_df = pd.DataFrame({
    'Model': df_rmse['Model'],
    'RMSE': df_rmse[column_name]
})

# Find best and worst models
best_idx = plot_df['RMSE'].idxmin()
worst_idx = plot_df['RMSE'].idxmax()
best_model = plot_df.loc[best_idx, 'Model']
worst_model = plot_df.loc[worst_idx, 'Model']
best_rmse = plot_df.loc[best_idx, 'RMSE']
worst_rmse = plot_df.loc[worst_idx, 'RMSE']

# === Display Key Metrics ===
st.subheader(f"📈 {plant_select} - {power_select} Power RMSE Results")

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric(
        label="🏆 Best Model",
        value=best_model,
        delta=f"RMSE: {best_rmse:.1f} W"
    )

with metric_col2:
    st.metric(
        label="📉 Best RMSE",
        value=f"{best_rmse:.1f} W",
        # delta="Lowest error"
    )

with metric_col3:
    st.metric(
        label="📊 Average RMSE",
        value=f"{plot_df['RMSE'].mean():.1f} W",
        delta=f"Range: {best_rmse:.1f} - {worst_rmse:.1f}"
    )

st.divider()

# === Bar Chart ===
st.subheader("📊 RMSE Comparison - Bar Chart")

# Create color array (highlight best model)
colors = ['#FF4B4B' if model == best_model else '#0068C9' for model in plot_df['Model']]

fig_bar = go.Figure(data=[
    go.Bar(
        x=plot_df['Model'],
        y=plot_df['RMSE'],
        marker_color=colors,
        text=plot_df['RMSE'].round(1),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>RMSE: %{y:.1f} W<extra></extra>'
    )
])

fig_bar.update_layout(
    title=f"RMSE by Model - {plant_select} ({power_select} Power)",
    xaxis_title="Model Type",
    yaxis_title="RMSE",
    height=500,
    showlegend=False,
    hovermode='x'
)

fig_bar.update_xaxes(title_font=dict(size=20), tickfont=dict(size=18))
fig_bar.update_yaxes(title_font=dict(size=20), tickfont=dict(size=18))


st.plotly_chart(fig_bar, use_container_width=True)
st.divider()
