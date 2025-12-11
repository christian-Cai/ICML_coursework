# src/app_lstm_predictions.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import numpy as np

# Configure page
st.set_page_config(
    page_title="LSTM Prediction Analysis",
    page_icon="🤖",
    layout="wide"
)

# === Helper Functions ===

@st.cache_data
def load_prediction_data(plant_num):
    """Load LSTM prediction data for a specific plant"""
    if plant_num == 1:
        csv_path = os.path.join("models", "Plant 1_1d_FS_V2_predictions.csv")
    else:
        csv_path = os.path.join("models", "Plant 2_1d_FS_V2_predictions.csv")
    
    if not os.path.exists(csv_path):
        return None
    
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df

def calculate_metrics(true_values, predicted_values):
    """Calculate prediction metrics"""
    mae = np.mean(np.abs(true_values - predicted_values))
    rmse = np.sqrt(np.mean((true_values - predicted_values)**2))
    mape = np.mean(np.abs((true_values - predicted_values) / (true_values + 1e-10))) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }

# === Main Dashboard ===

st.title("🤖 LSTM Model Prediction ")
st.markdown("### Compare LSTM predictions with observed power generation values")

st.divider()


st.subheader("Power Predictions")

# === Plant Selection ===
plant_option = st.radio(
    "🏭 Select Plant to View",
    ["Plant 1", "Plant 2", "Both Plants"],
    horizontal=True,
    key="ac_plant"
)

st.divider()

# === Load and Display Data ===

if plant_option in ["Plant 1", "Both Plants"]:
    st.markdown("### 📊 Plant 1 - Power")
    
    df_p1 = load_prediction_data(1)
    
    if df_p1 is not None:
        # Aggregate by timestamp (average across all inverters)
        df_p1_agg = df_p1.groupby('timestamp').agg({
            'AC_POWER_true': 'mean',
            'AC_POWER_LSTM': 'mean',
            'AC_POWER_persistence': 'mean',
            'AC_POWER_MA_1h': 'mean'
        }).reset_index()
        
        # Calculate metrics
        metrics_p1 = calculate_metrics(
            df_p1_agg['AC_POWER_true'].values,
            df_p1_agg['AC_POWER_LSTM'].values
        )
        
        col1, col2 = st.columns(2)            
        with col1:
            st.metric("📉 RMSE", f"{metrics_p1['RMSE']:.2f} W")
        with col2:
            st.metric("📏 MAE", f"{metrics_p1['MAE']:.2f} W")

        # Create comparison chart
        fig_p1 = go.Figure()
        
        # Sample data for better performance (show every 10th point if too large)
        sample_rate = max(1, len(df_p1_agg) // 1000)
        df_plot = df_p1_agg.iloc[::sample_rate]
        
        fig_p1.add_trace(go.Scatter(
            x=df_plot['timestamp'],
            y=df_plot['AC_POWER_true'],
            mode='lines',
            name='Observed Value',
            line=dict(color='#0068C9', width=2),
            hovertemplate='Time: %{x}<br>True: %{y:.2f} W<extra></extra>'
        ))
        
        fig_p1.add_trace(go.Scatter(
            x=df_plot['timestamp'],
            y=df_plot['AC_POWER_LSTM'],
            mode='lines',
            name='LSTM Prediction',
            line=dict(color='#FF4B4B', width=2, dash='dash'),
            hovertemplate='Time: %{x}<br>LSTM: %{y:.2f} W<extra></extra>'
        ))
        
        fig_p1.update_layout(
            title="Plant 1: Observed vs LSTM Predicted ",
            xaxis_title="Timestamp",
            yaxis_title="AC Power (W)",
            height=700,
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig_p1, use_container_width=True)
        
        # Scatter plot for correlation
        st.markdown("#### 🎯 Prediction Accuracy Scatter Plot")
        
        # Sample for scatter plot
        sample_scatter = min(5000, len(df_p1_agg))
        df_scatter = df_p1_agg.sample(n=sample_scatter)
        
        fig_scatter_p1 = px.scatter(
            df_scatter,
            x='AC_POWER_true',
            y='AC_POWER_LSTM',
            opacity=0.5,
            title="Plant 1: Predicted vs Observed  ",
            labels={'AC_POWER_true': 'Observed AC Power (W)', 'AC_POWER_LSTM': 'LSTM Predicted (W)'}
        )
        
        # Add perfect prediction line
        max_val = max(df_scatter['AC_POWER_true'].max(), df_scatter['AC_POWER_LSTM'].max())
        fig_scatter_p1.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig_scatter_p1.update_layout(height=700,width=700)
        st.plotly_chart(fig_scatter_p1, width='content')

    else:
        st.error("⚠️ Plant 1 prediction data not found")

if plant_option in ["Plant 2", "Both Plants"]:
    if plant_option == "Both Plants":
        st.divider()
    
    st.markdown("### 📊 Plant 2 - Power")
    
    df_p2 = load_prediction_data(2)
    
    if df_p2 is not None:
        # Aggregate by timestamp
        df_p2_agg = df_p2.groupby('timestamp').agg({
            'AC_POWER_true': 'mean',
            'AC_POWER_LSTM': 'mean',
            'AC_POWER_persistence': 'mean',
            'AC_POWER_MA_1h': 'mean'
        }).reset_index()
        
        # Calculate metrics
        metrics_p2 = calculate_metrics(
            df_p2_agg['AC_POWER_true'].values,
            df_p2_agg['AC_POWER_LSTM'].values
        )
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📉 RMSE", f"{metrics_p2['RMSE']:.2f} W")
        with col2:
            st.metric("📏 MAE", f"{metrics_p2['MAE']:.2f} W")

        
        # Create comparison chart
        fig_p2 = go.Figure()
        
        # Sample data
        sample_rate = max(1, len(df_p2_agg) // 1000)
        df_plot = df_p2_agg.iloc[::sample_rate]
        
        fig_p2.add_trace(go.Scatter(
            x=df_plot['timestamp'],
            y=df_plot['AC_POWER_true'],
            mode='lines',
            name='Observed Value',
            line=dict(color='#0068C9', width=2),
            hovertemplate='Time: %{x}<br>True: %{y:.2f} W<extra></extra>'
        ))
        
        fig_p2.add_trace(go.Scatter(
            x=df_plot['timestamp'],
            y=df_plot['AC_POWER_LSTM'],
            mode='lines',
            name='LSTM Prediction',
            line=dict(color='#FF4B4B', width=2, dash='dash'),
            hovertemplate='Time: %{x}<br>LSTM: %{y:.2f} W<extra></extra>'
        ))
        
        fig_p2.update_layout(
            title="Plant 2: Observed vs LSTM Predicted Power",
            xaxis_title="Timestamp",
            yaxis_title="AC Power (W)",
            height=700,
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig_p2, use_container_width=True)
        
        # Scatter plot
        st.markdown("#### 🎯 Prediction Accuracy Scatter Plot")
        
        sample_scatter = min(5000, len(df_p2_agg))
        df_scatter = df_p2_agg.sample(n=sample_scatter)
        
        fig_scatter_p2 = px.scatter(
            df_scatter,
            x='AC_POWER_true',
            y='AC_POWER_LSTM',
            opacity=0.5,
            title="Plant 2: Predicted vs Observed Power",
            labels={'AC_POWER_true': 'Observed AC Power (W)', 'AC_POWER_LSTM': 'LSTM Predicted (W)'}
        )
        
        max_val = max(df_scatter['AC_POWER_true'].max(), df_scatter['AC_POWER_LSTM'].max())
        fig_scatter_p2.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        

        fig_scatter_p2.update_layout(height=700,width=700)
        st.plotly_chart(fig_scatter_p2, width='content')
        
    else:
        st.error("⚠️ Plant 2 prediction data not found")
