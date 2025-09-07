import streamlit as st
import pandas as pd
import os
import joblib
import numpy as np
import time
from scripts.visualization import (
    plot_correlation_heatmap,
    plot_steps_vs_calories_scatter,
    plot_sleep_vs_activity_scatter
)

# --- Page Configuration ---
st.set_page_config(
    page_title="FitTrack AI Dashboard",
    page_icon="🏃",
    layout="wide"
)

# --- Custom CSS for Modern UI ---
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    /* Card-like containers */
    .st-emotion-cache-z5fcl4 {
        border-radius: 15px;
        padding: 25px;
        background-color: #0E1117; /* Dark background for cards */
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
    }
    
    .st-emotion-cache-z5fcl4:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    
    /* Custom button style */
    .stButton>button {
        color: #FFFFFF;
        background-color: #1E88E5; /* A nice blue */
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-size: 16px;
        font-weight: bold;
        transition-duration: 0.4s;
        cursor: pointer;
    }

    .stButton>button:hover {
        background-color: #1565C0; /* Darker blue on hover */
        color: white;
    }

    /* Header styling */
    h1, h2, h3 {
        color: #FFFFFF; /* White text for headers */
    }

    /* Metric styling for prediction */
    .st-emotion-cache-ocqkz7 {
        background-color: #1A202C;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
    }
    
    .st-emotion-cache-ocqkz7 p {
        font-size: 1.2rem; /* Metric label font size */
    }

    .st-emotion-cache-ocqkz7 div[data-testid="stMetricValue"] {
        font-size: 3rem; /* Metric value font size */
        color: #42A5F5; /* Highlight color for the metric value */
    }

</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---
@st.cache_data
def load_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_resource
def load_model_and_scaler(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# --- Header ---
st.title("FitTrack AI Dashboard 🏃‍♂️✨")
st.markdown("Your personal AI-powered fitness and sleep analysis hub.")

# --- Load Data and Models ---

data_path = os.path.join('data', 'cleaned_fitness_data.csv')
model_path = 'best_model.pkl'
scaler_path = 'scaler.pkl'

@st.cache_data

def get_dataset():
    """Load cleaned dataset if present; otherwise build it from raw daily files."""
    df_clean = load_data(data_path)
    if df_clean is not None:
        return df_clean

    # Fallback: construct cleaned dataset on the fly
    raw_activity = os.path.join('data', 'dailyActivity_merged.csv')
    raw_sleep = os.path.join('data', 'sleepDay_merged.csv')
    if not (os.path.exists(raw_activity) and os.path.exists(raw_sleep)):
        return None

    try:
        act = pd.read_csv(raw_activity)
        slp = pd.read_csv(raw_sleep)
        # Parse dates
        act['ActivityDate'] = pd.to_datetime(act['ActivityDate'], format='%m/%d/%Y', errors='coerce')
        slp['SleepDay'] = pd.to_datetime(slp['SleepDay'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
        slp['ActivityDate'] = pd.to_datetime(slp['SleepDay'].dt.date)
        # Merge
        merged = pd.merge(act, slp, on=['Id', 'ActivityDate'], how='inner')
        # Drop extras if present
        drop_cols = [c for c in ['TrackerDistance', 'LoggedActivitiesDistance', 'SleepDay'] if c in merged.columns]
        if drop_cols:
            merged = merged.drop(columns=drop_cols)
        merged = merged.drop_duplicates()
        return merged
    except Exception:
        return None


df = get_dataset()

if df is None:
    st.error("🚨 Data not found. Ensure raw CSVs exist in the data/ folder or run the notebook to generate cleaned data.")
    st.info("Expected files: data/dailyActivity_merged.csv and data/sleepDay_merged.csv.")
else:
    # --- Prediction Section ---
    st.header("🔮 Calorie Burn Predictor")
    with st.container():
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            st.warning("Model not found. Please run the notebook to train and save the model.")
        else:
            model, scaler = load_model_and_scaler(model_path, scaler_path)
            
            col1, col2 = st.columns([2, 1]) # Input sliders on the left, prediction on the right
            
            with col1:
                st.subheader("Enter Your Daily Metrics:")
                c1, c2 = st.columns(2)
                with c1:
                    total_steps = st.slider("Total Steps", 0, 25000, 8000, 100)
                    total_distance = st.slider("Total Distance (km)", 0.0, 20.0, 6.0, 0.5)
                    very_active_minutes = st.slider("Very Active Minutes", 0, 200, 25)
                    fairly_active_minutes = st.slider("Fairly Active Minutes", 0, 200, 20)
                with c2:
                    lightly_active_minutes = st.slider("Lightly Active Minutes", 0, 500, 200)
                    sedentary_minutes = st.slider("Sedentary Minutes", 0, 1440, 700)
                    total_minutes_asleep = st.slider("Total Minutes Asleep", 0, 720, 420, 10)
            
            with col2:
                st.subheader("AI Prediction")
                if st.button("Predict Calories", use_container_width=True):
                    with st.spinner('Calculating...'):
                        time.sleep(1) # Simulate calculation time
                        input_data = np.array([[
                            total_steps, total_distance, very_active_minutes,
                            fairly_active_minutes, lightly_active_minutes, sedentary_minutes,
                            total_minutes_asleep
                        ]])
                        input_data_scaled = scaler.transform(input_data)
                        prediction = model.predict(input_data_scaled)
                        st.metric(label="Predicted Calories Burned", value=f"{int(prediction[0])} kcal")
                else:
                    st.info("Click the button to see the prediction.")

    st.markdown("---") # Divider

    # --- Exploratory Data Analysis Section ---
    st.header("📊 Exploratory Data Analysis")
    with st.container():
        st.subheader("Correlation Heatmap")
        st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Steps vs. Calories")
            st.plotly_chart(plot_steps_vs_calories_scatter(df), use_container_width=True)
        with col2:
            st.subheader("Sleep vs. Activity")
            st.plotly_chart(plot_sleep_vs_activity_scatter(df), use_container_width=True)

