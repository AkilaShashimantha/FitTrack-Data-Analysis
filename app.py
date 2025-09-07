import streamlit as st
import pandas as pd
import os
import joblib
import numpy as np
import time
import io

# Optional import: provide graceful fallback if not installed yet
try:
    from fpdf import FPDF
except Exception:
    FPDF = None

from scripts.visualization import (
    plot_correlation_heatmap,
    plot_steps_vs_calories_scatter,
    plot_sleep_vs_activity_scatter,
    plot_activity_composition_donut,
    plot_sleep_duration_gauge,
    plot_sensitivity_bars,
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

# Sidebar navigation
section = st.sidebar.radio(
    "Navigate",
    ["Overview", "What-if Analysis", "Report", "Exploration", "About"],
    index=0,
)

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
    # Load model if available
    model = scaler = None
    has_model = os.path.exists(model_path) and os.path.exists(scaler_path)
    if has_model:
        model, scaler = load_model_and_scaler(model_path, scaler_path)

    # Overview
    if section == "Overview":
        st.header("✨ Overview")
        # High-level KPIs from dataset
        n_days = len(df)
        n_users = df['Id'].nunique() if 'Id' in df.columns else None
        avg_steps = int(df['TotalSteps'].mean()) if 'TotalSteps' in df.columns else None
        avg_cal = int(df['Calories'].mean()) if 'Calories' in df.columns else None
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Days", f"{n_days}")
        with c2:
            st.metric("Users", f"{n_users}")
        with c3:
            st.metric("Avg Steps", f"{avg_steps}")
        with c4:
            st.metric("Avg Calories", f"{avg_cal} kcal")

        st.markdown("---")
        st.subheader("Data Relationships")
        st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)

    # What-if Analysis
    elif section == "What-if Analysis":
        st.header("🔮 What-if Analysis & Personalized Insights")
        if not has_model:
            st.warning("Model not found. Please run the training notebook to generate best_model.pkl and scaler.pkl.")
        else:
            if 'last_pred' not in st.session_state:
                st.session_state['last_pred'] = None

            left, right = st.columns([2, 1])
            with left:
                st.subheader("Adjust Your Daily Metrics")
                a, b = st.columns(2)
                with a:
                    total_steps = st.slider("Total Steps", 0, 30000, 8000, 100)
                    total_distance = st.slider("Total Distance (km)", 0.0, 30.0, 6.0, 0.1)
                    very_active_minutes = st.slider("Very Active Minutes", 0, 300, 25)
                    fairly_active_minutes = st.slider("Fairly Active Minutes", 0, 300, 20)
                with b:
                    lightly_active_minutes = st.slider("Lightly Active Minutes", 0, 700, 200)
                    sedentary_minutes = st.slider("Sedentary Minutes", 0, 1440, 700)
                    total_minutes_asleep = st.slider("Total Minutes Asleep", 0, 720, 420, 10)

                # Compose visuals from inputs
                st.subheader("Your Day at a Glance")
                comp = {
                    'Very Active': very_active_minutes,
                    'Fairly Active': fairly_active_minutes,
                    'Lightly Active': lightly_active_minutes,
                    'Sedentary': sedentary_minutes,
                }
                donut, gauge = st.columns(2)
                with donut:
                    st.plotly_chart(plot_activity_composition_donut(comp), use_container_width=True)
                with gauge:
                    st.plotly_chart(plot_sleep_duration_gauge(total_minutes_asleep), use_container_width=True)

            with right:
                st.subheader("AI Prediction")
                x = np.array([
                    [
                        total_steps, total_distance, very_active_minutes,
                        fairly_active_minutes, lightly_active_minutes, sedentary_minutes,
                        total_minutes_asleep
                    ]
                ])
                x_scaled = scaler.transform(x)
                pred = float(model.predict(x_scaled)[0])
                delta = None if st.session_state['last_pred'] is None else pred - st.session_state['last_pred']
                st.metric("Predicted Calories", f"{int(pred)} kcal", delta=(None if delta is None else f"{int(delta)} kcal"))

                # Sensitivity: +10% change on each feature
                features = [
                    'TotalSteps', 'TotalDistance', 'VeryActiveMinutes',
                    'FairlyActiveMinutes', 'LightlyActiveMinutes', 'SedentaryMinutes',
                    'TotalMinutesAsleep'
                ]
                impacts = {}
                for i, name in enumerate(features):
                    x_pert = x.copy()
                    base = x_pert[0, i]
                    # Handle zero baseline gracefully
                    x_pert[0, i] = base * 1.1 if base != 0 else 1.0
                    pred_pert = float(model.predict(scaler.transform(x_pert))[0])
                    impacts[name] = pred_pert - pred

                st.plotly_chart(plot_sensitivity_bars(impacts), use_container_width=True)

                # Persist current scenario for Report section
                st.session_state['whatif_inputs'] = {
                    'TotalSteps': total_steps,
                    'TotalDistance': total_distance,
                    'VeryActiveMinutes': very_active_minutes,
                    'FairlyActiveMinutes': fairly_active_minutes,
                    'LightlyActiveMinutes': lightly_active_minutes,
                    'SedentaryMinutes': sedentary_minutes,
                    'TotalMinutesAsleep': total_minutes_asleep,
                }
                st.session_state['prediction'] = pred
                st.session_state['impacts'] = impacts

                # Plain-language insights
                st.subheader("Personalized Insight")
                total_minutes = max(1, very_active_minutes + fairly_active_minutes + lightly_active_minutes + sedentary_minutes)
                sedentary_ratio = sedentary_minutes / total_minutes
                if sedentary_ratio > 0.6:
                    st.info("High sedentary time detected. Small increases in Lightly/Fairly Active minutes can notably improve energy expenditure.")
                elif very_active_minutes < 20:
                    st.info("Consider adding short bouts of high-intensity activity; model suggests Very Active minutes have strong positive impact.")
                else:
                    st.success("Great balance of activity. Maintain consistency to sustain calorie burn.")

                # Update delta baseline at end of render
                st.session_state['last_pred'] = pred

        # Overlay your point on relationships using predicted Calories
        st.markdown("---")
        st.subheader("Where You Sit in the Data")
        user_point_steps = {'TotalSteps': total_steps, 'Calories': pred} if has_model else None
        user_point_sleep = {'VeryActiveMinutes': very_active_minutes, 'TotalMinutesAsleep': total_minutes_asleep}
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_steps_vs_calories_scatter(df, user_point=user_point_steps), use_container_width=True)
        with c2:
            st.plotly_chart(plot_sleep_vs_activity_scatter(df, user_point=user_point_sleep), use_container_width=True)

    # Report
    elif section == "Report":
        st.header("📝 Personalized Report")
        inputs = st.session_state.get('whatif_inputs')
        pred = st.session_state.get('prediction')
        impacts = st.session_state.get('impacts')
        if not inputs or pred is None:
            st.info("Use the 'What-if Analysis' tab to enter your metrics and generate a prediction first.")
        else:
            # Compose visuals
            comp = {
                'Very Active': inputs['VeryActiveMinutes'],
                'Fairly Active': inputs['FairlyActiveMinutes'],
                'Lightly Active': inputs['LightlyActiveMinutes'],
                'Sedentary': inputs['SedentaryMinutes'],
            }
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_activity_composition_donut(comp), use_container_width=True)
            with c2:
                st.plotly_chart(plot_sleep_duration_gauge(inputs['TotalMinutesAsleep']), use_container_width=True)

            # Build friendly markdown report
            def fmt_minutes(m):
                h = int(m // 60)
                mm = int(m % 60)
                return f"{h}h {mm}m"

            avg_steps = int(df['TotalSteps'].mean()) if 'TotalSteps' in df.columns else None
            steps_cmp = f"(dataset avg ~{avg_steps:,})" if avg_steps else ""

            top_impacts = sorted(impacts.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
            top_lines = "\n".join([f"- {k}: {v:+.0f} kcal for +10% change" for k, v in top_impacts])

            report_md = f"""
# FitTrack Personalized Report

## Summary
- Predicted calories: {int(pred)} kcal
- Steps: {int(inputs['TotalSteps']):,} {steps_cmp}
- Distance: {inputs['TotalDistance']:.1f} km
- Activity: Very {int(inputs['VeryActiveMinutes'])}m, Fairly {int(inputs['FairlyActiveMinutes'])}m, Lightly {int(inputs['LightlyActiveMinutes'])}m, Sedentary {int(inputs['SedentaryMinutes'])}m
- Sleep: {fmt_minutes(inputs['TotalMinutesAsleep'])} (goal 8h)

## What shapes your result most
{top_lines}

## Recommendations
- Increase Very Active minutes in small bouts (e.g., +10 min); this typically has a strong positive impact.
- If sedentary time is high, insert light activity breaks each hour to reduce sedentary minutes.
- Aim for consistent sleep (7–8 hours) to support recovery and daily energy expenditure.

Generated with your current inputs in the What‑if Analysis.
"""
            st.markdown(report_md)
            st.download_button(
                label="Download report (.md)",
                data=report_md,
                file_name="fittrack_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

            # --- PDF generation with embedded images ---
            def build_pdf_report(_inputs, _pred, _impacts, _df):
                # Ensure FPDF is available (lazy import to avoid requiring an app restart)
                global FPDF
                if FPDF is None:
                    try:
                        from fpdf import FPDF as _FPDF
                        FPDF = _FPDF
                    except Exception as e:
                        raise ImportError("fpdf2 is not installed") from e

                # Prepare figures
                comp_local = {
                    'Very Active': _inputs['VeryActiveMinutes'],
                    'Fairly Active': _inputs['FairlyActiveMinutes'],
                    'Lightly Active': _inputs['LightlyActiveMinutes'],
                    'Sedentary': _inputs['SedentaryMinutes'],
                }
                donut_fig_local = plot_activity_composition_donut(comp_local)
                gauge_fig_local = plot_sleep_duration_gauge(_inputs['TotalMinutesAsleep'])

                # Export figures to PNG bytes using Kaleido (best-effort; continue without images on failure)
                donut_png = gauge_png = None
                try:
                    donut_png = donut_fig_local.to_image(format='png', scale=2)
                    gauge_png = gauge_fig_local.to_image(format='png', scale=2)
                except Exception:
                    # Fallback to plotly.io API explicitly using kaleido engine
                    try:
                        import plotly.io as pio
                        donut_png = pio.to_image(donut_fig_local, format='png', scale=2, engine='kaleido')
                        gauge_png = pio.to_image(gauge_fig_local, format='png', scale=2, engine='kaleido')
                    except Exception:
                        # As a final fallback, proceed without images
                        donut_png = gauge_png = None

                # Compose PDF
                pdf = FPDF(orientation='P', unit='mm', format='A4')
                # Consistent margins to ensure enough text width
                pdf.set_margins(left=12, top=15, right=12)
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.add_page()

                # Title
                pdf.set_font('Helvetica', 'B', 16)
                pdf.set_x(pdf.l_margin)
                pdf.cell(pdf.w - pdf.l_margin - pdf.r_margin, 10, 'FitTrack Personalized Report', ln=True)
                pdf.ln(2)

                # Body
                pdf.set_font('Helvetica', '', 12)
                content_w = pdf.w - pdf.l_margin - pdf.r_margin
                # Summary text
                avg_steps_local = int(_df['TotalSteps'].mean()) if 'TotalSteps' in _df.columns else None
                steps_cmp_local = f"(dataset avg ~{avg_steps_local:,})" if avg_steps_local else ""
                lines = [
                    f"Predicted calories: {int(_pred)} kcal",
                    f"Steps: {int(_inputs['TotalSteps']):,} {steps_cmp_local}",
                    f"Distance: {_inputs['TotalDistance']:.1f} km",
                    (
                        "Activity: Very "
                        f"{int(_inputs['VeryActiveMinutes'])}m, Fairly {int(_inputs['FairlyActiveMinutes'])}m, "
                        f"Lightly {int(_inputs['LightlyActiveMinutes'])}m, Sedentary {int(_inputs['SedentaryMinutes'])}m"
                    ),
                    f"Sleep: {fmt_minutes(_inputs['TotalMinutesAsleep'])} (goal 8h)",
                    "",
                    "Top drivers (for +10% change):",
                ]
                top_imp_local = sorted(_impacts.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
                for k, v in top_imp_local:
                    lines.append(f"- {k}: {v:+.0f} kcal")

                # Ensure safe text encoding and enough width
                for ln in lines:
                    safe = ln.encode('latin-1', 'replace').decode('latin-1')
                    pdf.set_x(pdf.l_margin)
                    pdf.multi_cell(content_w, 8, safe)

                pdf.ln(4)
                # Add images (stacked) within content width
                try:
                    if donut_png:
                        pdf.image(io.BytesIO(donut_png), w=content_w, type='PNG')
                        pdf.ln(4)
                    if gauge_png:
                        pdf.image(io.BytesIO(gauge_png), w=content_w, type='PNG')
                except Exception:
                    # If embedding fails, continue without images
                    pass

                out = pdf.output(dest='S')
                # Normalize to bytes for Streamlit download_button
                if isinstance(out, str):
                    data = out.encode('latin-1', 'ignore')
                elif isinstance(out, bytearray):
                    data = bytes(out)
                else:
                    data = out  # already bytes
                return data

            # Show a single-click PDF download button
            try:
                pdf_bytes = build_pdf_report(inputs, pred, impacts, df)
                st.download_button(
                    label="Download report (.pdf)",
                    data=pdf_bytes,
                    file_name="fittrack_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.info(
                    "PDF download requires extra packages. Install them and rerun:\n\n"
                    "pip install fpdf2 kaleido\n\n"
                    f"Details: {e}"
                )

    # Exploration
    elif section == "Exploration":
        st.header("📊 Exploratory Data Analysis")
        st.subheader("Correlation Heatmap")
        st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Steps vs. Calories")
            st.plotly_chart(plot_steps_vs_calories_scatter(df), use_container_width=True)
        with col2:
            st.subheader("Sleep vs. Activity")
            st.plotly_chart(plot_sleep_vs_activity_scatter(df), use_container_width=True)

    # About
    else:
        st.header("ℹ️ About")
        st.markdown(
            "This app predicts daily calorie burn from activity and sleep inputs, and provides what-if insights.\n\n"
            "Data comes from daily activity and sleep summaries. Models are trained offline and loaded here for fast inference."
        )
