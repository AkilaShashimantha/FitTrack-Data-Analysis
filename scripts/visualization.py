import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Optional


def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Creates a modern, interactive correlation heatmap using Plotly.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        A Plotly Figure object.
    """
    # Ensure only numeric columns are used for correlation
    numeric_df = df.select_dtypes(include=['number'])
    corr = numeric_df.corr(numeric_only=True)

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='Viridis',
        zmin=-1,
        zmax=1,
        hoverongaps=False
    ))
    fig.update_layout(
        title='Correlation Matrix of Fitness Metrics',
        xaxis_nticks=len(corr.columns),
        yaxis_nticks=len(corr.columns),
    )
    return fig


def plot_steps_vs_calories_scatter(df: pd.DataFrame, user_point: Optional[Dict[str, float]] = None):
    """
    Scatter plot of Total Steps vs. Calories with OLS trendline and optional user point overlay.

    Args:
        df: dataset
        user_point: optional dict with keys {'TotalSteps', 'Calories'} to highlight user's scenario
    """
    fig = px.scatter(
        df,
        x='TotalSteps',
        y='Calories',
        trendline='ols',  # requires statsmodels, installed in cloud
        color='Calories',
        color_continuous_scale='Viridis',
        title='Total Steps vs. Calories Burned',
        labels={'TotalSteps': 'Total Daily Steps', 'Calories': 'Calories Burned'}
    )
    if user_point is not None:
        fig.add_trace(go.Scatter(
            x=[user_point.get('TotalSteps', None)],
            y=[user_point.get('Calories', None)],
            mode='markers',
            marker=dict(size=14, color='red', symbol='star'),
            name='Your point'
        ))
    fig.update_layout(
        xaxis_title='Total Steps',
        yaxis_title='Calories Burned'
    )
    return fig


def plot_sleep_vs_activity_scatter(df: pd.DataFrame, user_point: Optional[Dict[str, float]] = None):
    """
    Scatter plot of Sleep Duration vs. Very Active Minutes with optional user overlay.
    """
    fig = px.scatter(
        df,
        x='VeryActiveMinutes',
        y='TotalMinutesAsleep',
        color='TotalMinutesAsleep',
        color_continuous_scale='Cividis_r',
        title='Sleep Duration vs. Very Active Minutes',
        labels={'VeryActiveMinutes': 'Very Active Minutes', 'TotalMinutesAsleep': 'Minutes Asleep'}
    )
    if user_point is not None:
        fig.add_trace(go.Scatter(
            x=[user_point.get('VeryActiveMinutes', None)],
            y=[user_point.get('TotalMinutesAsleep', None)],
            mode='markers',
            marker=dict(size=14, color='orange', symbol='diamond'),
            name='Your point'
        ))
    fig.update_layout(
        xaxis_title='Very Active Minutes per Day',
        yaxis_title='Total Minutes Asleep'
    )
    return fig


def plot_activity_composition_donut(segments: Dict[str, float]):
    """Donut chart for activity composition (Very/Fairly/Lightly Active vs Sedentary)."""
    labels = list(segments.keys())
    values = list(segments.values())
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5)])
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(title_text='Daily Time Composition', showlegend=True)
    return fig


def plot_sleep_duration_gauge(minutes_asleep: float, goal_minutes: int = 480):
    """Gauge showing sleep duration vs goal (default 8h)."""
    pct = max(0, min(100, 100.0 * minutes_asleep / goal_minutes if goal_minutes else 0))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={'suffix': '%'},
        title={'text': 'Sleep Goal Attainment'},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': '#42A5F5'},
            'steps': [
                {'range': [0, 60], 'color': '#512DA8'},
                {'range': [60, 85], 'color': '#1565C0'},
                {'range': [85, 100], 'color': '#1E88E5'},
            ],
        }
    ))
    return fig


def plot_sensitivity_bars(impacts: Dict[str, float]):
    """Horizontal bar chart for +/- impact (delta kcal) per feature."""
    features = list(impacts.keys())
    deltas = [impacts[k] for k in features]
    colors = ['#2E7D32' if d >= 0 else '#C62828' for d in deltas]
    fig = go.Figure(go.Bar(
        x=deltas,
        y=features,
        orientation='h',
        marker_color=colors,
        hovertemplate='%{y}: %{x:.0f} kcal<extra></extra>'
    ))
    fig.update_layout(
        title='Sensitivity: +10% change effect on predicted calories',
        xaxis_title='Δ Calories (kcal)',
        yaxis_title='Feature',
    )
    return fig

