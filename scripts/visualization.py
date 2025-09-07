import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

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
    corr = numeric_df.corr()
    
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

def plot_steps_vs_calories_scatter(df: pd.DataFrame):
    """
    Creates an interactive scatter plot of Total Steps vs. Calories with a trendline.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        A Plotly Figure object.
    """
    fig = px.scatter(
        df,
        x='TotalSteps',
        y='Calories',
        trendline='ols', # Ordinary Least Squares regression line
        color='Calories',
        color_continuous_scale='Viridis',
        title='Total Steps vs. Calories Burned',
        labels={'TotalSteps': 'Total Daily Steps', 'Calories': 'Calories Burned'}
    )
    fig.update_layout(
        xaxis_title='Total Steps',
        yaxis_title='Calories Burned'
    )
    return fig

def plot_sleep_vs_activity_scatter(df: pd.DataFrame):
    """
    Creates an interactive scatter plot of Sleep Duration vs. Very Active Minutes.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        A Plotly Figure object.
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
    fig.update_layout(
        xaxis_title='Very Active Minutes per Day',
        yaxis_title='Total Minutes Asleep'
    )
    return fig

