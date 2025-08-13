import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import html, dcc
from constants import TASKS
from snr.constants import get_pretty_task_name
from snr.constants.plot import MODEL_FAMILY_COLORS


def load_signal_noise_data():
    from snr.download.hf import pull_predictions_from_hf

    local_path = pull_predictions_from_hf("allenai/signal-and-noise", "core")
    df = pd.read_parquet(local_path)

    print(f"Loaded {len(df):,} model evaluations")
    return df


def prepare_compute_performance_data(df):
    """Prepare data for compute vs performance plotting."""
    if df is None:
        return None

    # Filter for rows with valid FLOPs data
    valid_data = df[df["flops"].notna()].copy()

    if len(valid_data) == 0:
        print("No valid FLOPs data found")
        return None

    # Group by task and model to get average performance
    grouped = (
        valid_data.groupby(["task", "model"])
        .agg(
            {
                "flops": "first",  # FLOPs should be the same for a given model
                "primary_score": "mean",
                "primary_metric": "first",
                "task_category": "first",
                "model_params": "first",
                "model_type": "first",
                "model_path": "first",
                "model_revision": "first",
            }
        )
        .reset_index()
    )

    # Extract model family from model_path
    grouped["model_family"] = grouped["model_path"].str.split("/").str[0]

    # Remove any remaining NaN values
    grouped = grouped.dropna(subset=["flops", "primary_score"])

    print(f"Prepared {len(grouped):,} data points for plotting")
    return grouped


def create_task_specific_plot(df_plot, task_name):
    if df_plot is None or len(df_plot) == 0:
        return go.Figure()

    # Filter data for this specific task
    task_data = df_plot[df_plot["task"] == task_name].copy()

    if len(task_data) == 0:
        return go.Figure()

    fig = go.Figure()

    # Get unique model families and assign colors
    unique_families = task_data["model_family"].unique()
    
    # Use plotly's default color palette
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    
    # Create a trace for each model family
    for i, family in enumerate(unique_families):
        family_data = task_data[task_data["model_family"] == family]
        
        fig.add_trace(
            go.Scatter(
                x=family_data["flops"],
                y=family_data["primary_score"],
                mode="markers",
                name=family,
                marker=dict(
                    size=8,
                    color=MODEL_FAMILY_COLORS.get(family, colors[i % len(colors)]),
                    opacity=0.7,
                    line=dict(width=0.5, color="white"),
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    + "FLOPs: %{x:.2e}<br>"
                    + "Score: %{y:.3f}<br>"
                    + "Model Type: %{customdata[1]}<br>"
                    + "Parameters: %{customdata[2]:.1e}<br>"
                    + "Revision: %{customdata[3]}"
                    + "<extra></extra>"
                ),
                customdata=np.column_stack(
                    (family_data["model_path"], family_data["model_type"], 
                     family_data["model_params"], family_data["model_revision"])
                ),
            )
        )

    display_name = get_pretty_task_name(task_name)

    fig.update_layout(
        title=dict(
            text=f"{display_name}", font=dict(size=18, family="Manrope"), x=0.5, xanchor="center"
        ),
        xaxis=dict(
            title="Compute (FLOPs)",
            tickfont=dict(size=12),
            type="log",
            showgrid=True,
            gridwidth=1,
            gridcolor="#f0f0f0",
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
        ),
        yaxis=dict(
            title=task_data["primary_metric"].iloc[0] if len(task_data) > 0 else "Primary Metric",
            tickfont=dict(size=12),
            showgrid=True,
            gridwidth=1,
            gridcolor="#f0f0f0",
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
        ),
        font=dict(family="Manrope", size=14),
        plot_bgcolor="white",
        paper_bgcolor="#faf2e9",
        height=400,
        hovermode="closest",
        margin=dict(t=50, r=20, b=50, l=60),
    )

    return fig


def setup_compute_performance_app(app):
    # Load and prepare data
    df_raw = load_signal_noise_data()
    df_plot = prepare_compute_performance_data(df_raw)

    if df_plot is None:
        return html.Div(
            [
                html.H2("Data Loading Error", style={"fontFamily": "Manrope"}),
                html.P(
                    "Could not load the signal-and-noise dataset.", style={"fontFamily": "Manrope"}
                ),
            ]
        )

    all_tasks = TASKS

    # Create individual plots for each task
    task_plots = []
    for i, task in enumerate(all_tasks):
        fig = create_task_specific_plot(df_plot, task)
        task_plots.append(
            html.Div(
                [dcc.Graph(id=f"task-plot-{i}", figure=fig)],
                style={"width": "50%", "display": "inline-block", "padding": "10px"},
            )
        )

    # Arrange plots in a grid
    N_COLS = 2
    plot_grid = []
    for i in range(0, len(task_plots), N_COLS):
        row_plots = task_plots[i : i + N_COLS]
        plot_grid.append(html.Div(row_plots, style={"display": "flex", "width": "100%"}))

    layout = html.Div(
        [
            html.Div(plot_grid),
        ]
    )

    return layout
