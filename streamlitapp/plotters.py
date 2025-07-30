# ────────────────────────────────────────────────────────────────────────────────────
# This is one of the most important pages of the app, every single plot is here
# The functions are named relative to the name of the file they are used into (e.g. plot_plotting_data_1 is used in pages/plotting_data.py)
# ────────────────────────────────────────────────────────────────────────────────────
import streamlit as st
import plotly.graph_objects as go
from utils import MODEL_LIST, EXPLAINER_LIST
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import numpy as np

COLOR9 = sns.color_palette("colorblind", 9) 
COLOR9 = list(map(lambda rgb: f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.8)", COLOR9))
COLOR17 = sns.color_palette("colorblind", 17) # This is a color palette from seaborn, it is used to color the bars in the plots
COLOR17 = list(map(lambda rgb: f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.8)", COLOR17)) # Convert RGB to RGBA with 0.8 opacity for Plotly compatibility

SHAPES = ["triangle-up", "square", "diamond", "cross", "circle", "triangle-down", "star", "hexagon", 
          "pentagon", "hourglass", "bowtie", "triangle-left", "triangle-right", "x", "octagon", 
          "star-diamond", "asterisk"]  # Shapes for scatter plots
GOLDMEDAL = "🏅" # The strings used for the medals given across the plots
SILVERMEDAL = "🥈"
BRONZEMEDAL = "🥉"
WHICHISBETTER = { # Every single metric has a different way of being read, this correspondance table allows to know which is the best value for each metric
    "Deletion": "Lower is better",
    "MuFidelity": "Higher is better",
    "Sparseness": "Higher is better",
    "Complexity": "Lower is better",
    "Insertion": "Higher is better"
}
ARROWUP = "↑"
ARROWDOWN = "↓"
ARROWCORRESPONDANCE = { # Once we know which is the best value for each metric, we can use this correspondance table to display the arrows in the plots
    "Higher is better": ARROWUP,
    "Lower is better": ARROWDOWN
}

# ────────────────────────────────────────────────────────────
# Plots scores for each model organized by (explainer, metric)
# ────────────────────────────────────────────────────────────
def plot_plotting_data_1(df, layout):
    """Create grouped bar chart organized by (explainer, metric) pairs.

    Generates an interactive bar chart where models are grouped by explainer-metric
    combinations. Shows average scores with medal indicators for top 3 performers
    in each group. Includes directional arrows indicating whether higher or lower
    scores are better for each metric.

    Args:
        df (pd.DataFrame): DataFrame containing score data with columns:
            - 'model': Model names
            - 'explainer': Explainer method names  
            - 'metric': Evaluation metric names
            - 'score': Numeric performance scores
        layout: Streamlit layout object for displaying the chart.

    Returns:
        None: Displays the chart directly in the Streamlit layout.

    Example:
        >>> df = pd.DataFrame({'model': ['ResNet'], 'explainer': ['Lime'], 'metric': ['Deletion'], 'score': [0.85]})
        >>> plot_plotting_data_1(df, st)
        >>> # Displays grouped bar chart with medal rankings
    """
    
    df = df.copy() # The df.copy is not really useful in our case because we work on query results but it is still best for debugging to keep it
    df['score'] = df['score'].apply(lambda x: abs(float(x)) if isinstance(x, str) else x) # Convert scores to float

    unique_pairs = df[['explainer', 'metric']].drop_duplicates().values.tolist() # Get unique pairs of (explainer, metric) to group data
    unique_models = df['model'].unique().tolist() # Get unique models to plot data for each model
    
    model_colors = {MODEL_LIST[i]: COLOR9[i] for i in range(len(MODEL_LIST))} # Assign colors to models for the bar chart

    fig = go.Figure() 

    # For each (explainer, metric), we group all the data associated with it as a database, allowing for better processing of the data
    grouped_data = {} 
    for explainer, metric in unique_pairs:
        key = f"{explainer} - {metric}(abs) {ARROWCORRESPONDANCE[WHICHISBETTER[metric]]}" if metric == "MuFidelity" else f"{explainer} - {metric} {ARROWCORRESPONDANCE[WHICHISBETTER[metric]]}" # These are the names of the columns
        grouped_data[key] = df[(df['explainer'] == explainer) & (df['metric'] == metric)]

    # Calculate mean scores for each model-explainer-metric combination
    mean_grouped_data = {}
    for key, group_df in grouped_data.items():
        if not group_df.empty:
            # Group by model and calculate mean score
            model_means = group_df.groupby('model')['score'].mean().reset_index()
            mean_grouped_data[key] = model_means.to_dict('records')
        else:
            mean_grouped_data[key] = []
    
    # Sort and assign medals
    for key in mean_grouped_data:
        if mean_grouped_data[key]:
            mean_grouped_data[key].sort(key=lambda x: x['score'], reverse=key.endswith(ARROWUP))

    # We assign medals to the top 3 models for each (explainer, metric) group, once they are sorted
    for key in mean_grouped_data:
        for row in range(len(mean_grouped_data[key])):
            mean_grouped_data[key][row]['medal'] = GOLDMEDAL if row == 0 else SILVERMEDAL if row == 1 else BRONZEMEDAL if row == 2 else ""

    # Add bars for each model
    for model in unique_models:
        x_values = []
        y_values = []
        medal_values = []
        for group, rows in mean_grouped_data.items():
            # Filter rows for the current model
            filtered_rows = [row for row in rows if row['model'] == model]
            if filtered_rows:
                x_values.append(group)  # Group name (explainer-metric)
                y_values.append(filtered_rows[0]['score'])  # Average score
                medal_values.append(filtered_rows[0]['medal'])  # Medal emoji

        # Add a bar trace for the current model
        fig.add_trace(go.Bar(
            x=x_values,
            y=y_values,
            name=model,
            marker=dict(color=model_colors[model]),
            cliponaxis=False,
            text=medal_values,  # Medal emoji
            textposition='outside',
            textfont=dict(size=20)
        ))

    fig.add_annotation( # Add annotation for arrow meanings, on the top left of the plot
        text=f"{ARROWUP} Higher is better<br>{ARROWDOWN} Lower is better",
        xref="paper", yref="paper",
        x=1.02, y=1, 
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
        font=dict(size=12)
    )

    # Update layout
    fig.update_layout(
        title='Bar Chart Organized by (Explainer, Metric)',
        xaxis_title='Explainer - Metric',
        yaxis_title='Average Score',
        barmode='group',  # Group bars by x-axis categories
        showlegend=True,
    )

    # The plotly plot is then displayed in the Streamlit layout
    layout.plotly_chart(fig, use_container_width=True, key="plot_plotting_data_1") 
# ──────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────
# Plots scores for each explainer organized by (model, metric)
# ────────────────────────────────────────────────────────────
def plot_plotting_data_2(df, layout):
    """Create grouped bar chart organized by (model, metric) pairs.

    Generates an interactive bar chart where explainers are grouped by model-metric
    combinations. Shows average scores with medal indicators for top 3 performers
    in each group. Includes directional arrows indicating whether higher or lower
    scores are better for each metric.

    Args:
        df (pd.DataFrame): DataFrame containing score data with columns:
            - 'model': Model names
            - 'explainer': Explainer method names  
            - 'metric': Evaluation metric names
            - 'score': Numeric performance scores
        layout: Streamlit layout object for displaying the chart.

    Returns:
        None: Displays the chart directly in the Streamlit layout.

    Example:
        >>> df = pd.DataFrame({'model': ['ResNet'], 'explainer': ['Lime'], 'metric': ['Deletion'], 'score': [0.85]})
        >>> plot_plotting_data_2(df, st)
        >>> # Displays grouped bar chart with explainers grouped by model-metric pairs
    """
    
    df = df.copy() # The df.copy is not really useful in our case because we work on query results but it is still best for debugging to keep it
    df['score'] = df['score'].apply(lambda x: float(x) if isinstance(x, str) else x) # Convert scores to float

    unique_pairs = df[['model', 'metric']].drop_duplicates().values.tolist() # Get unique pairs of (model, metric) to group data
    unique_explainers = df['explainer'].unique().tolist() # Get unique explainers to plot data for each explainer
    
    explainer_colors = {EXPLAINER_LIST[i]: COLOR17[i] for i in range(len(EXPLAINER_LIST))} # Assign colors to explainers for the bar chart

    fig = go.Figure()

    # For each (model, metric), we group all the data associated with it as a database, allowing for better processing of the data
    grouped_data = {}
    for model, metric in unique_pairs:
        key = f"{model} - {metric}(abs) {ARROWCORRESPONDANCE[WHICHISBETTER[metric]]}" if metric == "MuFidelity" else f"{model} - {metric} {ARROWCORRESPONDANCE[WHICHISBETTER[metric]]}" # These are the names of the columns
        grouped_data[key] = df[(df['model'] == model) & (df['metric'] == metric)]

    # Calculate mean scores for each explainer-model-metric combination
    mean_grouped_data = {}
    for key, group_df in grouped_data.items():
        if not group_df.empty:
            # Group by explainer and calculate mean score
            explainer_means = group_df.groupby('explainer')['score'].mean().reset_index()
            mean_grouped_data[key] = explainer_means.to_dict('records')
        else:
            mean_grouped_data[key] = []
    
    # Sort and assign medals
    for key in mean_grouped_data:
        if mean_grouped_data[key]:
            mean_grouped_data[key].sort(key=lambda x: x['score'], reverse=key.endswith(ARROWUP))

    # We assign medals to the top 3 explainers for each (model, metric) group, once they are sorted
    for key in mean_grouped_data:
        for row in range(len(mean_grouped_data[key])):
            mean_grouped_data[key][row]['medal'] = GOLDMEDAL if row == 0 else SILVERMEDAL if row == 1 else BRONZEMEDAL if row == 2 else ""

    # Add bars for each explainer
    for explainer in unique_explainers:
        x_values = []
        y_values = []
        medal_values = []
        for group, rows in mean_grouped_data.items():
            # Filter rows for the current explainer
            filtered_rows = [row for row in rows if row['explainer'] == explainer]
            if filtered_rows:
                x_values.append(group)  # Group name (model-metric)
                y_values.append(filtered_rows[0]['score'])  # Average score
                medal_values.append(filtered_rows[0]['medal'])  # Medal emoji

        # Add a bar trace for the current explainer
        if x_values and y_values:  # Only add if we have data for this explainer
            color = explainer_colors.get(explainer, '#1f77b4')  # Fallback color if explainer not in color mapping
            fig.add_trace(go.Bar(
                x=x_values,
                y=y_values,
                name=explainer,
                marker=dict(color=color),
                cliponaxis=False,
                text=medal_values,  # Medal emoji
                textposition='outside',
                textfont=dict(size=20)
            ))

    fig.add_annotation( # Add annotation for arrow meanings, on the top right of the plot
        text=f"{ARROWUP} Higher is better<br>{ARROWDOWN} Lower is better",
        xref="paper", yref="paper",
        x=1.02, y=1, 
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
        font=dict(size=12)
    )

    # Update layout
    fig.update_layout(
        title='Bar Chart Organized by (Model, Metric)',
        xaxis_title='Model - Metric',
        yaxis_title='Average Score',
        barmode='group',  # Group bars by x-axis categories
        showlegend=True,
        margin=dict(r=150)  # Add right margin for annotation
    )

    layout.plotly_chart(fig, use_container_width=True, key="plot_plotting_data_2") # The plotly plot is then displayed in the Streamlit layout
# ──────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────
# Plots mean scores by standard deviation for each model, explainer
# ──────────────────────────────────────────────────────────────────
def plot_mean_scatter_map(df, layout, plot_key):
    """Create scatter plot showing mean scores vs standard deviation.

    Generates an interactive scatter plot where each point represents a model-explainer
    combination. Uses dual encoding: shapes represent models, colors represent explainers.
    Shows the relationship between mean performance and variability.

    Args:
        df (pd.DataFrame): DataFrame containing aggregated data with columns:
            - 'model': Model names
            - 'explainer': Explainer method names
            - 'mean': Mean performance scores
            - 'std_dev': Standard deviation of scores
        layout: Streamlit layout object for displaying the chart.
        plot_key (str): Unique key for the Streamlit plotly chart component.

    Returns:
        None: Displays the scatter plot directly in the Streamlit layout.

    Example:
        >>> df = pd.DataFrame({'model': ['ResNet'], 'explainer': ['Lime'], 'mean': [0.85], 'std_dev': [0.12]})
        >>> plot_mean_scatter_map(df, st, "scatter_1")
        >>> # Displays scatter plot with dual shape/color encoding
    """

    df = df.copy() # The df.copy is not really useful in our case because we work on query results but it is still best for debugging to keep it
    
    unique_models = df['model'].unique().tolist() # Get unique models to plot data for each model
    unique_explainers = df['explainer'].unique().tolist() # Get unique explainers to plot data for each explainer

    model_colors = {MODEL_LIST[i]: COLOR9[i] for i in range(len(MODEL_LIST))} # Assign shapes to models for scatter plot differentiation
    explainer_shapes = {EXPLAINER_LIST[i]: SHAPES[i] for i in range(len(EXPLAINER_LIST))} # Assign colors to explainers for scatter plot differentiation

    fig = go.Figure()

    # Add legend entries for explainers (colors) - invisible points to show legend only
    for explainer in unique_explainers:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], # No actual data points, just for legend
            mode='markers',
            marker=dict(
                symbol=explainer_shapes.get(explainer, 'circle'), # Use explainer shape or fallback to circle
                color='gray',
                size=10,
            ),
            name=f"{explainer}",
            showlegend=True,
            legendgroup='explainer' # Group explainer legends together
        ))

    # Add legend entries for models (shapes) - invisible points to show legend only
    for model in unique_models:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], # No actual data points, just for legend
            mode='markers',
            marker=dict(
                symbol='circle',
                color=model_colors.get(model, 'gray'), # Gray color for shape legend consistency
                size=10,
            ),
            name=f"{model}",
            showlegend=True,
            legendgroup='model' # Group model legends together
        ))

    # Add actual data points with both shape and color encoding
    for _, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['mean']], # Mean score on x-axis
            y=[row['std_dev']], # Standard deviation on y-axis
            mode='markers',
            marker=dict(
                symbol=explainer_shapes.get(row['explainer'], 'circle'), # Shape represents model
                color=model_colors.get(row['model'], '#1f77b4'), # Color represents explainer
                size=10,
            ),
            text=f"{row['model']} - {row['explainer']}, {row['mean']} ± {row['std_dev']}", # Hover text with detailed information
            name=f"{row['model']} - {row['explainer']}",
            hoverinfo="text",
            showlegend=False # Don't show individual data points in legend to avoid clutter
        ))

    # Create legend text for models with color coding
    model_legend_text = "Model (Color):<br>" + "<br>".join([
        f"<span style='color:{model_colors.get(model, '#1f77b4')}'>{model}</span>"
        for model in unique_models
    ])
    
    # Create legend text for explainers with shape information
    explainer_legend_text = "Explainer (Shape):<br>" + "<br>".join([
        f"{explainer}: {explainer_shapes.get(model, 'circle')}"
        for explainer in unique_explainers
    ])

    # Update layout with titles and axis configuration
    fig.update_layout(
        title='Mean Scores by Standard Deviation',
        xaxis_title=f'Mean Score ({WHICHISBETTER[df["metric"].iloc[0]]})',
        yaxis_title='Standard Deviation',
        xaxis=dict(type='linear'), # Linear scale for mean scores
        yaxis=dict(type='linear'), # Linear scale for standard deviation
        showlegend=True, # Show the legend with color and shape mappings
    )
    
    layout.plotly_chart(fig, use_container_width=True, key=plot_key) # The plotly plot is then displayed in the Streamlit layout
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────────────────
# Plots a leaderboard of explainers based on their mean scores for a selected metric, model
# ─────────────────────────────────────────────────────────────────────────────────────────
def plot_leaderboard(df, layout, dark_mode):
    """Create styled leaderboard table with medal rankings.

    Generates a formatted table showing explainer rankings for a specific metric.
    Includes medal emojis for top 3 performers and colored backgrounds. Sorts
    results based on whether higher or lower scores are better for the metric.

    Args:
        df (pd.DataFrame): DataFrame containing performance data with columns:
            - 'explainer': Explainer method names
            - 'metric': Evaluation metric name (assumes all rows have same metric)
            - 'mean': Mean performance scores
            - 'std_dev': Standard deviation of scores
        layout: Streamlit layout object for displaying the table.
        dark_mode (bool): Whether to use dark color scheme for styling.

    Returns:
        None: Displays the styled leaderboard table directly in the Streamlit layout.

    Example:
        >>> df = pd.DataFrame({'explainer': ['Lime', 'SHAP'], 'metric': ['Deletion'], 'mean': [0.8, 0.9], 'std_dev': [0.1, 0.05]})
        >>> plot_leaderboard(df, st, False)
        >>> # Displays ranked table with medals and colored backgrounds
    """
    # Convert DataFrame to work with existing logic
    df = df.copy() # The df.copy is not really useful in our case because we work on query results but it is still best for debugging to keep it
    
    # Get the metric (assuming all rows have the same metric)
    metric = df['metric'].iloc[0] # Extract metric name to determine sorting direction
    
    # Since we already have mean and std columns, just rename them
    leaderboard_df = df[['explainer', 'mean', 'std_dev']].copy() # Select relevant columns for leaderboard
    leaderboard_df.columns = ['explainer', 'mean', 'std_dev'] # Ensure consistent column naming
    
    # Sort based on metric preference (higher or lower is better)
    leaderboard_df = leaderboard_df.sort_values(
        'mean', 
        ascending=(WHICHISBETTER[metric] == "Lower is better") # Sort ascending if lower scores are better, descending if higher scores are better
    ).reset_index(drop=True)
    
    # Add medals and create final DataFrame
    medals = []
    for i in range(len(leaderboard_df)):
        if i == 0:
            medals.append(GOLDMEDAL) # First place gets gold medal
        elif i == 1:
            medals.append(SILVERMEDAL) # Second place gets silver medal
        elif i == 2:
            medals.append(BRONZEMEDAL) # Third place gets bronze medal
        else:
            medals.append("") # No medal for other positions
    
    # Create final display DataFrame with formatted scores
    final_df = pd.DataFrame({
        'Rank': medals, # Medal emojis for visual ranking
        'Explainer': leaderboard_df['explainer'], # Explainer names
        'Score': [f"{mean:.4f} ± {std:.4f}" for mean, std in zip(leaderboard_df['mean'], leaderboard_df['std_dev'])] # Formatted mean ± std scores
    })
    
    # Apply background colors to top 3 rows based on dark mode setting
    def highlight_top_rows(row):
        if dark_mode:
            if row.name == 0:  # Dark Gold for first place
                return ['background-color: #B8860B'] * len(row)
            elif row.name == 1:  # Dark Silver for second place
                return ['background-color: #708090'] * len(row)
            elif row.name == 2:  # Dark Bronze for third place
                return ['background-color: #8B4513'] * len(row)
        else:
            if row.name == 0:  # Light Gold for first place
                return ['background-color: #FFD700'] * len(row)
            elif row.name == 1:  # Light Silver for second place
                return ['background-color: #C0C0C0'] * len(row)
            elif row.name == 2:  # Light Bronze for third place
                return ['background-color: #CD7F32'] * len(row)
        return [''] * len(row) # No styling for other rows
    
    styled_df = final_df.style.apply(highlight_top_rows, axis=1) # Apply row-wise styling function
    
    # Display the styled leaderboard table
    layout.dataframe(
        styled_df, 
        hide_index=True, # Don't show row indices
        use_container_width=True, # Use full container width
        column_config={
            "Rank": st.column_config.TextColumn(width="small"), # Small width for medal column
            "Explainer": st.column_config.TextColumn(width="medium"), # Medium width for explainer names
            "Score": st.column_config.TextColumn(width="medium") # Medium width for score display
        }
    )
# ────────────────────────────────────────────────────────────────────────────────────────────

# ───────────────────────────────────────────────────────────────
# Plots training time vs model performance for a given metric
# ───────────────────────────────────────────────────────────────
def plot_training_time(df_fidelity, df_wb, layout, plot_key):
    """Create multi-subplot scatter plot showing training time vs performance by GPU.

    Combines fidelity and timing data to show the relationship between training time
    and explainer performance for a single model and metric. Creates separate subplots for each 
    GPU type with color encoding for different explainers.

    Args:
        df_fidelity (pd.DataFrame): DataFrame containing fidelity scores with columns:
            - 'model': Single model name (all rows should have same model)
            - 'explainer': Explainer method names
            - 'mean': Mean performance scores
            - 'std_dev': Standard deviation of scores
        df_wb (pd.DataFrame): DataFrame containing timing data with columns:
            - 'model': Single model name (should match fidelity data)
            - 'explainer': Explainer method names (renamed to 'explainer')
            - 'gpu_type': GPU type identifiers
            - 'mean_time_per_sample': Average training time per sample
            - 'std_time_per_sample': Standard deviation of training times
        layout: Streamlit layout object for displaying the chart.

    Returns:
        None: Displays the multi-subplot scatter plot directly in the Streamlit layout.

    Example:
        >>> df_fid = pd.DataFrame({'model': ['ResNet'], 'explainer': ['Lime'], 'mean': [0.85], 'std_dev': [0.1]})
        >>> df_time = pd.DataFrame({'model': ['ResNet'], 'explainer': ['Lime'], 'gpu_type': ['L40S'], 'mean_time_per_sample': [2.5], 'std_time_per_sample': [0.2]})
        >>> plot_training_time(df_fid, df_time, st)
        >>> # Displays scatter plot with time vs performance by GPU for single model
    """
    
    df_fidelity = df_fidelity.copy()
    df_wb = df_wb.copy()
    
    # Get the minimum time for each explainer-gpu combination
    df_wb_min = df_wb.groupby(['explainer', 'gpu_type'])['mean_time_per_sample'].min().reset_index()
    df_wb_min['std_time_per_sample'] = df_wb.groupby(['explainer', 'gpu_type'])['std_time_per_sample'].min().reset_index()['std_time_per_sample']

    
    # Merge the dataframes on explainer only
    merged_df = pd.merge(
        df_fidelity, 
        df_wb_min, 
        on=['explainer'], 
        how='inner'
    )
    
    # Get unique values for plotting
    model_name = merged_df['model'].iloc[0] if not merged_df.empty else "Unknown Model"
    unique_explainers = merged_df['explainer'].unique().tolist()
    unique_gpus = merged_df['gpu_type'].unique().tolist()
    
    # Simplified color assignment
    explainer_colors = {EXPLAINER_LIST[i]: COLOR17[i] for i in range(len(EXPLAINER_LIST))}
    
    # Create subplots for each GPU type
    fig = make_subplots(
        rows=1, 
        cols=len(unique_gpus),
        subplot_titles=[f"{gpu} GPU" for gpu in unique_gpus],
        shared_yaxes=True,
        shared_xaxes=True,
        horizontal_spacing=0.1
    )
    
    # Add traces for each GPU type
    for i, gpu in enumerate(unique_gpus, 1):
        gpu_data = merged_df[merged_df['gpu_type'] == gpu]
        
        for explainer in unique_explainers:
            explainer_data = gpu_data[gpu_data['explainer'] == explainer]
            
            if not explainer_data.empty:
                # Extract scalar values from the data
                time_value = explainer_data['mean_time_per_sample'].iloc[0]
                time_std = explainer_data['std_time_per_sample'].iloc[0]
                perf_value = explainer_data['mean'].iloc[0]
                perf_std = explainer_data['std_dev'].iloc[0]
                
                fig.add_trace(
                    go.Scatter(
                        x=explainer_data['mean_time_per_sample'],
                        y=explainer_data['mean'],
                        mode='markers',
                        marker=dict(
                            color=explainer_colors.get(explainer, '#000000'),
                            size=10,
                            symbol='circle' 
                        ),
                        name=f"{explainer}",
                        legendgroup=explainer,
                        showlegend=(i == 1),  # Only show legend for first subplot
                        hovertemplate=(
                            f"<b>{explainer}</b><br>"
                            f"Model: {model_name}<br>"
                            f"GPU: {gpu}<br>"
                            f"Time per sample: {time_value:.3f} ± {time_std:.3f} s<br>"
                            f"Performance: {perf_value:.3f} ± {perf_std:.3f}<br>"
                            "<extra></extra>"
                        )
                    ),
                    row=1, col=i
                )
    
    # Update layout
    fig.update_layout(
        title=f"Training Time vs Performance Analysis - {model_name}",
        height=400,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1
        ),
        annotations=[
        # Annotation between subplots (centered)
            dict(
                text="⬆ Higher Performance<br>is Better" if WHICHISBETTER[df_fidelity['metric'].iloc[0]] == "Higher is better" else "⬇ Lower Performance<br>is Better",
                x=0.5,  # Center between subplots
                y=0.5,  # Middle of the plot area
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=10),
                bgcolor="rgba(255, 255, 255, 0.1)",  # Semi-transparent white background
                bordercolor="gray",
                borderwidth=1,
                xanchor="center",
                yanchor="middle"
            )
        ]
    )
    
    # Update axes labels
    for i in range(1, len(unique_gpus) + 1):
        fig.update_xaxes(title_text="Time per Sample (seconds)", row=1, col=i)
    fig.update_yaxes(title_text="Performance Score", row=1, col=1)
    
    layout.plotly_chart(fig, use_container_width=True, key=plot_key) # Display the plotly chart in the Streamlit layout
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
# Plots batch inference time for each model and GPU type
# ──────────────────────────────────────────────────────────────
def plot_batch_time(df, layout, plot_key):
    """Create line plots showing inference time vs batch size by GPU for single model.

    Generates matplotlib line plots with error bands showing how inference time per
    sample varies with batch size. Creates separate subplots for each GPU type.
    Shows performance scaling characteristics for a single model.

    Args:
        df (pd.DataFrame): DataFrame containing timing data with columns:
            - 'model': Single model name (all rows should have same model)
            - 'explainer': Explainer method names
            - 'gpu_type': GPU type identifiers
            - 'batch_size': Batch sizes used for inference
            - 'mean_time_per_sample': Average inference time per sample
            - 'std_time_per_sample': Standard deviation of inference times
        layout: Streamlit layout object for displaying the charts.

    Returns:
        None: Displays the line plots directly in the Streamlit layout.

    Example:
        >>> df = pd.DataFrame({'model': ['ResNet'], 'explainer': ['Lime'], 'gpu_type': ['V100'], 
                              'batch_size': [1, 8, 16], 'mean_time_per_sample': [2.0, 1.5, 1.2], 
                              'std_time_per_sample': [0.1, 0.08, 0.05]})
        >>> plot_batch_time(df, st)
        >>> # Displays line plot showing batch size scaling for single model
    """
    # Data preprocessing and cleaning
    df_clean = df.dropna(subset=['mean_time_per_sample', 'std_time_per_sample'])

    unique_gpus = df_clean['gpu_type'].unique()
    unique_explainers = df_clean['explainer'].unique()
    model_name = df_clean['model'].iloc[0] if not df_clean.empty else "Unknown Model"
    batch_sizes = sorted(df_clean["batch_size"].unique())

    # Convert rgba colors to hex format for matplotlib compatibility
    explainer_colors = {EXPLAINER_LIST[i]: COLOR17[i] for i in range(len(EXPLAINER_LIST))}


    fig = make_subplots(
        rows=1, 
        cols=len(unique_gpus),
        subplot_titles=[f"{gpu} GPU" for gpu in unique_gpus],
        shared_yaxes=True,
        horizontal_spacing=0.1
    )

    # Create subplot for single model (one row, multiple GPUs)
    for i, gpu in enumerate(unique_gpus, 1):
        gpu_data = df_clean[df_clean['gpu_type'] == gpu]
        
        for explainer in unique_explainers:
            explainer_data = gpu_data[gpu_data['explainer'] == explainer]
            
            if not explainer_data.empty:
                # Sort by batch_size for proper line drawing
                explainer_data_sorted = explainer_data.sort_values('batch_size')
                
                # Get data
                x_vals = explainer_data_sorted['batch_size']
                y_vals = explainer_data_sorted['mean_time_per_sample']
                y_error = explainer_data_sorted['std_time_per_sample']
                
                # Calculate upper and lower bounds
                y_upper = y_vals + y_error
                y_lower = y_vals - y_error
                
                # Get color for this explainer
                color = explainer_colors.get(explainer, '#000000')
                # Convert rgba to rgb for fill
                fill_color = color.replace("0.8)", "0.3)")
                
                # Add filled area (error band)
                fig.add_trace(
                    go.Scatter(
                        x=list(x_vals) + list(x_vals[::-1]),  # x, then x reversed
                        y=list(y_upper) + list(y_lower[::-1]),  # upper, then lower reversed
                        fill='toself',
                        fillcolor=fill_color,
                        line=dict(color='rgba(255,255,255,0)'),  # Transparent line
                        hoverinfo="skip",
                        showlegend=False,
                        name=f"{explainer}_fill"
                    ),
                    row=1, col=i
                )
                
                # Add main line with markers
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode='lines+markers',
                        line=dict(color=color, width=2),
                        marker=dict(
                            color=color,
                            size=8,
                            symbol='circle'
                        ),
                        name=f"{explainer}",
                        legendgroup=explainer,
                        showlegend=(i == 1),  # Only show legend for first subplot
                        hovertemplate=(
                            f"<b>{explainer}</b><br>"
                            f"GPU: {gpu}<br>"
                            f"Batch Size: %{{x}}<br>"
                            f"Mean Time: %{{y:.4f}}s<br>"
                            f"Std Dev: {y_error.iloc[0]:.4f}s<br>"
                            "<extra></extra>"
                        )
                    ),
                    row=1, col=i
                )
    
    # Update layout
    batch_sizes = sorted(df_clean["batch_size"].unique())

    for i in range(len(unique_gpus)):
        col_num = i + 1
        
        # Update x-axis for each subplot
        fig.update_xaxes(
            type="log",  # Use base 2 for batch sizes (1, 2, 4, 8, 16, 32, etc.)
            tickvals=batch_sizes,
            ticktext=[str(bs) for bs in batch_sizes],
            title_text="Batch Size (log scale)" if i == len(unique_gpus)//2 else "",
            row=1, 
            col=col_num
        )
        
        # Ensure y-axis is consistent
        fig.update_yaxes(
            title_text="Mean Time per Sample (s)" if i == 0 else "",
            row=1, 
            col=col_num
        )

    # Update overall layout
    fig.update_layout(
        title=dict(
            text=f"Batch Size Performance Scaling - {model_name}",
            x=0.5
        ),
        height=400,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    # Update axes labels
    for i in range(1, len(unique_gpus) + 1):
        fig.update_xaxes(title_text="Time per Sample (seconds)", row=1, col=i)
    fig.update_yaxes(title_text="Performance Score", row=1, col=1)
    
    layout.plotly_chart(fig, use_container_width=True, key=plot_key) # Display the plotly chart in the Streamlit layout
# ─────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
# Plots rank agreement between models based on Spearman correlation
# ──────────────────────────────────────────────────────────────
def plot_rank_agreement(df, layout, plot_key, dark_mode):
    df = df.copy()

    models = df['model'].unique()
    ranking_matrix = df.pivot_table(index='explainer', columns='model', values='rank')

    # Reorder columns and index according to MODEL_LIST
    present_models = [model for model in MODEL_LIST if model in models]
    ranking_matrix = ranking_matrix.reindex(columns=present_models, index=ranking_matrix.index)

    model_correlation = ranking_matrix.corr(method='spearman')

    fig,ax = plt.subplots(figsize=(12, 8))

    if dark_mode:
        fig.patch.set_facecolor("dark") # Set figure background to black
        ax.set_facecolor("dark") # Set axes background to black

    heatmap = sns.heatmap(model_correlation, annot=True, cmap='YlOrRd', ax=ax, center=0.5, vmin=0, vmax=1, square=True, fmt='.2f')
    ax.set_title('Spearman Rank Correlation Between Models', fontsize=16, color='white' if dark_mode else 'black')
    ax.set_xlabel('Models', fontsize=14, color='white' if dark_mode else 'black')
    ax.set_ylabel('Models', fontsize=14, color='white' if dark_mode else 'black')
    ax.tick_params(colors='white' if dark_mode else 'black')

    if dark_mode:
        cbar = heatmap.collections[0].colorbar # Get colorbar reference
        cbar.ax.tick_params(colors='white') # Set colorbar tick colors to white
        cbar.ax.yaxis.label.set_color('white') # Set colorbar label color to white

    plt.xticks(rotation=45, ha='right', fontsize=12, color='white' if dark_mode else 'black')
    plt.yticks(rotation=0)

    plt.tight_layout()
    layout.pyplot(fig)
    plt.close()
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────
# Just the dialog in which the data is displayed
# ──────────────────────────────────────────────
@st.dialog("Plotted Data", width='large')
def dialog_plotted_data(data,plot):
    plot(data, st)

# ─────────────────────────────────────────────────────────────────────────────────────────
# Plots correlation matrix of explainers' rankings across models and metrics for subsamples
# ─────────────────────────────────────────────────────────────────────────────────────────
def plot_ranking_correlation_matrix(df_full, df_subsample, layout):
    """
    Create a correlation matrix showing how well explainer rankings correlate 
    between full data and subsample across different model-metric combinations.
    
    Args:
        df_full: DataFrame with full dataset rankings
        df_subsample: DataFrame with subsample rankings  
        layout: Streamlit layout object
    This function calculates the correlation of rankings for each explainer across
    model-metric combinations, comparing full data to subsample.
    Returns:
        None: Displays the heatmap directly in the Streamlit layout.
    """
    
    # Get unique values
    models = sorted(df_full['model'].unique())
    metrics = sorted(df_full['metric'].unique()) 
    explainers = sorted(df_full['explainer'].unique())
    
    # Create correlation matrix - explainers vs model-metric combinations
    correlation_data = []
    model_metric_combinations = []
    
    for model in models:
        for metric in metrics:
            model_metric_combinations.append(f"{model}")
            explainer_correlations = []
            
            for explainer in explainers:
                # Get rankings for this explainer in this model-metric combination
                full_data = df_full[
                    (df_full['model'] == model) & 
                    (df_full['metric'] == metric) & 
                    (df_full['explainer'] == explainer)
                ]
                
                sub_data = df_subsample[
                    (df_subsample['model'] == model) & 
                    (df_subsample['metric'] == metric) & 
                    (df_subsample['explainer'] == explainer)
                ]
                
                if not full_data.empty and not sub_data.empty:
                    # For single values, correlation doesn't make sense, so we'll use rank difference
                    full_rank = full_data['rank'].iloc[0]
                    sub_rank = sub_data['rank'].iloc[0]
                    
                    # Convert rank difference to correlation-like score (1 = perfect, 0 = worst)
                    max_possible_diff = len(explainers) - 1  # Max rank difference
                    rank_diff = abs(full_rank - sub_rank)
                    correlation_score = 1 - (rank_diff / max_possible_diff) if max_possible_diff > 0 else 1
                    explainer_correlations.append(correlation_score)
                else:
                    explainer_correlations.append(np.nan)
            
            correlation_data.append(explainer_correlations)
    
    # Create DataFrame for heatmap
    correlation_df = pd.DataFrame(
        np.array(correlation_data).T,  # Transpose so explainers are rows
        index=explainers,
        columns=model_metric_combinations
    )
    
    # Get theme settings
    dark_mode = st.session_state.get('dark_mode', False)
    
    # Set up the matplotlib figure
    plt.style.use('dark_background' if dark_mode else 'default')
    fig, ax = plt.subplots(figsize=(max(12, len(model_metric_combinations) * 0.8), max(8, len(explainers) * 0.6)))
    
    # Create heatmap
    heatmap = sns.heatmap(
        correlation_df,
        annot=True,              # Show values in cells
        fmt='.2f',               # Format numbers to 2 decimals
        cmap='RdYlGn',           # Red-Yellow-Green (higher=better=green)
        center=0.5,              # Center colormap at 0.5
        vmin=0,                  # Minimum value is 0
        vmax=1,                  # Maximum value is 1
        square=False,            # Don't force square cells
        linewidths=0.5,          # Grid lines
        cbar_kws={
            'label': 'Ranking Stability Score (1 = Perfect Stability)',
            'shrink': 0.8
        },
        ax=ax
    )
    
    # Dark mode adjustments
    if dark_mode:
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(colors='white')
        cbar.ax.yaxis.label.set_color('white')
    
    # Customize appearance
    ax.set_title(
        'Explainer Ranking Stability Across Model-Metric Combinations\n(Higher values = More stable rankings)',
        fontsize=14,
        pad=20,
        color='white' if dark_mode else 'black'
    )
    
    ax.set_xlabel('Models', fontsize=12, color='white' if dark_mode else 'black')
    ax.set_ylabel('Explainers', fontsize=12, color='white' if dark_mode else 'black')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10, color='white' if dark_mode else 'black')
    plt.yticks(rotation=0, fontsize=10, color='white' if dark_mode else 'black')
    
    plt.tight_layout()
    layout.pyplot(fig)
    plt.close()
    
    # Add interpretation guide
    layout.markdown("""
    **📊 Interpretation Guide:**
    - 🟢 **Green (0.8-1.0)**: Highly stable - ranking almost identical
    - 🟡 **Yellow (0.5-0.8)**: Moderately stable - small ranking changes
    - 🟠 **Orange (0.2-0.5)**: Less stable - noticeable ranking changes  
    - 🔴 **Red (0.0-0.2)**: Unstable - large ranking changes
    
    **Ideal explainers** show green across all models.
    """)

def plot_explainer_stability_summary(df_full, df_subsample, layout):
    """
    Create a summary plot showing average stability of explainers across models and metrics
    Args:
        df_full: DataFrame with full dataset rankings
        df_subsample: DataFrame with subsample rankings  
        layout: Streamlit layout object
    This function calculates the average stability score for each explainer based on the rank differences
    between the full dataset and subsample rankings. It then plots these scores in a horizontal bar chart.
    """
    
    explainers = sorted(df_full['explainer'].unique())
    models = sorted(df_full['model'].unique())
    metrics = sorted(df_full['metric'].unique())
    
    # Calculate average stability per explainer
    explainer_stability_scores = []
    
    for explainer in explainers:
        stability_scores = []
        
        for model in models:
            for metric in metrics:
                full_data = df_full[
                    (df_full['model'] == model) & 
                    (df_full['metric'] == metric) & 
                    (df_full['explainer'] == explainer)
                ]
                
                sub_data = df_subsample[
                    (df_subsample['model'] == model) & 
                    (df_subsample['metric'] == metric) & 
                    (df_subsample['explainer'] == explainer)
                ]
                
                if not full_data.empty and not sub_data.empty:
                    full_rank = full_data['rank'].iloc[0]
                    sub_rank = sub_data['rank'].iloc[0]
                    
                    # Calculate stability score
                    max_possible_diff = len(explainers) - 1
                    rank_diff = abs(full_rank - sub_rank)
                    stability_score = 1 - (rank_diff / max_possible_diff) if max_possible_diff > 0 else 1
                    stability_scores.append(stability_score)
        
        avg_stability = np.mean(stability_scores) if stability_scores else np.nan
        explainer_stability_scores.append(avg_stability)
    
    # Create DataFrame for plotting
    summary_df = pd.DataFrame({
        'explainer': explainers,
        'stability_score': explainer_stability_scores
    }).sort_values('stability_score', ascending=False)
    
    # Set up plot
    dark_mode = st.session_state.get('dark_mode', False)
    plt.style.use('dark_background' if dark_mode else 'default')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar plot with color mapping
    colors = plt.cm.RdYlGn(summary_df['stability_score'])
    bars = ax.barh(summary_df['explainer'], summary_df['stability_score'], color=colors)
    
    # Customize plot
    ax.set_title('Overall Explainer Stability Ranking\n(Higher = More Stable)', 
                fontsize=14, color='white' if dark_mode else 'black')
    ax.set_xlabel('Average Stability Score', fontsize=12, color='white' if dark_mode else 'black')
    ax.set_ylabel('Explainers', fontsize=12, color='white' if dark_mode else 'black')
    ax.set_xlim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, summary_df['stability_score']):
        if not np.isnan(value):
            ax.text(value + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', va='center', fontsize=10,
                    color='white' if dark_mode else 'black')
    
    # Add grid for easier reading
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    layout.pyplot(fig)
    plt.close()

# Usage function
def display_ranking_correlation_analysis(df_full, df_subsample, layout):
    """
    Display the ranking correlation analysis dashboard with explainer stability summary.
    Args:
        df_full: DataFrame with full dataset rankings
        df_subsample: DataFrame with subsample rankings  
        layout: Streamlit layout object
    Returns:
        None: Displays the analysis dashboard directly in the Streamlit layout.
    """
    layout.subheader("🔗 Explainer Ranking Correlation Analysis")
    
    # Main correlation matrix
    plot_ranking_correlation_matrix(df_full, df_subsample, layout)
    
    # Summary ranking
    layout.subheader("🏆 Overall Stability Ranking")
    plot_explainer_stability_summary(df_full, df_subsample, layout)

# ──────────────────────────────────────────────────────────────