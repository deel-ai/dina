import streamlit as st
import plotly.graph_objects as go
from utils import MODEL_LIST, EXPLAINER_LIST
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


COLOR14 = sns.color_palette("colorblind", 14, )
COLOR14 = list(map(lambda rgb: f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.8)", COLOR14))
COLOR5 = sns.color_palette("colorblind", 5)
COLOR5 = list(map(lambda rgb: f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.8)", COLOR5))
SHAPES = ["triangle-up", "square", "diamond", "cross", "circle"]
GOLDMEDAL = "🏅"
SILVERMEDAL = "🥈"
BRONZEMEDAL = "🥉"
WHICHISBETTER = {
    "Deletion": "Lower is better",
    "MuFidelity": "Higher is better",
    "Sparseness": "Higher is better",
    "Complexity": "Lower is better",
    "Insertion": "Higher is better"
}
ARROWUP = "↑"
ARROWDOWN = "↓"
ARROWCORRESPONDANCE = {
    "Higher is better": ARROWUP,
    "Lower is better": ARROWDOWN
}

# ────────────────────────────────────────────────────────────
# Plots scores for each model organized by (explainer, metric)
# ────────────────────────────────────────────────────────────
def plot1_1(data, layout, sl):
    
    for item in data:
        item['score'] = abs(float(item['score'])) if isinstance(item['score'], str) else item['score']

    pairs = [(data[i]['explainer'],data[i]['metric']) for i in range(len(data))]
    unique_pairs = list(set(pairs))
    unique_models = list(set([data[i]['model'] for i in range(len(data))]))
    
    model_colors = {MODEL_LIST[i]: COLOR5[i] for i in range(len(MODEL_LIST))}

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(color='rgba(0,0,0,0)', size=5),
        name=f"{ARROWUP} Higher is better",
        showlegend=True,
        legendgroup='Arrow Legend'
    ))
    
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(color='rgba(0,0,0,0)', size=0),
        name=f"{ARROWDOWN} Lower is better",
        showlegend=True,
        legendgroup='Arrow Legend'
    ))

    grouped_data = {}
    for explainer, metric in unique_pairs:
        key = f"{explainer} - {metric}(abs) {ARROWCORRESPONDANCE[WHICHISBETTER[metric]]}" if metric == "MuFidelity" else f"{explainer} - {metric} {ARROWCORRESPONDANCE[WHICHISBETTER[metric]]}"
        grouped_data[key] = [row for row in data if row['explainer'] == explainer and row['metric'] == metric]

    # Adding rows with same model

    mean_grouped_data = {}
    for key in list(grouped_data.keys()):
        mean_grouped_data[key] = []
    
    for model in unique_models:
        for key in list(grouped_data.keys()):
            if model in [row['model'] for row in grouped_data[key]]:
                mean_grouped_data[key].append({
                    'model': model,
                    'score': sum(row['score'] for row in grouped_data[key] if row['model'] == model) / len([row for row in grouped_data[key] if row['model'] == model]),
                })
    
    for key in mean_grouped_data:
        if mean_grouped_data[key]:
            mean_grouped_data[key].sort(key=lambda x: x['score'], reverse = key.endswith(ARROWUP))

    for key in mean_grouped_data:
        for row in range(len(mean_grouped_data[key])):
            mean_grouped_data[key][row]['medal'] = GOLDMEDAL if row == 0 else SILVERMEDAL if row == 1 else BRONZEMEDAL if row == 2 else ""

    # Add bars for each metric grouped by (model, explainer)
    for model in unique_models:
        x_values = []
        y_values = []
        medal_values = []
        for group, rows in mean_grouped_data.items():
            # Filter rows for the current metric
            filtered_rows = [row for row in rows if row['model'] == model]
            if filtered_rows:
                x_values.append(group)  # Group name (model-explainer)
                y_values.append(filtered_rows[0]['score'])  # Average score
                medal_values.append(filtered_rows[0]['medal'])  # Medal emoji

        # Add a bar trace for the current metric
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


    # Update layout
    fig.update_layout(
        title='Bar Chart Organized by (Explainer, Metric)',
        xaxis_title='Explainer - Metric',
        yaxis_title='Average Score',
        barmode='group',  # Group bars by x-axis categories
        showlegend=sl,
    )

    layout.plotly_chart(fig, use_container_width=True, key="plot1_1")
# ──────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────
# Plots scores for each explainer organized by (model, metric)
# ────────────────────────────────────────────────────────────
def plot1_2(data, layout, sl):

    for item in data:
        item['score'] = float(item['score']) if isinstance(item['score'], str) else item['score']
        
    pairs = [(data[i]['model'],data[i]['metric']) for i in range(len(data))]
    unique_pairs = list(set(pairs))
    unique_explainers = list(set([data[i]['explainer'] for i in range(len(data))]))
    
    explainer_colors = {EXPLAINER_LIST[i]: COLOR14[i] for i in range(len(EXPLAINER_LIST))} 


    fig = go.Figure()

    grouped_data = {}
    for model, metric in unique_pairs:
        key = f"{explainer} - {metric}(abs) {ARROWCORRESPONDANCE[WHICHISBETTER[metric]]}" if metric == "MuFidelity" else f"{explainer} - {metric} {ARROWCORRESPONDANCE[WHICHISBETTER[metric]]}"
        grouped_data[key] = [row for row in data if row['model'] == model and row['metric'] == metric]

    mean_grouped_data = {}
    for key in list(grouped_data.keys()):
        mean_grouped_data[key] = []
    
    for explainer in unique_explainers:
        for key in list(grouped_data.keys()):
            if explainer in [row['explainer'] for row in grouped_data[key]]:
                mean_grouped_data[key].append({
                    'explainer': explainer,
                    'score': sum(row['score'] for row in grouped_data[key] if row['explainer'] == explainer) / len([row for row in grouped_data[key] if row['explainer'] == explainer]),
                })
    
    for key in mean_grouped_data:
        mean_grouped_data[key].sort(key=lambda x: x['score'], reverse = key.endswith(ARROWUP))

    for key in mean_grouped_data:
        for row in range(len(mean_grouped_data[key])):
            mean_grouped_data[key][row]['medal'] = GOLDMEDAL if row == 0 else SILVERMEDAL if row == 1 else BRONZEMEDAL if row == 2 else ""

    for explainer in unique_explainers:
        x_values = []
        y_values = []
        medal_values = []
        for group, rows in mean_grouped_data.items():
            # Filter rows for the current metric
            filtered_rows = [row for row in rows if row['explainer'] == explainer]
            if filtered_rows:
                x_values.append(group)  # Group name (model-metric)
                y_values.append(filtered_rows[0]['score'])  # Average score
                medal_values.append(filtered_rows[0]['medal'])  # Medal emoji

        # Add a bar trace for the current explainer
        fig.add_trace(go.Bar(
            x=x_values,
            y=y_values,
            name=explainer,
            marker=dict(color=explainer_colors[explainer]),
            cliponaxis=False,
            text=medal_values,  # Medal emoji
            textposition='outside',
            textfont=dict(size=20)
        ))

    # Update layout
    fig.update_layout(
        title='Bar Chart Organized by (Model, Metric)',
        xaxis_title='Model - Metric',
        yaxis_title='Average Score',
        barmode='group',  # Group bars by x-axis categories
        showlegend=sl,
    )

    layout.plotly_chart(fig, use_container_width=True, key="plot1_2")
# ──────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────
# Plots mean scores by standard deviation for each model, explainer
# ──────────────────────────────────────────────────────────────────
def plot3(data, layout, sl, plot_key, dark_mode):
    unique_models = list(set([row['model'] for row in data]))
    unique_explainers = list(set([row['explainer'] for row in data]))

    model_shapes = {MODEL_LIST[i]: SHAPES[i] for i in range(len(MODEL_LIST))} 
    explainer_colors = {EXPLAINER_LIST[i]: COLOR14[i] for i in range(len(EXPLAINER_LIST))}

    fig = go.Figure()

    # ─────────Dummy traces for legend────────────
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            symbol=None,
            color='black' if dark_mode else 'white',
            size=10,
        ),
        name='Explainer:',
        showlegend=True,
        legendgroup='Explainer'
    ))
    for explainer in unique_explainers:
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                symbol='diamond',
                color=explainer_colors[explainer],
                size=10,
            ),
            name=explainer,
            showlegend=True,
            legendgroup='Explainer'
        ))
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            symbol=None,
            color='black' if dark_mode else 'white',
            size=10,
        ),
        name='Model:',
        showlegend=True,
        legendgroup='Model'
    ))
    for model in unique_models:
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                symbol=model_shapes[model],
                color='white' if dark_mode else 'black',
                size=10,
            ),
            name=model,
            showlegend=True,
            legendgroup='Model'
        ))
    # ────────────────────────────────────────────

    for row in data:
        fig.add_trace(go.Scatter(
            x=[row['mean']],
            y=[row['std_dev']],
            mode='markers',
            marker=dict(
                symbol=model_shapes[row['model']],
                color=explainer_colors[row['explainer']],
                size=10,
            ),
            text=f"{row['model']} - {row['explainer']}, {row['mean']} ± {row['std_dev']}",
            name=f"{row['model']} - {row['explainer']}",
            hoverinfo="text",
            showlegend=False
        ))
    fig.update_layout(
        title='Mean Scores by Standard Deviation',
        xaxis_title='Mean Score',
        yaxis_title='Standard Deviation',
        xaxis=dict(type='linear'),
        yaxis=dict(type='linear'),
        showlegend=sl,
    )
    layout.plotly_chart(fig, use_container_width=True, key=plot_key)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────
# Plots standard deviation of mean scores across explainers
# ─────────────────────────────────────────────────────────
def plot4_1(data, layout, mode, dark_mode):
    data['mean'] = pd.to_numeric(data['mean'], errors='coerce')
    data = data.dropna(subset=['mean'])

    grouped_data = data.groupby(['metric', 'model']).agg(std_dev=('mean', mode)).reset_index()

    grouped_data['metric'] = grouped_data['metric'].apply(lambda metric: f"{metric}")

    pivot_table = grouped_data.pivot(index='model', columns='metric', values='std_dev')

    fig, ax = plt.subplots(figsize=(12, 8))
    if dark_mode:
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
    
    heatmap = sns.heatmap(pivot_table, annot=True, cmap="flare", ax=ax)
    
    if dark_mode:
        # Style the colorbar
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(colors='white')
        cbar.ax.yaxis.label.set_color('white')
    
    ax.set_title(f"{"Std Dev" if mode == "std" else "Mean"} of Mean Scores Across Explainers", color='white' if dark_mode else 'black')
    ax.set_xlabel('Metric', color='white' if dark_mode else 'black')
    ax.set_ylabel('Model', color='white' if dark_mode else 'black')
    ax.tick_params(colors='white' if dark_mode else 'black')

    layout.pyplot(fig)

# ─────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────
# Plots standard deviation of mean scores across models
# ─────────────────────────────────────────────────────
def plot4_2(data, layout, mode, dark_mode=False):
    data['mean'] = pd.to_numeric(data['mean'], errors='coerce')
    data = data.dropna(subset=['mean'])

    grouped_data = data.groupby(['metric', 'explainer']).agg(std_dev=('mean', mode)).reset_index()

    grouped_data['metric'] = grouped_data['metric'].apply(lambda metric: f"{metric}")
        
    pivot_table = grouped_data.pivot(index='explainer', columns='metric', values='std_dev')
           
    fig, ax = plt.subplots(figsize=(12, 8))
    if dark_mode:
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
    
    heatmap = sns.heatmap(pivot_table, annot=True, cmap='crest', ax=ax)
    
    if dark_mode:
        # Style the colorbar
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(colors='white')
        cbar.ax.yaxis.label.set_color('white')
    
    ax.set_title(f"{"Std Dev" if mode == "std" else "Mean"} of Mean Scores Across Models", color='white' if dark_mode else 'black')
    ax.set_xlabel('Metric', color='white' if dark_mode else 'black')
    ax.set_ylabel('Explainer', color='white' if dark_mode else 'black')
    ax.tick_params(colors='white' if dark_mode else 'black')

    layout.pyplot(fig)
# ──────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────────────────────
# Plots a 
# ────────────────────────────────────────────────────────────────────────────────────────────
def plot5(data, layout, dark_mode):
    metric = data[0]['metric']

    leaderboard = [{'explainer': row['explainer'], 'mean': row['mean'], 'std_dev': row['std_dev']} for row in data]

    leaderboard.sort(key=lambda x: x['mean'], reverse= WHICHISBETTER[metric] == "Higher is better")
    
    # Create final leaderboard with merged columns and proper order
    final_leaderboard = []
    for ind in range(len(leaderboard)):
        medal = GOLDMEDAL if ind == 0 else SILVERMEDAL if ind == 1 else BRONZEMEDAL if ind == 2 else ""
        final_leaderboard.append({
            'Rank': medal,
            'Explainer': leaderboard[ind]['explainer'],
            'Score': f"{leaderboard[ind]['mean']:.4f} ± {leaderboard[ind]['std_dev']:.4f}"
        })
    
    df = pd.DataFrame(final_leaderboard)
    
    # Apply background colors to top 3 rows
    def highlight_top_rows(row):
        if dark_mode:
            if row.name == 0:  # Dark Gold
                return ['background-color: #B8860B'] * len(row)
            elif row.name == 1:  # Dark Silver
                return ['background-color: #708090'] * len(row)
            elif row.name == 2:  # Dark Bronze
                return ['background-color: #8B4513'] * len(row)
        else:
            if row.name == 0:  # Light Gold
                return ['background-color: #FFD700'] * len(row)
            elif row.name == 1:  # Light Silver
                return ['background-color: #C0C0C0'] * len(row)
            elif row.name == 2:  # Light Bronze
                return ['background-color: #CD7F32'] * len(row)
        return [''] * len(row)
    
    styled_df = df.style.apply(highlight_top_rows, axis=1)
    
    layout.dataframe(
        styled_df, 
        hide_index=True, 
        use_container_width=True,
        column_config={
            "Rank": st.column_config.TextColumn(width="small"),
            "Explainer": st.column_config.TextColumn(width="medium"),
            "Score": st.column_config.TextColumn(width="medium")
        }
    )

# ────────────────────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────
# Just the dialog in which the data is displayed
# ──────────────────────────────────────────────
@st.dialog("Plotted Data", width='large')
def dialog_plotted_data(data,plot):
    plot(data, st, True)