# ──────────────────────────────────────────────────────────
# This is the code for the subsample agreement analysis page
# ──────────────────────────────────────────────────────────
import streamlit as st
from utils import clean_all_filters, query_parquet_data
from plotters import WHICHISBETTER, display_ranking_correlation_analysis
import random

# ──────────────────────────────────────────────────────────────────────────────────────
# Code for the Apply Filters button, the selected filters are stored in session state
# on the press of the button
# ──────────────────────────────────────────────────────────────────────────────────────
def store_selected_filters():
    """Store current filter selections in session state.
    
    Transfers the current filter values from UI components to the selected_filters
    dictionary in session state. This allows filters to persist across page
    interactions and be used for data querying.
    
    Args:
        None: Reads from st.session_state UI component keys.
        
    Returns:
        None: Updates st.session_state.selected_filters dictionary.
        
    Example:
        >>> store_selected_filters()
        >>> # Stores current UI filter values in session state
    """
    st.session_state.selected_filters['unique_metric'] = st.session_state.metric_select
    st.session_state.selected_filters['subsample'] = random.sample(range(1,50001), st.session_state.subsample_filter)
# ──────────────────────────────────────────────────────────────────────────────────────

# The logic for the dark mode toggle
def switch_dark_mode():
    """Toggle dark mode setting in session state.
    
    Switches the dark_mode boolean value in session state, which affects
    the styling of charts and visualizations throughout the application.
    
    Args:
        None: Reads from st.session_state.dark_mode.
        
    Returns:
        None: Updates st.session_state.dark_mode boolean value.
        
    Example:
        >>> switch_dark_mode()
        >>> # Toggles dark mode on/off
    """
    st.session_state.dark_mode = not st.session_state.dark_mode

# Sidebar filters
with st.sidebar:
    st.toggle("Toggle when in dark mode", key="dark_mode_toggle", value=st.session_state.dark_mode, on_change=switch_dark_mode) # Dark mode toggle
    st.write("------")

    st.header("🔧 Filters")

    col1, col2 = st.columns(2, vertical_alignment="bottom") # Two columns for the title and the button to be side by side
    with col1:
        st.button("Clean filters", on_click=clean_all_filters, key="clean_filters_button") # defined in utils.py
    with col2:
        st.button("Apply Filters", on_click=store_selected_filters, key="apply_filters_button")

    # Metric filter - single selection for specific leaderboard ranking
    metric_select=st.selectbox(
        "Metric:",
        st.session_state.filter_options['metric'],
        index=st.session_state.filter_options['metric'].index(st.session_state.selected_filters.get('unique_metric', 'Deletion')), # Default to Deletion metric
        key="metric_select"
    )

    # Subsample Filter
    subsample_filter=st.number_input(
        "Subsample Size (recommended < 5,000):",
        min_value=1,
        max_value=50000,
        value=len(st.session_state.selected_filters.get('subsample')),
        key="subsample_filter", # Change subsample size when input changes
    )

df = query_parquet_data(st.session_state.selected_filters, mean_values=True, skip_model=True, skip_explainer=True, unique_metric=True, columns=['model', 'activation', 'explainer', 'metric', 'mean', 'std_dev', 'rank'])
df_sub = query_parquet_data(st.session_state.selected_filters, subsample=True, skip_model=True, skip_explainer=True, unique_metric=True, max_rows=1000000, columns=['model', 'activation', 'explainer', 'metric', 'score'])

# ──────────────────────
# Add rankings to df_sub
# ──────────────────────
# Calculate the mean across all instances of the subsample
df_sub = df_sub.groupby(['model', 'explainer', 'metric'])['score'].agg(['mean', 'std']).reset_index()

# Flatten the column names and rename appropriately
df_sub.columns = ['model', 'explainer', 'metric', 'mean', 'std_dev']

# Add rankings based on the mean score
df_sub['mean'] = df_sub['mean'].round(3)
df_sub['std_dev'] = df_sub['std_dev'].round(3)

df_sub['rank'] = df_sub.groupby('model')['mean'].rank(ascending=WHICHISBETTER[st.session_state.selected_filters['unique_metric']] == "Lower is better", method='min')
df_sub['rank'] = df_sub['rank'].astype(int)  # Convert rank to integer type for better readability

df = df.drop(columns=['activation'])


display_ranking_correlation_analysis(df, df_sub, st)