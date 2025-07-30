# ──────────────────────────────────────────────────────────────────────────────
# This is the code for the training time analysis and batch inference time page
# ──────────────────────────────────────────────────────────────────────────────
import streamlit as st
from utils import clean_all_filters, MODEL_LIST, query_parquet_data
from plotters import plot_training_time, plot_batch_time
from utils import METRIC_LIST

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
    st.session_state.selected_filters['unique_model'] = st.session_state.model_select
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
     
    # Model filter
    model_filter = st.selectbox(
        "Model:",
        st.session_state.filter_options['model'],
        index=st.session_state.filter_options['model'].index(st.session_state.selected_filters.get('unique_model', 'MLPMixer')), # Default to X model
        key="model_select"
    )

    # Subsample Filter
    subsample_filter=st.number_input(
        "Subsample Size (recommended < 5,000):",
        min_value=1,
        max_value=50000,
        value=st.session_state.selected_filters.get('subsample_size'),
        key="subsample_size_input", # Change subsample size when input changes
    )


# Execute queries and retrieve data
df_fidelity = query_parquet_data(st.session_state.selected_filters, mean_values=True, skip_explainer = True, skip_metric=True, unique_model=True, columns=['model', 'activation', 'explainer', 'metric', 'mean', 'std_dev']) # Query fidelity scores
df_it = query_parquet_data(st.session_state.selected_filters, inference_values=True, skip_explainer=True, unique_model=True,columns=['model', 'activation', 'explainer', 'batch_size', 'mean_time', 'std_time', 'gpu_type'])
# ─────────────────────────────────────────────────────────────────────────────────────
# Data preprocessing and normalization
# ─────────────────────────────────────────────────────────────────────────────────────
df_it["mean_time_per_sample"] = df_it["mean_time"] / df_it["batch_size"] # Calculate per-sample time from batch time
df_it["std_time_per_sample"] = df_it["std_time"] / df_it["batch_size"] # Calculate per-sample standard deviation
df_it['model'] = df_it['model'].replace('ConvNeXtV2', 'ConvNeXtV2Base') # Replace model naming convention from database
df_it = df_it[df_it['model'].isin(MODEL_LIST)] # Filter to only include valid models from our model list


# Process all metrics - each gets its own tab for comparison
metric_tabs = st.tabs([f"{metric}" for metric in METRIC_LIST]) # Create tabs for each metric

# Filter dataframe for each metric
df_list = [df_fidelity[df_fidelity['metric'] == metric] for metric in METRIC_LIST]  # Filter by metric using pandas

# ─────────────────────────────────────────────────────────────────────────────────────
# Display content in tabs - either scatter plots or styled DataFrames
# ─────────────────────────────────────────────────────────────────────────────────────
for index in range(len(METRIC_LIST)):
    if not df_list[index].empty:
        plot_training_time(df_list[index], df_it, metric_tabs[index], plot_key=f"training_plot_{index}")
        plot_batch_time(df_it, metric_tabs[index], plot_key = f"batch_time_{index}") # Plot batch inference time data
    else:
        with metric_tabs[index]: # User feedback when no data matches their filter criteria
            st.info(f"No data available for {METRIC_LIST[index]} with the selected filters.")
