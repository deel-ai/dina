# ─────────────────────────────────────────────────────
# This is the code for the rank agreement analysis page
# ─────────────────────────────────────────────────────
import streamlit as st
from utils import clean_all_filters, METRIC_LIST, query_parquet_data
from plotters import plot_rank_agreement

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
    st.session_state.selected_filters['model'] = st.session_state.model_filter
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
    model_filter = st.multiselect(
        "Model:", 
        st.session_state.filter_options['model'], # The possible values, which correspond to the existing models in the database
        default=st.session_state.selected_filters.get('model', []), # The selected values are stored in session state, so they persist across pages
        key="model_filter"
    )

df_all = query_parquet_data(st.session_state.selected_filters, mean_values=True, skip_metric=True, skip_explainer=True, columns=['model', 'activation', 'explainer', 'metric', 'rank'])
df_list = [df_all[df_all['metric'] == metric] for metric in METRIC_LIST]

metric_tabs = st.tabs([f"{metric}" for metric in METRIC_LIST]) # Create tabs for each metric

for index in range(len(METRIC_LIST)):
    if not df_list[index].empty and len(df_list[index]['model'].unique()) > 1:
        plot_rank_agreement(df_list[index], metric_tabs[index], plot_key=f"rank_agreement_{index}", dark_mode=st.session_state.dark_mode)
    else:
        with metric_tabs[index]:
            st.info(f"Not enough data available for {METRIC_LIST[index]} with the current filters.")