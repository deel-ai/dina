# ───────────────────────────────────────────────────────────────────────────
# This is the code for the interactive plotting data page with AgGrid table
# ───────────────────────────────────────────────────────────────────────────
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from utils import clean_all_filters, query_parquet_data
from plotters import plot_plotting_data_1, plot_plotting_data_2, dialog_plotted_data

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
    st.session_state.selected_filters['explainer'] = st.session_state.explainer_filter
    st.session_state.selected_filters['metric'] = st.session_state.metric_filter
    st.session_state.selected_filters['instance'] = st.session_state.instance_input
    st.session_state.selected_filters['label_prediction'] = st.session_state.label_prediction_filter
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
    
    # Explainer filter
    explainer_filter = st.multiselect(
        "Explainer:", 
        st.session_state.filter_options['explainer'], 
        default=st.session_state.selected_filters.get('explainer', []), # The selected values are stored in session state, so they persist across pages
        key="explainer_filter"
    )
    
    # Metric filter
    metric_filter = st.multiselect(
        "Metric:", 
        st.session_state.filter_options['metric'], 
        default=st.session_state.selected_filters.get('metric', []), # The selected values are stored in session state, so they persist across pages
        key="metric_filter"
    )
    
    # Instance filter
    st.multiselect(
        "Choose Instance:", 
        options=None, # No values are predefined, the users must input them themselves
        key="instance_input",
        accept_new_options=True # Allows the user to input a new instance
    )
    
    # Label/Prediction filter
    st.selectbox(
        "Display Options for Label and Prediction:",
        options=["Display all", "Display only good predictions", "Display only wrong predictions"],
        index=0,
        key="label_prediction_filter",
    )

# ─────────────────────────────────────────────────────────────────────────────────────
# Data querying and processing
# ─────────────────────────────────────────────────────────────────────────────────────
df = query_parquet_data(st.session_state.selected_filters)

# ─────────────────────────────────────────────────────────────────────────────────────
# Main content area - interactive table and plotting controls
# ─────────────────────────────────────────────────────────────────────────────────────
if not df.empty:
    button_holder = st.columns([5,3,2]) # Layout for plot controls - wider column for plot type selector

    # Plot type selection
    plot_type_select = button_holder[0].selectbox(
        "Plot Selection,",
        label_visibility="collapsed",
        options=["model / (explainer-metric)", "explainer / (model-metric)"], # Two different grouping perspectives
        index=0, 
        key="plot_type_select",
    )

    # Mapping of plot types to their corresponding functions
    plot_correspondance = {
        "model / (explainer-metric)": plot_plotting_data_1,
        "explainer / (model-metric)": plot_plotting_data_2,
    }

    # Action buttons for plotting
    plot_selected_data_button = button_holder[1].button("Plot Selected Data") # Plot only user-selected rows
    plot_all_data_button = button_holder[2].button("Plot All Data") # Plot all filtered data

    # ─────────────────────────────────────────────────────────────────────────────────
    # AgGrid interactive table setup
    # ─────────────────────────────────────────────────────────────────────────────────
    grid_options = GridOptionsBuilder.from_dataframe(df)
    grid_options.configure_selection("multiple") # Allow multiple row selection
    grid_options.configure_grid_options(enableCellTextSelection=True) # Enable text selection within cells

    datatable = AgGrid(
        df,
        gridOptions=grid_options.build(),
        enable_enterprise_modules=True,
        update_mode="MODEL_CHANGED",
        fit_columns_on_grid_load=True,
        theme="streamlit",
    )

    # ─────────────────────────────────────────────────────────────────────────────────
    # Button action handlers
    # ─────────────────────────────────────────────────────────────────────────────────
    if plot_selected_data_button:
        if not(getattr(datatable['selected_rows'],'empty',True)): # Check if any rows are selected
            sampled_data = datatable['selected_rows'] if len(datatable['selected_rows']) <= 100 else datatable['selected_rows'].sample(n=100) # Limit to 100 rows for performance
            dialog_plotted_data(sampled_data, plot_correspondance[plot_type_select]) # Open plot dialog with selected data
        else:
            st.error("No row selected. Please select a row to plot data.") # User feedback for no selection

    if plot_all_data_button:
        sampled_data = df.sample(n=100) if len(df) > 100 else df # Sample 100 rows if dataset is large, otherwise use all data
        dialog_plotted_data(sampled_data, plot_correspondance[plot_type_select]) # Open plot dialog with sampled data
else:
    st.info("No data found for the selected filters.") # User feedback when no data matches their filter criteria