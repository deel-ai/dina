import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from utils import add_suffix, fast_query, transform_data, create_view_table
from plotters import plot1_1, plot1_2, dialog_plotted_data

def store_selected_filters():
    st.session_state.selected_filters['model'] = st.session_state.model_filter
    st.session_state.selected_filters['activation'] = st.session_state.activation_filter
    st.session_state.selected_filters['explainer'] = st.session_state.explainer_filter
    st.session_state.selected_filters['metric'] = st.session_state.metric_filter
    st.session_state.selected_filters['instance'] = st.session_state.instance_input
    st.session_state.selected_filters['label_prediction'] = st.session_state.label_prediction_filter


def switch_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Sidebar filters
with st.sidebar:
    st.toggle("Toggle when in dark mode", key="dark_mode_toggle", value=st.session_state.dark_mode, on_change=switch_dark_mode)
    st.write("------")

    st.subheader("🔧 Filters")

    col1, col2 = st.columns(2, vertical_alignment="bottom")
    with col1:
        st.title("Filters")
    with col2:
        st.button("Apply Filters", on_click=store_selected_filters, key="apply_filters_button")
    
    # Model filter
    model_filter = st.multiselect(
        "Model:", 
        st.session_state.filter_options['model'], 
        default=st.session_state.selected_filters.get('model', []),
        key="model_filter"
    )
    
    # Activation filter
    activation_filter = st.selectbox(
        "Activation:",
        st.session_state.filter_options['activation'],
        index=st.session_state.filter_options['activation'].index(st.session_state.selected_filters.get('activation', 'softmax')),
        key="activation_filter"
    )
    
    # Explainer filter
    explainer_filter = st.multiselect(
        "Explainer:", 
        st.session_state.filter_options['explainer'], 
        default=st.session_state.selected_filters.get('explainer', []),
        key="explainer_filter"
    )
    
    # Metric filter
    metric_filter = st.multiselect(
        "Metric:", 
        st.session_state.filter_options['metric'], 
        default=st.session_state.selected_filters.get('metric', []),
        key="metric_filter"
    )
    
    # Instance filter
    st.multiselect(
        "Choose Instance:", 
        options=None,
        key="instance_input",
        accept_new_options=True
    )
    
    # Label/Prediction filter
    st.selectbox(
        "Display Options for Label and Prediction:",
        options=["Display all", "Display only good predictions", "Display only wrong predictions"],
        index=0,
        key="label_prediction_filter",
    )
    

# Main content area
union_query=create_view_table(st.session_state.selected_filters)

suffix, query_args = add_suffix(st.session_state.selected_filters)
suffix += "LIMIT 10000" 
df1 = fast_query(union_query, suffix, query_args)

button_holder = st.columns([5,3,2])

# Plot type selection
plot_type_select = button_holder[0].selectbox(
    "Plot Selection,",
    label_visibility="collapsed",
    options=["model / (explainer-metric)", "explainer / (model-metric)"], 
    index=0, 
    key="plot_type_select",
)

plot_correspondance = {
    "model / (explainer-metric)": plot1_1,
    "explainer / (model-metric)": plot1_2
}

# Action buttons
plot_selected_data_button = button_holder[1].button("Plot Selected Data")
plot_all_data_button = button_holder[2].button("Plot All Data")

# AgGrid setup
grid_options = GridOptionsBuilder.from_dataframe(df1)
grid_options.configure_selection("multiple")
grid_options.configure_grid_options(enableCellTextSelection=True)

datatable = AgGrid(
    df1,
    gridOptions=grid_options.build(),
    enable_enterprise_modules=True,
    update_mode="MODEL_CHANGED",
    fit_columns_on_grid_load=True,
    theme="streamlit",
)

# Button actions
if plot_selected_data_button:
    if not(getattr(datatable['selected_rows'],'empty',True)):
        transformed_data = transform_data(datatable['selected_rows']) if len(datatable['selected_rows']) <= 100 else transform_data(datatable['selected_rows'].sample(n=100))
        dialog_plotted_data(transformed_data, plot_correspondance[plot_type_select])
    else:
        st.error("No row selected. Please select a row to plot data.")

if plot_all_data_button:
    transformed_data = transform_data(df1) if len(df1) <= 100 else transform_data(df1.sample(n=100))
    dialog_plotted_data(transformed_data, plot_correspondance[plot_type_select])