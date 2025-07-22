import streamlit as st
from utils import naive_update_query_table, naive_fast_query
from plotters import plot4_1, plot4_2

def store_selected_filters():
    st.session_state.selected_filters['model'] = st.session_state.model_filter
    st.session_state.selected_filters['activation'] = st.session_state.activation_filter
    st.session_state.selected_filters['explainer'] = st.session_state.explainer_filter
    st.session_state.selected_filters['metric'] = st.session_state.metric_filter

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
    st.multiselect(
        "Model:", 
        st.session_state.filter_options['model'], 
        default=st.session_state.selected_filters.get('model', []),
        key="model_filter"
    )
    
    # Activation filter
    st.selectbox(
        "Activation:",
        st.session_state.filter_options['activation'],
        index=st.session_state.filter_options['activation'].index(st.session_state.selected_filters.get('activation', 'softmax')),
        key="activation_filter"
    )
    
    # Explainer filter
    st.multiselect(
        "Explainer:", 
        st.session_state.filter_options['explainer'], 
        default=st.session_state.selected_filters.get('explainer', []),
        key="explainer_filter"
    )
    
    # Metric filter
    st.multiselect(
        "Metric:", 
        st.session_state.filter_options['metric'], 
        default=st.session_state.selected_filters.get('metric', []),
        key="metric_filter"
    )
    
# Main content
col1, col2 = st.columns([3, 2], vertical_alignment="bottom")
with col1:
    st.write("Display Mean or Standard Deviation of the Mean Scores")
with col2:
    st.selectbox("", options=["Standard Deviation", "Mean"], index=0, key="mode_select", label_visibility="collapsed")

# Create tabs for different heatmap views
corrtab1, corrtab2 = st.tabs(["Across Explainers", "Across Models"])

mean_table = "'.\\datasets\\all_mean_agg_fidelity.csv'"
# Query the mean data
mean_query, mean_query_args = naive_update_query_table(st.session_state.selected_filters, mean_table)
df4 = naive_fast_query(mean_query + " LIMIT 1050", mean_query_args)

# Determine plot mode
plot_mode = "mean" if st.session_state.mode_select == "Mean" else "std"

# Display heatmaps in tabs
with corrtab1:
    plot4_1(df4, st, mode=plot_mode, dark_mode=st.session_state.dark_mode)

with corrtab2:
    plot4_2(df4, st, mode=plot_mode, dark_mode=st.session_state.dark_mode)