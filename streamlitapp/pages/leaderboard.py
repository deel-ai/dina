import streamlit as st
from utils import naive_fast_query, transform_data, naive_update_query_table
from datasets.alleviater import alleviate_metric, alleviate_model, alleviate_activation
from plotters import plot5

def store_selected_filters():
    st.session_state.selected_filters['activation'] = st.session_state.activation_filter
    st.session_state.selected_filters['explainer'] = st.session_state.explainer_filter

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
    model_select=st.selectbox(
        "Model:",
        st.session_state.filter_options['model'],
        index=0,
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

    metric_select=st.selectbox(
        "Metric:",
        st.session_state.filter_options['metric'],
        index=0,
        key="metric_select"
    )


# Query leaderboard data
mean_table = "'.\\datasets\\all_mean_agg_fidelity.csv'"
mean_query, mean_query_args = naive_update_query_table(st.session_state.selected_filters, mean_table, skip_model=True, skip_metric=True, skip_instance=True, skip_label_prediction=True)


mean_query += " AND model = ? AND metric = ? LIMIT 1050"
mean_query_args += [alleviate_model[model_select], alleviate_metric[metric_select]]

df5 = naive_fast_query(mean_query, mean_query_args)

# Display leaderboard
if not df5.empty: 
    plot5(transform_data(df5), st, dark_mode=st.session_state.dark_mode)
else: 
    st.info("No data found for the selected model and metric.")