import streamlit as st
from utils import add_suffix, fast_query, create_view_table

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
    
union_query = create_view_table(st.session_state.selected_filters)
suffix, query_args = add_suffix(st.session_state.selected_filters)
suffix += " LIMIT 100" 

df2 = fast_query(union_query, suffix, query_args)
# Adding a gradient requires a certain computation time, thus why the data is sampled to 100 rows

if len(df2) > 100:
    df2 = df2.sample(n=100)
sdf2 = df2.style.background_gradient(cmap="YlOrRd", subset=["score"])
st.dataframe(sdf2, use_container_width=True)

