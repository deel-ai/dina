import streamlit as st
from utils import transform_data, naive_update_query_table, naive_fast_query
from plotters import plot3
from datasets.alleviater import alleviate_metric

def store_selected_filters():
    st.session_state.selected_filters['model'] = st.session_state.model_filter
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
    

# Switch to DataFrame view or plot view
switch_to_dataframe = st.checkbox("Switch to DataFrame View", value=False, key="switch_to_dataframe")

mean_table = "'.\\datasets\\all_mean_agg_fidelity.csv'"
mean_query, mean_query_args = naive_update_query_table(st.session_state.selected_filters, mean_table, skip_instance=True, skip_label_prediction=True, skip_metric=True)
# Everything is in list since all the metrics are always plotted

metric_list = ['Deletion', 'Insertion', 'MuFidelity', 'Sparseness', 'Complexity']
metric_tabs = st.tabs([f"{metric}" for metric in metric_list])
query3_list = [mean_query + " AND metric = ? LIMIT 1050" for metric in metric_list] # The 1050 limit is unnecessary, there are 1050 rows in the datatable
query3_args_list = [mean_query_args + [alleviate_metric[metric]] for metric in metric_list]
df3_list = [naive_fast_query(query3, query3_args) for query3, query3_args in zip(query3_list, query3_args_list)]

if switch_to_dataframe: sdf3_list = [df3.style.background_gradient(cmap="YlOrRd", subset=["mean", "std_dev"]) for df3 in df3_list]
for index in range(len(metric_list)):
    if switch_to_dataframe:
        with metric_tabs[index]:
            st.dataframe(sdf3_list[index], use_container_width=True, key=f"mean_dataframe_{index}")
    else:
        data = transform_data(df3_list[index])
        plot3(data, metric_tabs[index], True, plot_key=f"mean_plot_{index}", dark_mode=st.session_state.dark_mode)
