import streamlit as st
from utils import MODEL_LIST, ACTIVATION_LIST, EXPLAINER_LIST, METRIC_LIST

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False


# ──────────────────Setting session state──────────────────────
# Initialize selected values in session state
if "selected_filters" not in st.session_state:
    st.session_state.selected_filters = {
        'model': [],
        'activation': 'softmax',
        'explainer': [],
        'metric': [],
        'instance': [],
        'label_prediction': 'Display all'
    }

if "filter_options" not in st.session_state:
    st.session_state.filter_options = {
        'model': MODEL_LIST, 
        'activation': ACTIVATION_LIST, 
        'explainer': EXPLAINER_LIST, 
        'metric': METRIC_LIST,
        'instance': []
    }
# ─────────────────────────────────────────────────────────────


# ──────────────────Streamlit App Main Area────────────────────
# Execute the query and convert the result to a DataFrame
try:
    
    def home():
        st.image("banner.png")
        st.subheader("What are the different tabs, and what do they help visualize ?")
        
        with st.expander("🏠 Home Page"):
            st.write("A brief introduction to the app and its purpose.")

        with st.expander("📊 Plotting Purpose Data"):
            st.write("Allows you to plot data based on selected filters, " \
            "two plotting options are available, one allows to compare model performances across different explainers and metrics, " \
            "the other allows to compare explainer performances across different models and metrics." \
            "The table is limited to a total of 10000 rows, for visualization purposes, it is recommended to select a smaller subset of instances before plotting. " \
            "You can also manually select data, through the use of mouse clicks, shift and control")

        with st.expander("🎨 Gradient Score Data"):
            st.write("Displays a gradient score visualization for selected data" \
            ", the data is limited to 100 rows for performance reasons. " \
            "The gradient score is a measure of the importance of each feature in the model's prediction, " \
            "Note that score signification varies depending on metric, meaning that for some, a higher score is better, while for others, a lower score is better. " \
            "It is thus recommended to limit the data to a single metric when looking purely at scores ")
            
        with st.expander("📈 Mean Visualization"):
            st.write("Provides mean and standard deviation visualizations for different metrics." \
            "What is displayed is the mean and standard deviation over the 50000 instances of imagenet of the selected model and explainer for a selected metric")
            
        with st.expander("🔥 Heatmap"):
            st.write("Displays a heatmap visualization of the data." \
            "What is displayed here is the standard deviation of the mean scores given by a certain metric across either explainers or models" \
            "Lower std_dev of the mean scores indicates that the metric gives consistent scores across the different explanations of a model " \
            "or across the different explanations of an explainer for different models. ")
            
        with st.expander("🏆 Leaderboard"):
            st.write("Displays a leaderboard of models based on selected metrics and activations." \
            "The leaderboard gives intel about what could be the best explainer for a chosen model and metric")
            
        st.subheader("Other useful information")
        with st.expander("🎯 Regarding Filters"):
            st.write("Filters are used to select the data that is displayed in the app. " \
            "You can select multiple models, explainers, metrics and instances. " \
            "You will need to aply your selected filters for them to take effect on the selected data."
            "The instance filter is a multiselect input box, you must manually input your preferred instances of image net" \
            " and press enter to add them to the list. ")
        st.write("🔗 Link to the official DEEL github: https://github.com/deel-ai")




    pages = [
        st.Page(home, title="🏠 Home Page"),
        st.Page(".\\pages\\leaderboard.py", title="🏆 Leaderboard"),
        st.Page(".\\pages\\plotting_data.py", title="📊 Plotting Purpose Data"),
        st.Page(".\\pages\\gradient_data.py", title="🎨 Gradient Score Data"),
        st.Page(".\\pages\\mean_scatter_map.py", title="📈 Mean Visualization"),
        st.Page(".\\pages\\heatmap.py", title="🔥 Heatmap"),
    ]
    #        st.Page(rank_info, title="🔍 Rank Info")]
    pg = st.navigation(pages, position="top")
    pg.run()


except Exception as e:
    st.error(f"Error: {e}")
# ─────────────────────────────────────────────────────────────


