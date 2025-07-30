# DINA app - Main File
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# This is the main file for the DINA app, it initializes the app, sets up the sidebar filters, and runs the pages.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────

import streamlit as st
import random
from utils import MODEL_LIST, EXPLAINER_LIST, METRIC_LIST

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False


# ──────────────────Setting session state──────────────────────
# Initialize selected values in session state
if "selected_filters" not in st.session_state: # Those are the values that are changed when filters are selected and that allow for the chosen filters to carry from one page to another
    st.session_state.selected_filters = {
        'model': [],
        'activation': 'softmax',
        'explainer': [],
        'metric': [],
        'instance': [],
        'label_prediction': 'Display all',
        'unique_model': 'MLPMixer', # Sometimes, we need only one model (e.g. leaderboard)
        'unique_metric': 'Deletion', # Same for the metrics
        'subsample': random.sample(range(1, 50001), 5000),
        'subsample_size': 5000, # This is the size of the subsample, it is used in the subsample stability page
    }

if "filter_options" not in st.session_state: # Those are the values that are used to populate the filter options in the sidebar
    # If you ever want to change it so that the filter options are dynamic, this is one of the elements that are going to be useful
    st.session_state.filter_options = {
        'model': MODEL_LIST, 
        'explainer': EXPLAINER_LIST, 
        'metric': METRIC_LIST,
        'instance': [],
    }
# ─────────────────────────────────────────────────────────────


# ──────────────────Streamlit App Main Area────────────────────
# Execute the query and convert the result to a DataFrame
try:
    def home():
        """Display the home page with introduction and feature explanations.
        
        Creates the main landing page of the application with a banner image and
        expandable sections explaining each tab's functionality. Provides user
        guidance on how to use filters and navigate the different visualization
        options available in the app.
        
        Args:
            None
            
        Returns:
            None: Displays content directly in the Streamlit interface.
            
        Example:
            >>> home()
            >>> # Displays the home page with banner and expandable help sections
        """

        st.image("banner.png") # This one is the wrong at the time I write this, this should be a DINA banner not an XPlique one
        st.subheader("What are the different tabs, and what do they help visualize ?")
        
        with st.expander("🏠 Home Page"): # st.expander are used here since they allow the explanations to take less space, making it more enjoyable for the user
            st.write("A brief introduction to the app and its purpose.")
            
        with st.expander("🏆 Leaderboard"):
            st.write("Displays a leaderboard of explainers for a chosen model and metric." \
            "The leaderboard gives intel about what could be the best explainer for a chosen model and metric")

        with st.expander("🤝 Rank Agreement"):
            st.write("Shows the rank agreement between different explainers for a chosen model and metric." \
            "This can help understand how much the explainers agree on the importance of features for a given model and metric.")

        with st.expander("⏱️ Training Time"):
            st.write("Displays the training time and batch inference time for different models and explainers." \
            "This can help understand the computational cost of using different explainers with different models. " \
            "Both the training time and batch inference time are displayed in milliseconds. ")
        
        with st.expander("📈 Mean Visualization"):
            st.write("Provides mean and standard deviation visualizations for different metrics." \
            "What is displayed is the mean and standard deviation over the 50000 instances of imagenet of the selected model and explainer for a selected metric")

        with st.expander("🎨 Gradient Score Data"):
            st.write("Displays a gradient score visualization for selected data" \
            ", the data is limited to 1000 rows for performance reasons. " \
            "The gradient score is a measure of the importance of each feature in the model's prediction, " \
            "Note that score signification varies depending on metric, meaning that for some, a higher score is better, while for others, a lower score is better. " \
            "It is thus recommended to limit the data to a single metric when looking purely at scores ")
            
        with st.expander("📊 Plotting Purpose Data"):
            st.write("Allows you to plot data based on selected filters, " \
            "two plotting options are available, one allows to compare model performances across different explainers and metrics, " \
            "the other allows to compare explainer performances across different models and metrics." \
            "The table is limited to a total of 10000 rows, for visualization purposes, it is recommended to select a smaller subset of instances before plotting. " \
            "You can also manually select data, through the use of mouse clicks, shift and control")

        with st.expander("🔄 Subsample Stability"):
            st.write("Displays the stability of explainers across different subsamples of data." \
            "The subsample stability is a measure of how much the explainers agree on the importance of features for a given model and metric, " \
            "when using different subsets of the data. " \
            "The subsample size can be adjusted, but it is recommended to keep it below 5000 for performance reasons. " \
            "The subsample size is set to 5000 by default, but you can change it in the filters sidebar. ")

        st.subheader("Other useful information")
        with st.expander("🎯 Regarding Filters"):
            st.write("Filters are used to select the data that is displayed in the app. " \
            "You can select multiple models, explainers, metrics and instances. " \
            "You will need to aply your selected filters for them to take effect on the selected data."
            "The instance filter is a multiselect input box, you must manually input your preferred instances of image net" \
            " and press enter to add them to the list. ")
        st.write("🔗 Link to the official DEEL github: https://github.com/deel-ai")




    pages = [ # If you ever want to put one page before the other, just change their order in here, but please, let the homepage be first. Since some of the filter values are created on the loading of the home page, not loading it can cause issues
        st.Page(home, title="🏠 Home Page"),
        st.Page(".\\pages\\leaderboard.py", title="🏆 Leaderboard"),
        st.Page(".\\pages\\rank_agreement.py", title="🤝 Rank Agreement"),
        st.Page(".\\pages\\mean_scatter_map.py", title="📈 Mean Visualization"),
        st.Page(".\\pages\\training_time.py", title="⏱️ Training Time"),
        st.Page(".\\pages\\gradient_data.py", title="🎨 Gradient Score Data"),
        st.Page(".\\pages\\plotting_data.py", title="📊 Plotting Purpose Data"),
        st.Page(".\\pages\\subsample.py", title="🔄 Subsample Stability"),
    ]
    pg = st.navigation(pages, position="top") # The position top is just a personal preference, if you delete the parameter, the page selection will be in the sidebar
    pg.run()


except Exception as e:
    st.error(f"Error: {e}")
# ─────────────────────────────────────────────────────────────


