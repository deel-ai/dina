import duckdb as db
import streamlit as st
from datasets.alleviater import unalleviate_activation, unalleviate_explainer, unalleviate_model, unalleviate_metric, alleviate_model, alleviate_activation, alleviate_explainer, alleviate_metric

METRIC_LIST = ["Deletion", "MuFidelity", "Sparseness", "Complexity", "Insertion"]
EXPLAINER_LIST = ["Rise", "KernelShap", "HsicAttributionMethod", "IntegratedGradients", "Lime", 
                  "Saliency", "Occlusion", "VarGrad", "GradCAM", "SmoothGrad",
                  "GradientInput", "GradCAMPP", "SquareGrad", "SobolAttributionMethod"]
MODEL_LIST = ["MLPMixer", "ResNest50", "MaxVIT", "ConvNeXtV2Base", "DinoV2"]
ACTIVATION_LIST = ["softmax", "sigmoid", "logits"]

NUMBER_OF_MODEL = 5
NUMBER_OF_ACTIVATION = 3
NUMBER_OF_EXPLAINER = 14
NUMBER_OF_METRIC = 5



# Build the query dynamically based on filters
@st.cache_data
def naive_update_query_table(session_state, table, skip_model=False, skip_explainer=False, skip_metric=False, skip_instance=False, skip_label_prediction=False):
    query = f"SELECT * FROM {table} WHERE 1=1"
    args = []
    if not skip_model and session_state['model'] and len(session_state['model']) > 0:
        query += " AND model IN ?"
        args.append([f"{alleviate_model[m]}" for m in session_state['model']])
    if session_state['activation'] and len(session_state['activation']) > 0:
        query += " AND activation = ?"
        args.append(alleviate_activation[session_state['activation']])
    if not skip_explainer and session_state['explainer'] and len(session_state['explainer']) > 0:
        query += " AND explainer IN ?"
        args.append([f"{alleviate_explainer[e]}" for e in session_state['explainer']])
    if not skip_metric and session_state['metric'] and len (session_state['metric'])>0:
        query += " AND metric IN ?"
        args.append([f"{alleviate_metric[m]}" for m in session_state['metric']])
    if not skip_instance and session_state['instance'] and len(session_state['instance']) > 0:
        query += " AND instance = ?"
        args.append([f"{i}" for i in session_state['instance']])
    if not skip_label_prediction and session_state['label_prediction'] == "Display only good predictions":
        query += " AND label = prediction"
    elif not skip_label_prediction and session_state['label_prediction'] == "Display only wrong predictions":
        query += " AND label != prediction"
    return query, args

@st.cache_data
def add_suffix(session_state):
    suffix = ""
    args = []
    if session_state['label_prediction'] == "Display only good predictions":
        suffix = "AND label = prediction "
    elif session_state['label_prediction'] == "Display only wrong predictions":
        suffix = "AND label != prediction "
    if session_state['instance'] and len(session_state['instance']) > 0:
        suffix += f"AND instance IN ? "
        args = [[i for i in session_state['instance']]]
    return(suffix, args)        

@st.cache_data
def create_view_table(session_state):
    all_model = (len(session_state['model']) == 0 or len(session_state['model']) == NUMBER_OF_MODEL)
    all_explainer = (len(session_state['explainer']) == 0 or len(session_state['explainer']) == NUMBER_OF_EXPLAINER)
    all_metric = (len(session_state['metric']) == 0 or len(session_state['metric']) == NUMBER_OF_METRIC)

    number_of_model = 1 if all_model else len(session_state['model'])
    number_of_explainer = 1 if all_explainer else len(session_state['explainer'])
    number_of_metric = 1 if all_metric else len(session_state['metric'])

    partition_files = []
    for model in range(number_of_model):
        model_name = alleviate_model[session_state['model'][model]] if not all_model else "*"
        for explainer in range(number_of_explainer):
            explainer_name = alleviate_explainer[session_state['explainer'][explainer]] if not all_explainer else "*"
            for metric in range(number_of_metric):
                metric_name = alleviate_metric[session_state['metric'][metric]] if not all_metric else "*"
                filename = f".\\datasets\\streamlit_partition\\agg_fidelity_{model_name}_{alleviate_activation[session_state['activation']]}_{explainer_name}_{metric_name}.csv"
                partition_files.append(filename)
    union_query = " UNION ALL ".join([f"SELECT * FROM read_csv_auto('{file}', files_to_sniff=-1)" for file in partition_files])

    return union_query
            

# Transform dataframes into lists of dictionnaries for easier plotting
def transform_data(datatable):
    data = datatable.to_dict()
    transformed_data = [
        {key: value[row_id] for key, value in data.items()}
        for row_id in data['model'].keys()
    ]
    return transformed_data

# ──────────────────────────────────────
# Mainly part of the fast query function
def unalleviate_dataframe(dataframe, only_model=False, only_activation=False, only_explainer=False, only_metric=False):
    if not (only_activation or only_explainer or only_metric): dataframe['model'] = dataframe['model'].map(unalleviate_model)
    if not (only_model or only_explainer or only_metric): dataframe['activation'] = dataframe['activation'].map(unalleviate_activation)
    if not (only_model or only_activation or only_metric): dataframe['explainer'] = dataframe['explainer'].map(unalleviate_explainer)
    if not (only_model or only_activation or only_explainer): dataframe['metric'] = dataframe['metric'].map(unalleviate_metric)
    return dataframe


# Querys on a dataframe that take less space in memory are significanty faster
# The conversion from alleviated to unalleviated is abysmally fast (thanks dictionnaries)
@st.cache_data
def naive_fast_query(query, args, only_model=False, only_activation=False, only_explainer=False, only_metric=False):
    conn = db.connect()
    dataframe = conn.execute(query, args).fetchdf()
    conn.close()
    dataframe = unalleviate_dataframe(dataframe, only_model, only_activation, only_explainer, only_metric)
    return dataframe

@st.cache_data
def naive_execute_query(query, args):
    conn = db.connect()
    dataframe = conn.execute(query, args).fetchdf()
    conn.close()
    return dataframe

@st.cache_data
def fast_query(union_query, suffix, args):
    conn = db.connect()
    try:
        conn.execute(f"CREATE OR REPLACE VIEW agg_fidelity_view AS {union_query}")
    except Exception as e:
        st.error(f"Error creating: {e}")
        return db.query("SELECT * FROM '.\\datasets\\all_agg_fidelity.csv' LIMIT 0").to_df()
    query = f"SELECT * FROM agg_fidelity_view WHERE 1=1 {suffix}"
    dataframe = conn.execute(query, args).fetchdf()
    conn.close()
    dataframe = unalleviate_dataframe(dataframe)
    return dataframe

# ──────────────────────────────────────

