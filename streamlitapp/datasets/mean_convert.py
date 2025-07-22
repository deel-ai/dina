import duckdb as db
import pandas as pd
from tqdm import tqdm
import csv
import threading
import queue
from alleviater import alleviate_model, alleviate_activation, alleviate_explainer, alleviate_metric
import sys
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# This script converts aggregated fidelity into a csv containing the mean, standard deviation, min and max of 
# the score for each model, activation, explainer and metric combination.
#
# The multithreading is just for the heck of it, it makes us gain maybe a few seconds on the minutes needed to run the script.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

WHICHISBETTER = {
    "D": "Lower is better",
    "F": "Higher is better", 
    "S": "Higher is better",
    "C": "Lower is better",
    "I": "Higher is better"
}

def query_data(model, activation, explainer, metric, conn):
    query = f"SELECT * FROM '.\\streamlit_partition\\agg_fidelity_{model}_{activation}_{explainer}_{metric}.csv'"
    scoreset = conn.execute(query).fetchdf()
    return scoreset

def process_and_write_data(data, writer):
    row = {
        'model': data['model'][0] if not data.empty else None,
        'activation': data['activation'][0] if not data.empty else None,
        'explainer': data['explainer'][0] if not data.empty else None,
        'metric': data['metric'][0] if not data.empty else None,
        'mean': round(float(data['score'].mean()), 3) if not data.empty else None,
        'std_dev': round(float(data['score'].std()), 3) if not data['score'].empty else None,
        'min': round(float(data['score'].min()), 3) if not data['score'].empty else None,
        'max': round(float(data['score'].max()), 3) if not data['score'].empty else None,
        'rank': None
    }
    if not data.empty:
        writer.writerow(row)

def thread1(model_list, activation_list, explainer_list, metric_list, rows_left):
    conn = db.connect()
    for model in model_list:
        for activation in activation_list:
            for explainer in explainer_list:
                for metric in metric_list:
                    data = query_data(model, activation, explainer, metric, conn)
                    data['model'] = model
                    data['activation'] = activation
                    data['explainer'] = explainer
                    data['metric'] = metric
                    rows_left.put(data)

def thread2(global_len, rows_left, writer):
    for _ in tqdm(range(global_len)):
        data = rows_left.get()
        process_and_write_data(data, writer)

def add_rankings(filename):
    """Add rankings between explainers for each (model, activation, metric) combination"""
    import pandas as pd
    
    # Read the CSV file
    df = pd.read_csv(filename)
    
    # Initialize rank column
    df['rank'] = 0
    
    # Get unique combinations of model, activation, and metric
    unique_combinations = df[['model', 'activation', 'metric']].drop_duplicates()
    
    for _, row in unique_combinations.iterrows():
        model = row['model']
        activation = row['activation']
        metric = row['metric']
        
        # Filter rows for this specific combination
        mask = (df['model'] == model) & (df['activation'] == activation) & (df['metric'] == metric)
        combo_indices = df.index[mask].tolist()
        combo_means = df.loc[mask, 'mean'].values
        
        # Sort by mean score (ascending for some metrics, descending for others)
        if metric in ['D', 'S', 'C']:  # Deletion, Sparseness, Complexity - lower is better
            sorted_order = sorted(range(len(combo_means)), key=lambda i: combo_means[i])
        else:  # Insertion, MuFidelity - higher is better
            sorted_order = sorted(range(len(combo_means)), key=lambda i: -combo_means[i])
        
        # Assign ranks (1-based) between explainers
        for rank, position in enumerate(sorted_order, 1):
            original_index = combo_indices[position]
            df.loc[original_index, 'rank'] = rank
    
    # Save back to the same file
    df.to_csv(filename, index=False)

def main(model_list, activation_list, explainer_list, metric_list, testmode=False):
    filename = 'all_mean_agg_fidelity.csv' if not testmode else 'all_mean_agg_fidelity_t.csv'

    with open(filename, mode='w', encoding='utf-8', newline='') as outfile:
        headers = ['model', 'activation', 'explainer', 'metric', 'mean', 'std_dev', 'min', 'max', 'rank']
        writer = csv.DictWriter(outfile, fieldnames=headers)
        writer.writeheader()

        global_len = len(model_list) * len(activation_list) * len(explainer_list) * len(metric_list)

        rows_left = queue.Queue()

        secondary_thread = threading.Thread(target = thread2, daemon = True, args=(global_len, rows_left, writer))
        secondary_thread.start()
        thread1(model_list, activation_list, explainer_list, metric_list, rows_left)
        secondary_thread.join()
    add_rankings(filename)


def convert_mean_fidelity(testmode = False):
    model_list = alleviate_model.values()
    activation_list = alleviate_activation.values()
    explainer_list = alleviate_explainer.values()
    metric_list = alleviate_metric.values()
    
    ### For debugging purposes, you can limit the lists to a smaller set
    model_list_t = ['P']
    activation_list_t = ['O']
    explainer_list_t = ['R', 'K', 'H']
    metric_list_t = ['D', 'F', 'C']
    if testmode:
        (model_list, activation_list, explainer_list, metric_list) = (model_list_t, activation_list_t, explainer_list_t, metric_list_t)
    main(model_list, activation_list, explainer_list, metric_list, testmode=testmode)


if __name__ == "__main__":
    try:
        convert_mean_fidelity() # Pass testmode = True as a parameter to run a test conversion on a smaller dataframe
        print("✅ mean_convert.py completed successfully")
        sys.exit(0)
    except Exception as e:
        print(f"❌ mean_convert.py failed: {e}")
        sys.exit(1)