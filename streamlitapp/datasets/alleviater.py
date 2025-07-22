import csv
from tqdm import tqdm
import sys


# ──────────────Just the correspondance tables───────────
unalleviate_model = {
    'P': 'MLPMixer',
    'D': 'DinoV2',
    'R': 'ResNest50',
    'M': 'MaxVIT',
    'C': 'ConvNeXtV2Base',
    'B': 'BeitV2',
    'E': 'EfficientNetV2',
    'I': 'InceptionNeXt',
    'N': 'ResNet50',
    'X': 'ConvNeXtV2'
}
unalleviate_activation = {
    'S': 'sigmoid',
    'O': 'softmax',
    'L': 'logits',
}
unalleviate_explainer = {
    'R': 'Rise',
    'K': 'KernelShap',
    'H': 'HsicAttributionMethod',
    'I': 'IntegratedGradients',
    'L': 'Lime',
    'S': 'Saliency',
    'O': 'Occlusion',
    'V': 'VarGrad',
    'G': 'GradCAM',
    'M': 'SmoothGrad',
    'N': 'GradientInput',
    'P': 'GradCAMPP',
    'Q': 'SquareGrad',
    'B': 'SobolAttributionMethod'
}
unalleviate_metric = {
    'D': 'Deletion',
    'F': 'MuFidelity',
    'S': 'Sparseness',
    'C': 'Complexity',
    'I': 'Insertion'
}
alleviate_model = {v: k for k, v in unalleviate_model.items()}
alleviate_activation = {v: k for k, v in unalleviate_activation.items()}
alleviate_explainer = {v: k for k, v in unalleviate_explainer.items()}
alleviate_metric = {v: k for k, v in unalleviate_metric.items()}
# ─────────────────────────────────────────────────────────────


# This takes an abysmally small amount of time to run 
def count_lines(file_path):
    """Counts the number of lines in a file."""
    with open(file_path, mode='r', encoding='utf-8') as f:
        return sum(1 for _ in f)
    
# ───────────────────────────────────────────────────────────────────────────────
# At ~60k rows per second, the run may take a bit of time, but it is not that bad
# ───────────────────────────────────────────────────────────────────────────────
def alleviate_csv_line_by_line(input_file, output_file):
    nb_of_lines= count_lines(input_file)

    # First pass: find max Complexity scores per (model, activation)
    print("First pass: analyzing Complexity scores...")
    complexity_max = {}
        
    with open(input_file, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
            
        for row in tqdm(reader, desc="Finding max complexity scores", total=nb_of_lines):
            if row['metric'] == 'Complexity' and row['score']:
                model = row['model']
                activation = row['activation']
                score = abs(float(row['score']))
                key = (model, activation)
                complexity_max[key] = max(complexity_max.get(key, 0), score)
        
    # Second pass: process and normalize
    print("Second pass: processing and normalizing data...")
        
    with open(input_file, mode='r', encoding='utf-8') as infile, \
        open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
            
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=['model', 'activation', 'explainer', 'metric', 'instance', 'label', 'prediction', 'score'])
        writer.writeheader()
            
        for row in tqdm(reader, desc="Processing rows", total=nb_of_lines):
            # Process score
            if row['score']:
                score = float(row['score'])
                abs_score = abs(score)
                    
                if row['metric'] == 'Complexity':
                    key = (row['model'], row['activation'])
                    max_val = complexity_max.get(key, 1)
                    final_score = round(abs_score / max_val, 3) if max_val > 0 else round(abs_score, 3)
                else:
                    final_score = round(abs_score, 3)
            else:
                final_score = None
                
            # Write row
            filtered_row = {
                'model': alleviate_model[row['model']],
                'activation': alleviate_activation[row['activation']],
                'explainer': alleviate_explainer[row['explainer']],
                'metric': alleviate_metric[row['metric']],
                'instance': row['instance'],
                'label': row['label'],
                'prediction': row['prediction'],
                'score': final_score
            }
            writer.writerow(filtered_row)
            
if __name__ == "__main__":
    try:
        input_file = "agg_fidelity.csv"
        output_file = "all_agg_fidelity.csv"
        alleviate_csv_line_by_line(input_file, output_file)
        print("✅ alleviater.py completed successfully")
        sys.exit(0)
    except Exception as e:
        print(f"❌ alleviater.py failed: {e}")
        sys.exit(1)