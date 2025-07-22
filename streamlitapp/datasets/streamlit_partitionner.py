import csv
from tqdm import tqdm
from alleviater import unalleviate_activation, unalleviate_explainer, unalleviate_model, unalleviate_metric
import sys

# This takes an abysmally small amount of time to run 
def count_lines(file_path):
    """Counts the number of lines in a file."""
    with open(file_path, mode='r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def partition_agg_fidelity(input_file, output_dir):

    # Open the input file for reading
    with open(input_file, mode="r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)

        file_len = count_lines(input_file)

        # Create a dictionary to manage file handles and writers for partitions
        partition_files = {}
        partition_writers = {}

        for model in unalleviate_model.keys():
            for activation in unalleviate_activation.keys():
                for explainer in unalleviate_explainer.keys():
                    for metric in unalleviate_metric.keys():
                        # Create the corresponding file
                        partition_key = f"{model}_{activation}_{explainer}_{metric}"
                        partition_file_path = f"{output_dir}/agg_fidelity_{partition_key}.csv"
                        partition_files[partition_key] = open(partition_file_path, mode="w", encoding="utf-8", newline="")
                        partition_writers[partition_key] = csv.DictWriter(partition_files[partition_key], fieldnames=reader.fieldnames)
                        partition_writers[partition_key].writeheader()
        

        # Process each row
        for row in tqdm(reader, total=file_len, desc="Partitioning rows"):
            # Determine the partition key based on model, activation, explainer, and metric
            partition_key = f"{row['model']}_{row['activation']}_{row['explainer']}_{row['metric']}"

            # Write the row to the appropriate partition file
            partition_writers[partition_key].writerow(row)

        # Close all partition files
        for file in partition_files.values():
            file.close()


# Example usage

if __name__ == "__main__":
    try:
        partition_agg_fidelity("all_agg_fidelity.csv", "streamlit_partition")
        print("✅ streamlit_partitionner.py completed successfully")
        sys.exit(0)
    except Exception as e:
        print(f"❌ streamlit_partitionner.py failed: {e}")
        sys.exit(1)