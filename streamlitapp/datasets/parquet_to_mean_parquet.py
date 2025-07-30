import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import sys
from pathlib import Path

# WHICHISBETTER allows us to properly rank the explainers in regards of which metric is considered
WHICHISBETTER = {
    "Deletion": "Lower is better",
    "MuFidelity": "Higher is better", 
    "Sparseness": "Higher is better",
    "Complexity": "Lower is better",
    "Insertion": "Higher is better"
}

def calculate_mean_statistics_from_parquet(input_parquet_path, output_parquet_path, chunk_size=500000):
    """
    Calculate mean, std, min, max, and rank statistics from normalized parquet file.
    
    Args:
        input_parquet_path (str): Path to input normalized parquet file
        output_parquet_path (str): Path to output mean statistics parquet file
        chunk_size (int): Number of rows to process at once
    
    Returns:
        int: Number of unique combinations processed
    """
    
    print(f"Calculating statistics from {input_parquet_path}...")
    
    # Get total number of rows for progress tracking
    parquet_file = pq.ParquetFile(input_parquet_path)
    total_rows = parquet_file.metadata.num_rows
    print(f"Total rows to process: {total_rows:,}")
    
    # Process parquet file in chunks and collect all data
    print("Reading and aggregating data...")
    
    all_chunks = []
    rows_processed = 0
    
    with tqdm(total=total_rows, desc="Reading parquet", unit="rows") as pbar:
        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            df_batch = batch.to_pandas()
            all_chunks.append(df_batch)
            rows_processed += len(df_batch)
            pbar.update(len(df_batch))
    
    # Combine all chunks
    print("Combining all data...")
    df = pd.concat(all_chunks, ignore_index=True)
    del all_chunks  # Free memory
    
    print(f"Data loaded: {len(df):,} rows")
    
   # Calculate statistics grouped by model, activation, explainer, metric
    print("Calculating statistics...")
    
    # Group by the categorical columns and calculate stats
    grouped = df.groupby(['model', 'activation', 'explainer', 'metric'])['score']
    
    with tqdm(desc="Computing statistics", unit="groups") as pbar:
        # Fix: Use separate calls instead of nested dictionary
        stats_df = pd.DataFrame({
            'mean': grouped.mean(),
            'std_dev': grouped.std(),
            'min': grouped.min(),
            'max': grouped.max()
        }).reset_index()
        
        pbar.update(1)
    
    # Round statistics to 3 decimal places
    stats_df['mean'] = stats_df['mean'].round(3)
    stats_df['std_dev'] = stats_df['std_dev'].round(3)
    stats_df['min'] = stats_df['min'].round(3)
    stats_df['max'] = stats_df['max'].round(3)
    
    print(f"Calculated statistics for {len(stats_df):,} unique combinations")
    
    # Add rankings
    print("Adding rankings...")
    stats_df = add_rankings_to_dataframe(stats_df)
    
    # Save to parquet
    print("Saving to parquet...")
    with tqdm(desc="Writing parquet", unit="file") as pbar:
        stats_df.to_parquet(output_parquet_path, index=False)
        pbar.update(1)
    
    print(f"✓ Statistics calculation complete!")
    print(f"Output saved to: {output_parquet_path}")
    print(f"Processed {len(stats_df):,} unique combinations")
    
    return len(stats_df)

def add_rankings_to_dataframe(df):
    """
    Add explainer rankings for each model-activation-metric combination.
    
    Args:
        df (pd.DataFrame): DataFrame with mean statistics
        
    Returns:
        pd.DataFrame: DataFrame with added rank column
    """
    
    df['rank'] = 0
    
    # Get unique combinations of model, activation, and metric
    unique_combinations = df[['model', 'activation', 'metric']].drop_duplicates()
    
    print(f"Adding rankings for {len(unique_combinations):,} combinations...")
    
    with tqdm(total=len(unique_combinations), desc="Adding rankings", unit="combinations") as pbar:
        for _, row in unique_combinations.iterrows():
            model = row['model']
            activation = row['activation']
            metric = row['metric']
            
            # Filter rows for this specific combination
            mask = (df['model'] == model) & (df['activation'] == activation) & (df['metric'] == metric)
            combo_indices = df.index[mask].tolist()
            combo_means = df.loc[mask, 'mean'].values
            
            # Sort based on whether higher or lower is better for this metric
            if WHICHISBETTER[metric] == "Lower is better":
                sorted_order = sorted(range(len(combo_means)), key=lambda i: combo_means[i])
            else:  # Higher is better
                sorted_order = sorted(range(len(combo_means)), key=lambda i: -combo_means[i])
            
            # Assign ranks
            for rank, position in enumerate(sorted_order, 1):
                original_index = combo_indices[position]
                df.loc[original_index, 'rank'] = rank
            
            pbar.update(1)
    
    return df

def create_mean_statistics_simple(input_parquet_path, output_parquet_path):
    """
    Simple version that loads all data at once (for smaller files).
    
    Args:
        input_parquet_path (str): Path to input normalized parquet file
        output_parquet_path (str): Path to output mean statistics parquet file
    
    Returns:
        int: Number of unique combinations processed
    """
    
    print(f"Loading data from {input_parquet_path}...")
    
    # Load entire parquet file
    with tqdm(desc="Loading parquet", unit="file") as pbar:
        df = pd.read_parquet(input_parquet_path)
        pbar.update(1)
    
    print(f"Data loaded: {len(df):,} rows")
    
    # Calculate statistics
    print("Calculating statistics...")
    with tqdm(desc="Computing statistics", unit="operation") as pbar:
        # Fix: Use separate agg calls instead of nested dictionary
        grouped = df.groupby(['model', 'activation', 'explainer', 'metric'])['score']
        
        stats_df = pd.DataFrame({
            'mean': grouped.mean(),
            'std_dev': grouped.std(),
            'min': grouped.min(),
            'max': grouped.max(),
        }).reset_index()
        
        # Round to 3 decimal places
        stats_df['mean'] = stats_df['mean'].round(3)
        stats_df['std_dev'] = stats_df['std_dev'].round(3)
        stats_df['min'] = stats_df['min'].round(3)
        stats_df['max'] = stats_df['max'].round(3)
        
        pbar.update(1)
    
    print(f"Calculated statistics for {len(stats_df):,} unique combinations")
    
    # Add rankings
    stats_df = add_rankings_to_dataframe(stats_df)
    
    # Save to parquet
    print("Saving to parquet...")
    with tqdm(desc="Writing parquet", unit="file") as pbar:
        stats_df.to_parquet(output_parquet_path, index=False)
        pbar.update(1)
    
    print(f"✓ Statistics calculation complete!")
    print(f"Output saved to: {output_parquet_path}")
    
    return len(stats_df)

if __name__ == "__main__":
    
    input_path = "agg_fidelity.parquet"
    output_path = "mean_agg_fidelity.parquet"
    
    if not Path(input_path).exists():
        print(f"❌ Error: Input file {input_path} does not exist.")
        sys.exit(1)
    
    # Check file size to determine approach
    file_size_gb = Path(input_path).stat().st_size / (1000**3)
    
    print(f"📁 Input:  {input_path}")
    print(f"💾 Output: {output_path}")
    print(f"📏 File size: {file_size_gb:.1f} GB")
    
    try:
        if file_size_gb > 2:  # Files larger than 2GB use chunked approach
            print("🚀 Using chunked approach for large file")
            total_combinations = calculate_mean_statistics_from_parquet(input_path, output_path)
        else:
            print("📦 Using simple approach for smaller file")
            total_combinations = create_mean_statistics_simple(input_path, output_path)
        
        print(f"🎉 Processing completed! {total_combinations:,} combinations processed")
        
        # Show file size comparison
        input_size_gb = file_size_gb
        output_size_gb = Path(output_path).stat().st_size / (1000**3)
        
        print(f"📊 File size comparison:")
        print(f"   Input:  {input_size_gb:.1f} GB")
        print(f"   Output: {output_size_gb:.1f} GB")
        print(f"   Ratio:  {output_size_gb/input_size_gb:.1%} of original size")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        sys.exit(1)