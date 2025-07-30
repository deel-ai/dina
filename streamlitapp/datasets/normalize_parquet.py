import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from pathlib import Path

def normalize_parquet_complexity_chunked(input_parquet_path, output_parquet_path, chunk_size=100000):
    """
    Normalize complexity scores using chunked approach for large files.
    """
    
    print(f"Normalizing complexity scores in {input_parquet_path} (chunked approach)...")
    
    # Get total number of rows for progress tracking
    parquet_file = pq.ParquetFile(input_parquet_path)
    total_rows = parquet_file.metadata.num_rows
    print(f"Total rows to process: {total_rows:,}")
    
    # First pass: find max complexity scores for each (model, activation) pair
    print("First pass: Finding max complexity scores...")
    
    complexity_max = {}
    rows_processed = 0
    
    # Create progress bar for first pass
    with tqdm(total=total_rows, desc="Finding max complexity", unit="rows") as pbar:
        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            df_batch = batch.to_pandas()
            
            # Filter for complexity metric only
            complexity_batch = df_batch[df_batch['metric'] == 'Complexity'].copy()
            
            if not complexity_batch.empty:
                # Take absolute values
                complexity_batch['abs_score'] = complexity_batch['score'].abs()
                
                # Group by (model, activation) and find max
                max_scores = complexity_batch.groupby(['model', 'activation'])['abs_score'].max()
                
                for (model, activation), max_score in max_scores.items():
                    key = (model, activation)
                    complexity_max[key] = max(complexity_max.get(key, 0), max_score)
            
            rows_processed += len(df_batch)
            pbar.update(len(df_batch))
    
    print(f"Found {len(complexity_max)} unique (model, activation) pairs")
    
    # Second pass: normalize and process all data
    print("Second pass: Processing and normalizing data...")
    
    processed_chunks = []
    total_processed = 0
    
    # Reset the parquet file reader
    parquet_file = pq.ParquetFile(input_parquet_path)
    
    # Create progress bar for second pass
    with tqdm(total=total_rows, desc="Processing rows", unit="rows") as pbar:
        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            df_batch = batch.to_pandas()
            
            # Process scores vectorized for better performance
            def process_scores_vectorized(df):
                # Handle null scores
                mask_null = df['score'].isnull()
                
                # Process non-null scores
                mask_valid = ~mask_null
                result = df['score'].copy()
                
                # Process complexity scores
                complexity_mask = (df['metric'] == 'Complexity') & mask_valid
                if complexity_mask.any():
                    for idx in df[complexity_mask].index:
                        model = df.loc[idx, 'model']
                        activation = df.loc[idx, 'activation']
                        key = (model, activation)
                        max_val = complexity_max.get(key, 1)
                        abs_score = abs(df.loc[idx, 'score'])
                        result.loc[idx] = round(abs_score / max_val, 3) if max_val > 0 else round(abs_score, 3)
                
                # Process non-complexity scores
                non_complexity_mask = (df['metric'] != 'Complexity') & mask_valid
                if non_complexity_mask.any():
                    result.loc[non_complexity_mask] = df.loc[non_complexity_mask, 'score'].abs().round(3)
                
                return result
            
            # Apply processing
            df_batch['score'] = process_scores_vectorized(df_batch)
            
            processed_chunks.append(df_batch)
            total_processed += len(df_batch)
            pbar.update(len(df_batch))
            
            # Save intermediate results every 50 chunks to manage memory
            if len(processed_chunks) >= 50:
                print(f"\nSaving intermediate results... ({total_processed:,} rows processed)")
                
                combined_chunk = pd.concat(processed_chunks, ignore_index=True)
                
                if total_processed <= len(combined_chunk):  # First batch
                    combined_chunk.to_parquet(output_parquet_path, index=False)
                else:  # Append to existing file
                    existing_df = pd.read_parquet(output_parquet_path)
                    final_df = pd.concat([existing_df, combined_chunk], ignore_index=True)
                    final_df.to_parquet(output_parquet_path, index=False)
                
                processed_chunks = []  # Clear memory
                
                # Update progress bar description
                pbar.set_postfix({'saved': f"{total_processed:,} rows"})
    
    # Save any remaining chunks
    if processed_chunks:
        print(f"\nSaving final chunks... ({total_processed:,} rows total)")
        combined_chunk = pd.concat(processed_chunks, ignore_index=True)
        
        if total_processed <= len(combined_chunk):  # Only chunk
            combined_chunk.to_parquet(output_parquet_path, index=False)
        else:  # Append to existing
            existing_df = pd.read_parquet(output_parquet_path)
            final_df = pd.concat([existing_df, combined_chunk], ignore_index=True)
            final_df.to_parquet(output_parquet_path, index=False)
    
    return total_processed

def normalize_parquet_complexity_simple(input_parquet_path, output_parquet_path):
    """
    Normalize complexity scores using simple approach for smaller files.
    """
    
    print(f"Normalizing complexity scores in {input_parquet_path} (simple approach)...")
    
    # Load entire parquet file
    print("Loading data...")
    with tqdm(desc="Loading parquet", unit="file") as pbar:
        df = pd.read_parquet(input_parquet_path)
        pbar.update(1)
    
    # First pass: find max complexity scores
    print("Finding max complexity scores...")
    complexity_df = df[df['metric'] == 'Complexity'].copy()
    
    if not complexity_df.empty:
        complexity_df['abs_score'] = complexity_df['score'].abs()
        complexity_max = complexity_df.groupby(['model', 'activation'])['abs_score'].max().to_dict()
    else:
        complexity_max = {}
    
    print(f"Found {len(complexity_max)} unique (model, activation) pairs")
    
    # Second pass: normalize all data
    print("Processing and normalizing data...")
    
    def process_score_simple(row):
        if pd.isna(row['score']):
            return None
        
        abs_score = abs(float(row['score']))
        
        if row['metric'] == 'Complexity':
            key = (row['model'], row['activation'])
            max_val = complexity_max.get(key, 1)
            final_score = round(abs_score / max_val, 3) if max_val > 0 else round(abs_score, 3)
        else:
            final_score = round(abs_score, 3)
        
        return final_score
    
    # Apply processing with progress bar
    tqdm.pandas(desc="Processing scores")
    df['score'] = df.progress_apply(process_score_simple, axis=1)
    
    # Save to parquet
    print("Saving to parquet...")
    with tqdm(desc="Writing parquet", unit="file") as pbar:
        df.to_parquet(output_parquet_path, index=False)
        pbar.update(1)
    
    return len(df)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python normalize_parquet.py <input_file.parquet>")
        print("Example: python normalize_parquet.py all_fidelity_data.parquet")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = "agg_fidelity.parquet"
    
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
            total_rows = normalize_parquet_complexity_chunked(input_path, output_path)
        else:
            print("📦 Using simple approach for smaller file")
            total_rows = normalize_parquet_complexity_simple(input_path, output_path)
        
        print(f"🎉 Normalization completed! {total_rows:,} rows processed")
        
        # Show file size comparison
        input_size_gb = file_size_gb
        output_size_gb = Path(output_path).stat().st_size / (1000**3)
        compression_ratio = (1 - output_size_gb/input_size_gb) * 100
        
        print(f"📊 File size comparison:")
        print(f"   Input:  {input_size_gb:.1f} GB")
        print(f"   Output: {output_size_gb:.1f} GB")
        print(f"   Saved:  {compression_ratio:.1f}% space")
        
    except Exception as e:
        print(f"❌ Error during normalization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)