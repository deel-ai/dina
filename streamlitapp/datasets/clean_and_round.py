import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from pathlib import Path

def process_inference_time_chunked(input_parquet_path, output_parquet_path, chunk_size=100000):
    """
    Process inference time parquet file using chunked approach.
    """
    
    print(f"Processing inference time data in {input_parquet_path} (chunked approach)...")
    
    # Get total number of rows for progress tracking
    parquet_file = pq.ParquetFile(input_parquet_path)
    total_rows = parquet_file.metadata.num_rows
    print(f"Total rows to process: {total_rows:,}")
    
    processed_chunks = []
    total_processed = 0
    rows_dropped = 0
    
    # Create progress bar
    with tqdm(total=total_rows, desc="Processing rows", unit="rows") as pbar:
        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            df_batch = batch.to_pandas()
            
            if 'attribution' in df_batch.columns:
                df_batch = df_batch.rename(columns={'attribution': 'explainer'})

            # Count NaN rows before dropping
            nan_count = df_batch['mean_time'].isna().sum()
            rows_dropped += nan_count
            
            # Drop rows where mean_time is NaN and create explicit copy
            df_batch = df_batch.dropna(subset=['mean_time']).copy()  # Add .copy()
            
            # Round mean_time and std_time to 3 decimal places
            if 'mean_time' in df_batch.columns:
                df_batch.loc[:, 'mean_time'] = df_batch['mean_time'].round(3)  # Use .loc
            
            if 'std_time' in df_batch.columns:
                df_batch.loc[:, 'std_time'] = df_batch['std_time'].round(3)  # Use .loc
            
            if not df_batch.empty:
                processed_chunks.append(df_batch)
                total_processed += len(df_batch)
            
            pbar.update(len(batch.to_pandas()))
            
            # Save intermediate results every 50 chunks to manage memory
            if len(processed_chunks) >= 50:
                
                combined_chunk = pd.concat(processed_chunks, ignore_index=True)
                
                if total_processed <= len(combined_chunk):  # First batch
                    combined_chunk.to_parquet(output_parquet_path, index=False)
                else:  # Append to existing file
                    existing_df = pd.read_parquet(output_parquet_path)
                    final_df = pd.concat([existing_df, combined_chunk], ignore_index=True)
                    final_df.to_parquet(output_parquet_path, index=False)
                
                processed_chunks = []  # Clear memory
    
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
    
    print(f"\n✓ Processing complete!")
    print(f"  Rows dropped (NaN): {rows_dropped:,}")
    
    return total_processed

def process_inference_time_simple(input_parquet_path, output_parquet_path):
    """
    Process inference time parquet file using simple approach for smaller files.
    """
    
    print(f"Processing inference time data in {input_parquet_path} (simple approach)...")
    
    # Load data
    df = pd.read_parquet(input_parquet_path)

    if 'attribution' in df.columns:
        df = df.rename(columns={'attribution': 'explainer'})
    
    # Count NaN rows before dropping
    nan_count = df['mean_time'].isna().sum()
    print(f"Rows with NaN mean_time: {nan_count:,}")
    
    # Drop rows where mean_time is NaN and create explicit copy
    df_clean = df.dropna(subset=['mean_time']).copy()  # Add .copy() here
    print(f"Rows after dropping NaN: {len(df_clean):,}")
    
    # Round mean_time and std_time to 3 decimal places
    print("Rounding time columns...")
    if 'mean_time' in df_clean.columns:
        df_clean.loc[:, 'mean_time'] = df_clean['mean_time'].round(3)  # Use .loc
    
    if 'std_time' in df_clean.columns:
        df_clean.loc[:, 'std_time'] = df_clean['std_time'].round(3)  # Use .loc
    
    # Save to parquet
    print("Saving to parquet...")
    df_clean.to_parquet(output_parquet_path, index=False)
    
    print(f"✓ Processing complete!")
    
    return len(df_clean)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python process_inference_time.py <input_file.parquet>")
        print("Example: python process_inference_time.py wb_inference_time.parquet")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = "inference_time.parquet"
    
    if not Path(input_path).exists():
        print(f"❌ Error: Input file {input_path} does not exist.")
        sys.exit(1)
    
    # Check file size to determine approach
    file_size_gb = Path(input_path).stat().st_size / (1000**3)
    
    print(f"📁 Input:  {input_path}")
    print(f"💾 Output: {output_path}")
    print(f"📏 File size: {file_size_gb:.1f} GB")
    
    try:
        if file_size_gb > 1:  # Files larger than 1GB use chunked approach
            print("🚀 Using chunked approach for large file")
            total_rows = process_inference_time_chunked(input_path, output_path)
        else:
            print("📦 Using simple approach for smaller file")
            total_rows = process_inference_time_simple(input_path, output_path)
        
        print(f"🎉 Processing completed! {total_rows:,} rows in output")
        
        # Show file size comparison
        output_size_gb = Path(output_path).stat().st_size / (1000**3)
        
        print(f"📊 File size comparison:")
        print(f"   Input:  {file_size_gb:.1f} GB")
        print(f"   Output: {output_size_gb:.1f} GB")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)