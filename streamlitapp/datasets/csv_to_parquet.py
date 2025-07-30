import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import tempfile
import os

def convert_csv_to_parquet_memory_efficient(input_csv_path, output_parquet_path, chunk_size=100000):
    """
    Convert CSV file to Parquet format with minimal memory usage.
    Uses temporary parquet files and merges them at the end.
    
    Args:
        input_csv_path (str): Path to the input CSV file
        output_parquet_path (str): Path where the Parquet file will be saved
        chunk_size (int): Number of rows to process at once
    
    Returns:
        int: Total number of rows processed
    """
    
    print(f"Converting {input_csv_path} to {output_parquet_path}...")
    
    # Count total lines
    print("Counting total lines...")
    with open(input_csv_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for line in f) - 1
    
    print(f"Total rows to process: {total_lines:,}")
    
    # Create temporary directory for chunk files
    temp_dir = tempfile.mkdtemp()
    temp_files = []
    
    try:
        # Read CSV in chunks and save each as temporary parquet
        chunk_reader = pd.read_csv(input_csv_path, chunksize=chunk_size)
        total_rows = 0
        
        with tqdm(total=total_lines, desc="Processing chunks", unit="rows") as pbar:
            for chunk_num, chunk in enumerate(chunk_reader):
                # Save chunk as temporary parquet file
                temp_file = os.path.join(temp_dir, f"chunk_{chunk_num:04d}.parquet")
                chunk.to_parquet(temp_file, index=False)
                temp_files.append(temp_file)
                
                total_rows += len(chunk)
                pbar.update(len(chunk))
                
                # Clear chunk from memory
                del chunk
        
        print(f"Created {len(temp_files)} temporary files")
        
        # Merge temporary parquet files efficiently
        print("Merging temporary files...")
        
        # Read and combine temp files in batches to control memory
        batch_size = 10  # Process 10 files at a time
        final_chunks = []
        
        with tqdm(total=len(temp_files), desc="Merging files", unit="files") as pbar:
            for i in range(0, len(temp_files), batch_size):
                batch_files = temp_files[i:i + batch_size]
                
                # Read batch of temp files
                batch_chunks = []
                for temp_file in batch_files:
                    df_temp = pd.read_parquet(temp_file)
                    batch_chunks.append(df_temp)
                    pbar.update(1)
                
                # Combine batch
                if batch_chunks:
                    batch_combined = pd.concat(batch_chunks, ignore_index=True)
                    final_chunks.append(batch_combined)
                    
                    # Clear batch from memory
                    del batch_chunks
        
        # Final combination
        print("Final combination...")
        with tqdm(total=1, desc="Final merge", unit="operation") as pbar:
            df_final = pd.concat(final_chunks, ignore_index=True)
            pbar.update(1)
        
        # Write final parquet file
        print("Writing final parquet file...")
        with tqdm(total=1, desc="Writing output", unit="file") as pbar:
            df_final.to_parquet(output_parquet_path, index=False)
            pbar.update(1)
        
    finally:
        # Clean up temporary files
        print("Cleaning up temporary files...")
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass
    
    print(f"✓ Conversion complete! {total_rows:,} rows converted")
    return total_rows

def get_memory_usage_estimate(file_path):
    """Estimate memory usage for the file."""
    file_size_mb = file_path.stat().st_size / (1024**2)
    estimated_ram_mb = file_size_mb * 3  # Rough estimate: 3x file size in RAM
    return file_size_mb, estimated_ram_mb

def convert_massive_csv_to_parquet(input_csv_path, output_parquet_path, chunk_size=100000):
    """
    Convert massive CSV file to Parquet using PyArrow streaming.
    No temporary files, minimal memory usage.
    """
    
    print(f"Converting {input_csv_path} to {output_parquet_path}...")
    
    # Count total lines
    print("Counting total lines...")
    with open(input_csv_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for line in f) - 1
    
    print(f"Total rows to process: {total_lines:,}")
    
    # Read first chunk to get schema
    first_chunk = pd.read_csv(input_csv_path, nrows=chunk_size)
    schema = pa.Schema.from_pandas(first_chunk)
    
    # Create parquet writer
    parquet_writer = pq.ParquetWriter(output_parquet_path, schema)
    
    try:
        # Process CSV in chunks and write directly to parquet
        chunk_reader = pd.read_csv(input_csv_path, chunksize=chunk_size)
        total_rows = 0
        
        with tqdm(total=total_lines, desc="Converting", unit="rows") as pbar:
            for chunk in chunk_reader:
                # Convert chunk to arrow table and write
                table = pa.Table.from_pandas(chunk, schema=schema)
                parquet_writer.write_table(table)
                
                total_rows += len(chunk)
                pbar.update(len(chunk))
                
                # Clear chunk from memory
                del chunk, table
    
    finally:
        # Close the parquet writer
        parquet_writer.close()
    
    print(f"✓ Conversion complete! {total_rows:,} rows converted")
    return total_rows

def get_available_space(path):
    """Check available disk space."""
    import shutil
    total, used, free = shutil.disk_usage(path)
    return free / (1024**3)  # GB

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python csv_to_parquet.py <input_file.csv>")
        sys.exit(1)
    
    input_csv_path = Path(sys.argv[1])
    output_parquet_path = input_csv_path.with_suffix('.parquet')
    
    if not input_csv_path.exists():
        print(f"❌ Error: Input file {input_csv_path} does not exist.")
        sys.exit(1)
    
    # Check file size and available space
    file_size_gb = input_csv_path.stat().st_size / (1024**3)
    available_space_gb = get_available_space(input_csv_path.parent)
    
    print(f"📁 Input:  {input_csv_path}")
    print(f"💾 Output: {output_parquet_path}")
    print(f"📏 File size: {file_size_gb:.1f} GB")
    print(f"💿 Available space: {available_space_gb:.1f} GB")
    
    # Check if we have enough space (parquet is usually 60-80% smaller)
    estimated_output_size_gb = file_size_gb * 0.4  # Conservative estimate
    
    if estimated_output_size_gb > available_space_gb:
        print(f"❌ Error: Not enough disk space!")
        print(f"   Estimated output size: {estimated_output_size_gb:.1f} GB")
        print(f"   Available space: {available_space_gb:.1f} GB")
        sys.exit(1)
    
    # Choose approach based on file size
    if file_size_gb > 50:  # Files larger than 50GB
        print("🚀 Using streaming approach for massive file")
        try:
            total_rows = convert_massive_csv_to_parquet(input_csv_path, output_parquet_path)
        except ImportError:
            print("❌ PyArrow required for massive files: pip install pyarrow")
            sys.exit(1)
    else:
        print("📦 Using standard chunked approach")
        total_rows = convert_csv_to_parquet_memory_efficient(input_csv_path, output_parquet_path)
    
    print(f"🎉 Conversion completed! {total_rows:,} rows processed.")
    
    # File size comparison
    input_size_gb = file_size_gb
    output_size_gb = output_parquet_path.stat().st_size / (1000**3)
    compression_ratio = (1 - output_size_gb/input_size_gb) * 100
    
    print(f"📊 File size comparison:")
    print(f"   CSV:     {input_size_gb:.1f} GB")
    print(f"   Parquet: {output_size_gb:.1f} GB")
    print(f"   Saved:   {compression_ratio:.1f}% space ({input_size_gb-output_size_gb:.1f} GB)")