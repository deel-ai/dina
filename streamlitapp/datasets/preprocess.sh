#!/bin/bash

# Check if an argument was provided
if [ "$#" -eq 0 ]; then
    echo "Usage: preprocess.sh <input_filename>"
    echo "Example: preprocess.sh my_custom_file.csv"
    exit 1
fi

# Store the argument in a variable for clarity
INPUT_FILE="$1"

echo "Starting data processing pipeline with input file: $INPUT_FILE"

# Check if the input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

# Extract filename without extension and create parquet filename
PARQUET_FILE=$(basename "$INPUT_FILE" .csv).parquet

echo "Step 1: Running alleviater.py with $INPUT_FILE..."
python csv_to_parquet.py "$INPUT_FILE"

if [ $? -eq 0 ]; then
    echo "✓ csv_to_parquet.py completed successfully"
    echo "✓ Created $PARQUET_FILE"

    if [ -f "$INPUT_FILE" ]; then
        echo "Deleting original $INPUT_FILE..."
        rm -f "$INPUT_FILE"
    fi
    
    echo "Step 2: Running normalize_parquet.py with $PARQUET_FILE..."
    python normalize_parquet.py "$PARQUET_FILE"

    if [ $? -eq 0 ]; then
        echo "✓ normalize_parquet.py completed successfully"
        
        if [ -f "$PARQUET_FILE" ]; then
            echo "Deleting $PARQUET_FILE..."
            rm -f "$PARQUET_FILE"
        fi
        
        echo "Step 3: Running parquet_to_mean_parquet.py..."
        python parquet_to_mean_parquet.py

        if [ $? -eq 0 ]; then
            echo "✓ parquet_to_mean_parquet.py completed successfully"
            echo "🎉 Data processing pipeline completed!"
        else
            echo "❌ Error: parquet_to_mean_parquet.py failed with code $?"
            exit 1
        fi
    else
        echo "❌ Error: normalize_parquet.py failed with code $?"
        exit 1
    fi
else
    echo "❌ Error: csv_to_parquet.py failed with code $?"
    exit 1
fi