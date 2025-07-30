#!/bin/bash

if [ "$#" -eq 0 ]; then
    echo "Usage: it_preprocess.sh <input_filename>"
    echo "Example: it_preprocess.sh inference_time.csv"
    exit 1
fi

INPUT_FILE="$1"

if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

python csv_to_parquet.py "$INPUT_FILE"

if [ $? -eq 0 ]; then
    [ -f "$INPUT_FILE" ] && rm -f "$INPUT_FILE"
    
    PARQUET_FILE=$(basename "$INPUT_FILE" .csv).parquet
    python clean_and_round.py "$PARQUET_FILE"
    
    if [ $? -eq 0 ]; then
        [ -f "$PARQUET_FILE" ] && rm -f "$PARQUET_FILE"
    else
        exit 1
    fi
else
    exit 1
fi