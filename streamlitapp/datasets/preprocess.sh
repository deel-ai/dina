#!/bin/bash

echo "Starting data processing pipeline..."

echo "Step 1: Running alleviater.py..."
python alleviater.py
ALLEVIATER_RESULT=$?
echo "Debug: alleviater.py returned error level $ALLEVIATER_RESULT"

if [ "$ALLEVIATER_RESULT" -eq 0 ]; then
    echo "✓ alleviater.py completed successfully"
    
    if [ -f "agg_fidelity.csv" ]; then
        echo "Deleting original agg_fidelity.csv..."
        rm -f agg_fidelity.csv
    fi
    
    echo "Creating streamlit_partition directory..."
    if [ ! -d "streamlit_partition" ]; then
        mkdir streamlit_partition
        echo "✓ streamlit_partition directory created"
    else
        echo "⚠️ streamlit_partition directory already exists"
    fi
    
    echo "Step 2: Running streamlit_partitionner.py..."
    python streamlit_partitionner.py
    PARTITIONER_RESULT=$?
    
    if [ "$PARTITIONER_RESULT" -eq 0 ]; then
        echo "✓ streamlit_partitionner.py completed successfully"
        
        if [ -f "all_agg_fidelity.csv" ]; then
            echo "Deleting all_agg_fidelity.csv..."
            rm -f all_agg_fidelity.csv
        fi
        
        echo "Step 3: Running mean_convert.py..."
        python mean_convert.py
        CONVERT_RESULT=$?
        
        if [ "$CONVERT_RESULT" -eq 0 ]; then
            echo "✓ mean_convert.py completed successfully"
            echo "🎉 Data processing pipeline completed!"
        else
            echo "❌ Error: mean_convert.py failed with code $CONVERT_RESULT"
            exit 1
        fi
    else
        echo "❌ Error: streamlit_partitionner.py failed with code $PARTITIONER_RESULT"
        exit 1
    fi
else
    echo "❌ Error: alleviater.py failed with code $ALLEVIATER_RESULT"
    exit 1
fi

echo "Alt Step 1: Running wb_alleviater.py"
python wb_alleviater.py
WB_ALLEVIATER_RESULT=$?

if [ "$WB_ALLEVIATER_RESULT" -eq 0 ]; then
    echo "✓ wb_alleviater.py completed successfully"
    
    if [ -f "wb_inference_time.csv" ]; then
        echo "Deleting original wb_inference_time.csv..."
        rm -f wb_inference_time.csv
    fi
else
    echo "❌ Error: wb_alleviater.py failed with code $WB_ALLEVIATER_RESULT"
    exit 1
fi


echo "Press any key to continue..."
read -n 1 -s