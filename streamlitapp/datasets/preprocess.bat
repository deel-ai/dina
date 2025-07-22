@echo off
echo Starting data processing pipeline...

echo Alt Step 1: Running wb_alleviater.py
python wb_alleviater.py
set WB_ALLEVIATER_RESULT=%errorlevel%

if "%WB_ALLEVIATER_RESULT%"=="0" (
    echo ✓ wb_alleviater.py completed successfully
    
    if exist wb_inference_time.csv (
        echo Deleting original wb_inference_time.csv...
        del /f /q wb_inference_time.csv
    )
) else (
    echo ❌ Error: wb_alleviater.py failed with code %WB_ALLEVIATER_RESULT%
    exit /b 1
)

echo Step 1: Running alleviater.py...
python alleviater.py
set ALLEVIATER_RESULT=%errorlevel%
echo Debug: alleviater.py returned error level %ALLEVIATER_RESULT%

if "%ALLEVIATER_RESULT%"=="0" (
    echo ✓ alleviater.py completed successfully
    
    if exist agg_fidelity.csv (
        echo Deleting original agg_fidelity.csv...
        del /f /q agg_fidelity.csv
    )
    
    echo Creating streamlit_partition directory...
    if not exist streamlit_partition (
        mkdir streamlit_partition
        echo ✓ streamlit_partition directory created
    ) else (
        echo ⚠️ streamlit_partition directory already exists
    )
    
    echo Step 2: Running streamlit_partitionner.py...
    python streamlit_partitionner.py
    set PARTITIONER_RESULT=%errorlevel%
    
    if "%PARTITIONER_RESULT%"=="0" (
        echo ✓ streamlit_partitionner.py completed successfully
        
        if exist all_agg_fidelity.csv (
            echo Deleting all_agg_fidelity.csv...
            del /f /q all_agg_fidelity.csv
        )
        
        echo Step 3: Running mean_convert.py...
        python mean_convert.py
        set CONVERT_RESULT=%errorlevel%
        
        if "%CONVERT_RESULT%"=="0" (
            echo ✓ mean_convert.py completed successfully
            echo 🎉 Data processing pipeline completed!
        ) else (
            echo ❌ Error: mean_convert.py failed with code %CONVERT_RESULT%
            exit /b 1
        )
    ) else (
        echo ❌ Error: streamlit_partitionner.py failed with code %PARTITIONER_RESULT%
        exit /b 1
    )
) else (
    echo ❌ Error: alleviater.py failed with code %ALLEVIATER_RESULT%
    exit /b 1
)


pause