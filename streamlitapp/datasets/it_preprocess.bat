@echo off

REM Check if an argument was provided
if "%1"=="" (
    echo Usage: wb_preprocess.bat ^<input_filename^>
    echo Example: wb_preprocess.bat wb_inference_time.csv
    exit /b 1
)

REM Store the argument in a variable for clarity
set INPUT_FILE=%1

echo Starting weights & biases data processing pipeline with input file: %INPUT_FILE%

REM Check if the input file exists
if not exist "%INPUT_FILE%" (
    echo ❌ Error: Input file "%INPUT_FILE%" not found
    exit /b 1
)

REM Extract filename without extension and create parquet filename
for /f "delims=" %%i in ("%INPUT_FILE%") do (
    set PARQUET_FILE=%%~ni.parquet
)

echo Alt Step: Running csv_to_parquet.py with %INPUT_FILE%...
python csv_to_parquet.py "%INPUT_FILE%"

if %errorlevel%==0 (
    echo ✓ csv_to_parquet.py completed successfully
    echo ✓ Created %PARQUET_FILE%
    
    if exist "%INPUT_FILE%" (
        echo Deleting original %INPUT_FILE%...
        del /f /q "%INPUT_FILE%"
    )

    echo Alt Step 2: Running clean_and_round.py with %PARQUET_FILE%...
    python clean_and_round.py "%PARQUET_FILE%"

    if %errorlevel%==0 (
        echo ✓ clean_and_round.py completed successfully
        
        if exist "%PARQUET_FILE%" (
            echo Deleting "%PARQUET_FILE%"...
            del /f /q "%PARQUET_FILE%"
        )
        echo 🎉 Completed!
    ) else (
        echo ❌ Error: clean_and_round.py failed with code %errorlevel%
        exit /b 1
    )
) else (
    echo ❌ Error: wb_alleviater.py failed with code %errorlevel%
    exit /b 1
)

pause