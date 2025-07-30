@echo off

REM Check if an argument was provided
if "%1"=="" (
    echo Usage: preprocess.bat ^<input_filename^>
    echo Example: preprocess.bat my_custom_file.csv
    exit /b 1
)

REM Store the argument in a variable for clarity
set INPUT_FILE=%1

echo Starting data processing pipeline with input file: %INPUT_FILE%

REM Check if the input file exists
if not exist "%INPUT_FILE%" (
    echo ❌ Error: Input file "%INPUT_FILE%" not found
    exit /b 1
)

REM Extract filename without extension and create parquet filename
for /f "delims=" %%i in ("%INPUT_FILE%") do (
    set PARQUET_FILE=%%~ni.parquet
)

echo Step 1: Running alleviater.py with %INPUT_FILE%...
python csv_to_parquet.py "%INPUT_FILE%"

if %errorlevel%==0 (
    echo ✓ csv_to_parquet.py completed successfully
    echo ✓ Created %PARQUET_FILE%

    if exist "%INPUT_FILE%" (
        echo Deleting original %INPUT_FILE%...
        del /f /q "%INPUT_FILE%"
    )
    
    echo Step 2: Running normalize_parquet.py with %PARQUET_FILE%...
    python normalize_parquet.py "%PARQUET_FILE%"

    if %errorlevel%==0 (
        echo ✓ normalize_parquet.py completed successfully
        
        if exist "%PARQUET_FILE%" (
            echo Deleting "%PARQUET_FILE%"...
            del /f /q "%PARQUET_FILE%"
        )
        
        echo Step 3: Running parquet_to_mean_parquet.py...
        python parquet_to_mean_parquet.py

        if %errorlevel%==0 (
            echo ✓ parquet_to_mean_parquet.py completed successfully
            echo 🎉 Data processing pipeline completed!
        ) else (
            echo ❌ Error: parquet_to_mean_parquet.py failed with code %errorlevel%
            exit /b 1
        )
    ) else (
        echo ❌ Error: normalize_parquet.py failed with code %errorlevel%
        exit /b 1
    )
) else (
    echo ❌ Error: csv_to_parquet.py failed with code %errorlevel%
    exit /b 1
)