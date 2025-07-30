## 🚀 Run Locally

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd streamlitapp
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Process your data**
    #### Windows
    ```bash
    cd datasets
    preprocess.bat your_score_file.csv
    it_preprocess.bat your_training_time_file.csv
    ```
    #### Unix/Linux/max
    ```bash
    cd datasets
    chmod +x preprocess.sh
    chmod +x wb_preprocess.sh
    ./preprocess.sh your_score_file.csv
    ./it_preprocess.sh your_training_time_file.csv

4. **Run the application**
    ```bash
    streamlit run app.py
    ```

5. **Access the dashboard**
    - Open your browser and navigate to http://localhost:8501
    - The app should load with the Home Page displayed

### First-Time Setup

#### Data Preprocessing (Optional)

If you have the original agg_fidelity.csv and wb_inference_time.csv file and need to process them:
First be sure to place them in the datasets directory and that they are properly named agg_fidelity.csv and wb_inference_time.csv
(Note: The file will be deleted during the process so please, copy them elsewhere before this process if you wish to keep them)


### Development Setup

#### Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Unix/Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Running with Custom Port
```bash
streamlit run app.py --server.port 8052
```

#### Debug Mode
```bash
streamlit run app.py --logger.level debug
```

## 📁 Project Structure

```
streamlitapp/
├── 📁 datasets/
│   ├── 🐍 __init__.py                        # Necessary file for imports in the datasets repository from the main repository
│   ├── 📄 agg_fidelity.parquet               # Main file with all the data spanning accross all instances
│   ├── 📄 mean_agg_fidelity.parquet          # Aggregated means and std_dev by (metric, activation, model, explainer)
│   ├── 📄 inference_time.parquet             # File with inference times for given GPUS, models and explainers
│   ├── 🐍 csv_to_parquet.py                  # Processes your_file.csv -> your_file.parquet
│   ├── 🐍 clean_and_round.py                 # Processes your_inference_file.parquet -> inference_time.parquet
│   ├── 🐍 normalize_parquet.py               # Normalizes complexity and takes absolute values of data
│   ├── 🐍 parquet_to_parquet_mean.py         # Creates mean aggregations from parquet file
│   ├── 🔧 preprocess.sh                      # Unix/Linux preprocessing pipeline
│   ├── 🔧 preprocess.bat                     # Windows preprocessing pipeline
│   ├── 🔧 it_preprocess.sh                   # Unix/Linux preprocessing pipeline for training time csv
│   └── 🔧 it_preprocess.bat                  # Windows preprocessing pipeline for training time csv
│
├── 📁 pages/
│   ├── 🐍 gradient_data.py                   # Gradient score visualization
│   ├── 🐍 leaderboard.py                     # Explainer performance rankings
│   ├── 🐍 mean_scatter_map.py                # Mean and std_dev scatter plots
│   ├── 🐍 plotting_data.py                   # Interactive data plotting with AgGrid
│   ├── 🐍 rank_agreement.py                  # Agreement between explainer ranks for given metrics
│   ├── 🐍 subsample.py                       # Agreement between subsample and whole sample explainer ranks
│   └── 🐍 training_time.py                   # Performance vs Training time for different models
│
├── 📄 .gitignore                             # All csv files and streamlit_partition are ignored
├── 🐍 app.py                                 # Main Streamlit application entry point
├── 🐍 plotters.py                            # All plotting functions and visualizations
├── 🐍 utils.py                               # Parquet queries and utility functions
├── 📄 requirements.txt                       # Python dependencies
├── 🖼️ banner.png                             # Application banner image
└── 📖 README.md                              # This file
```

### File Descriptions

**Data Processing Pipeline:**
- `alleviater.py` - Abbreviates column names, rounds scores, normalizes Complexity metrics
- `streamlit_partitionner.py` - Splits data into smaller files for faster queries
- `mean_convert.py` - Calculates statistical aggregations and rankings
- `it_alleviater.py` - Abbreviates column names, round inference times

**Application Pages:**
- `gradient_data.py` - Feature importance visualization
- `leaderboard.py` - Performance rankings and comparisons
- `mean_scatter_map.py` - Statistical analysis over 50,000 ImageNet instances
- `plotting_data.py` - Interactive data exploration with manual row selection
- `rank_agreement.py`- Agreement between models on the relevance of explainers
- `subsample.py` - Just a little code snippet, the plots aren't really that revelant, I don't know honestly
- `training_time.py` - Performance compared to the necessary training time to achieve said performance


**Core Files:**
- `app.py` - Navigation, session state management, and page routing
- `plotters.py` - Plotly-based visualizations with dark mode support
- `utils.py` - Parquet queries, and filter logic

## ⚠️ Known Issues

### 🌙 Dark Mode
Currently, there's no automatic detection of the user's system theme. A manual toggle is provided since some plotting libraries don't natively support dark mode styling.

### 🎨 Gradient Score Data
- **Limited to 1000 rows** for performance reasons
- Gradient calculations are computationally expensive, causing significant loading times for larger datasets

### 📊 Mean Visualization 
- **Legend interactions are non-functional** - clicking legend items doesn't hide/show data series
- Colors represent explainers, shapes represent models, but Plotly limitations prevent proper interactive filtering

### 🏆 Leaderboard
- **No nested table support** in Streamlit-compatible libraries
- Alternative solutions (like Dash integration) would require running multiple servers simultaneously

### Subsample
- **I didn't have the time to really explore it** - but I feel like it could be an interesting way to analyze data, so I let the code snippet here

## 🛠️ Design Decisions

### Apply Filters Button
**Problem**: Streamlit reruns the entire app on every filter change, causing excessive loading times and potential crashes when multiple filters are applied simultaneously.

**Solution**: Manual filter application prevents unnecessary re-computations and provides better user experience with controlled data updates.

### Dark Mode Toggle
**Problem**: Some visualization libraries lack native dark mode support.

**Solution**: Manual toggle allows custom styling of plots for optimal readability in both light and dark themes.

### AgGrid for Data Selection
**Problem**: Native Streamlit dataframes don't support manual row selection.

**Solution**: AgGrid component enables interactive data selection for custom plotting subsets.

### Simplified Legends
**Challenge**: Displaying all explainer-model combinations would create overwhelming legends.

**Solution**: Dummy points provide color/shape references while maintaining legend readability, despite losing interactive functionality.

### Seaborn Color Palettes
**Challenge**: Providing comprehensible and user-friendly color palettes for plots

**Solution**: Seaborn provides color palettes adapted to colorblind people 

### mean_agg_fidelity.parquet
**Problem**: Querying the entire database to calculate means, std devs and ranks would take too much time to conduct every time something needs to be plotted

**Solution**: Create a file where data across all instances is already processed and querying on this file instead to cut loading times

### use of Parquet
**Issue**: Unless using partitions, queries on the csv files took too long to be a reliable and user friendly way to retrieve data

**Solution**: Parquet files are inherently compatible with pandas, and since files are treated as columns, columns of interest can be kept, furthermore, it allows to filter and ignores entirely the rows that don't correspond to the filter, which makes things extremly fast even for specific queries

**Consideration**: It is impossible to, like in SQL queries, put a limit to the number of rows scanned, thus if I want to have only 10000 rows of the datafile it's impossible by just putting a limit since the only way is to load everything and then take the first 10000 rows.
I eventually found a way to separate small queries and big queries and use a chunk based querying for the latter one, but the code is kind of hard to explain and interpret.
On top of al this, .parquet files can't be read by humans, unlike .csv

## 🔗 Resources

- [DEEL Official GitHub](https://github.com/deel-ai)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)