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

3. **Run the application**
    ```bash
    streamlit run app.py
    ```

4. **Access the dashboard**
    - Open your browser and navigate to http://localhost:8501
    - The app should load with the Home Page displayed

### First-Time Setup

#### Data Preprocessing (Optional)

If you have the original agg_fidelity.csv file and need to process it:
First be sure to place it in the datasets directory and that it is properly named agg_fidelity.csv

#### Windows
```bash
cd datasets
preprocess.bat
```
#### Unix/Linux/max
```bash
cd datasets
chmod +x preprocess.sh
./preprocess.sh
```

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
│   ├── 📄 agg_fidelity.csv                    # Original dataset (backup)
│   ├── 📄 all_agg_fidelity.csv               # Processed dataset with abbreviated names
│   ├── 📄 all_mean_agg_fidelity.csv          # Aggregated means and std_dev by (metric, activation, model, explainer)
│   ├── 🐍 alleviater.py                      # Processes agg_fidelity.csv → all_agg_fidelity.csv
│   ├── 🐍 streamlit_partitionner.py          # Partitions data into separate files
│   ├── 🐍 mean_convert.py                    # Creates mean aggregations from partitions
│   ├── 🔧 preprocess.sh                      # Unix/Linux preprocessing pipeline
│   ├── 🔧 preprocess.bat                     # Windows preprocessing pipeline
│   └── 📁 streamlit_partition/
│       └── 📄 agg_fidelity_*_*_*_*.csv       # Partitioned files by (metric, activation, model, explainer)
│
├── 📁 pages/
│   ├── 🐍 plotting_data.py                   # Interactive data plotting with AgGrid
│   ├── 🐍 gradient_data.py                   # Gradient score visualization
│   ├── 🐍 mean_scatter_map.py                # Mean and std_dev scatter plots
│   ├── 🐍 heatmap.py                         # Correlation heatmaps
│   └── 🐍 leaderboard.py                     # Explainer performance rankings
│
├── 🐍 app.py                                  # Main Streamlit application entry point
├── 🐍 plotters.py                            # All plotting functions and visualizations
├── 🐍 utils.py                               # Database queries and utility functions
├── 📄 requirements.txt                       # Python dependencies
├── 🖼️ banner.png                             # Application banner image
└── 📖 README.md                              # This file
```

### File Descriptions

**Data Processing Pipeline:**
- `alleviater.py` - Abbreviates column names, rounds scores, normalizes Complexity metrics
- `streamlit_partitionner.py` - Splits data into smaller files for faster queries
- `mean_convert.py` - Calculates statistical aggregations and rankings

**Application Pages:**
- `plotting_data.py` - Interactive data exploration with manual row selection
- `gradient_data.py` - Feature importance visualization (limited to 100 rows)
- `mean_scatter_map.py` - Statistical analysis over 50,000 ImageNet instances
- `heatmap.py` - Standard deviation analysis across explainers/models
- `leaderboard.py` - Performance rankings and comparisons

**Core Files:**
- `app.py` - Navigation, session state management, and page routing
- `plotters.py` - Plotly-based visualizations with dark mode support
- `utils.py` - DuckDB queries, data transformations, and filter logic

## ⚠️ Known Issues

### 🌙 Dark Mode
Currently, there's no automatic detection of the user's system theme. A manual toggle is provided since some plotting libraries don't natively support dark mode styling.

### 🎨 Gradient Score Data
- **Limited to 100 rows** for performance reasons
- Gradient calculations are computationally expensive, causing significant loading times for larger datasets

### 📊 Mean Visualization 
- **Legend interactions are non-functional** - clicking legend items doesn't hide/show data series
- Colors represent explainers, shapes represent models, but Plotly limitations prevent proper interactive filtering

### 🏆 Leaderboard
- **No nested table support** in Streamlit-compatible libraries
- Alternative solutions (like Dash integration) would require running multiple servers simultaneously

## 🛠️ Design Decisions

### Apply Filters Button
**Problem**: Streamlit reruns the entire app on every filter change, causing excessive loading times and potential crashes when multiple filters are applied simultaneously.

**Solution**: Manual filter application prevents unnecessary re-computations and provides better user experience with controlled data updates.

### Dark Mode Toggle
**Requirement**: Some visualization libraries lack native dark mode support.

**Implementation**: Manual toggle allows custom styling of plots for optimal readability in both light and dark themes.

### AgGrid for Data Selection
**Limitation**: Native Streamlit dataframes don't support manual row selection.

**Solution**: AgGrid component enables interactive data selection for custom plotting subsets.

### Simplified Legends
**Challenge**: Displaying all explainer-model combinations would create overwhelming legends.

**Approach**: Dummy points provide color/shape references while maintaining legend readability, despite losing interactive functionality.

### Seaborn Color Palettes
**Advantage**: Built-in colorblind-friendly palettes improve accessibility.

**Consideration**: With 14+ explainers, color differentiation becomes challenging. Future improvement could map models to colors and explainers to shapes for better distinction.

## 🔗 Resources

- [DEEL Official GitHub](https://github.com/deel-ai)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)