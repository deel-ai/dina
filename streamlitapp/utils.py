# ───────────────────────────────────────────────────────────────────────────
# This file contains utility functions for database querying and data processing
# ───────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
from pathlib import Path
# ─────────────────────────────────────────────────────────────────────────────────────
# Constants defining available options for filters and data organization
# ─────────────────────────────────────────────────────────────────────────────────────
METRIC_LIST = ["Insertion", "Deletion", "MuFidelity", "Sparseness", "Complexity"] # Available metrics for explainability evaluation
EXPLAINER_LIST = ["Rise", "KernelShap", "HsicAttributionMethod", "IntegratedGradients", "Lime", 
                  "Saliency", "Occlusion", "VarGrad", "GradCAM", "SmoothGrad",
                  "GradientInput", "GradCAMPP", "SquareGrad", "SobolAttributionMethod", "Laplace",
                  "Sobel", "RandomExp"] # Available explainer methods
MODEL_LIST = ["BeitV2", "DinoV2", "MaxVIT", "ResNest50","EfficientNetV2", "InceptionNeXt", "ConvNeXtV2Base", "MLPMixer", ] # Available deep learning models
ACTIVATION_LIST = ["softmax", "sigmoid", "logits"] # Available activation functions for model outputs

# Counts for optimization and file partitioning logic
NUMBER_OF_MODEL = len(MODEL_LIST)
NUMBER_OF_EXPLAINER = len(EXPLAINER_LIST)
NUMBER_OF_METRIC = len(METRIC_LIST)
NUMBER_OF_ACTIVATION = len(ACTIVATION_LIST)
NUMBER_OF_INSTANCE = 50000
NUMBER_OF_BATCH = 6
NUMBER_OF_GPUS = 2

# ─────────────────────────────────────────────────────────────────────────────────────
# Dynamic query building functions for filtering data based on user selections
# ─────────────────────────────────────────────────────────────────────────────────────

def apply_label_prediction_filter(df, filter_option):
    """
    Apply label/prediction filtering to the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        filter_option (str): Filter option from selectbox
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    if filter_option == "Display only good predictions":
        return df[df['label'] == df['prediction']]
    elif filter_option == "Display only wrong predictions":
        return df[df['label'] != df['prediction']]
    else:  # "Display all"
        return df

@st.cache_data
def query_parquet_data(selected_filters, max_rows=10000, skip_metric=False, skip_instance=False, skip_explainer=False, skip_model=False, skip_label_prediction=False, subsample=False, columns=[], unique_metric=False, unique_model=False, mean_values=False, inference_values=False):
    """
    Query Parquet data based on selected filters with optimized reading strategies.
    This function reads a Parquet file and applies the selected filters to return a DataFrame.
    It uses either direct reading with PyArrow filters for good selectivity or sequential reading with early stopping for poor selectivity.
    Args:
        selected_filters (dict): Dictionary of selected filters with keys like 'model', 'explainer', 'metric', 'instance', etc.
        max_rows (int): Maximum number of rows to return
        skip_metric (bool): Skip metric filter
        skip_instance (bool): Skip instance filter
        skip_explainer (bool): Skip explainer filter
        skip_model (bool): Skip model filter
        skip_label_prediction (bool): Skip label/prediction filter
        subsample (bool): Use subsample instances instead of all
        columns (list): Specific columns to read from the Parquet file
        unique_metric (bool): If True, only return data for a single metric
        unique_model (bool): If True, only return data for a single model
        mean_values (bool): If True, read mean values from a different Parquet file
        inference_values (bool): If True, read inference time data
    Returns:
        pd.DataFrame: Filtered DataFrame with the requested data
    """
    
    modified_filters = selected_filters.copy()
    
    parquet_path = ".\\datasets\\agg_fidelity.parquet" 
    
    if mean_values:
        parquet_path = ".\\datasets\\mean_agg_fidelity.parquet"
        skip_instance = True
        skip_label_prediction = True
        
    elif inference_values:
        parquet_path = ".\\datasets\\inference_time.parquet"
        skip_instance=True
        skip_label_prediction=True
        skip_metric=True
        modified_filters.pop('activation', None)

    if skip_model:
        modified_filters.pop('model', None)
    if skip_label_prediction:
        modified_filters.pop('label_prediction', None) 
    if skip_explainer:
        modified_filters.pop('explainer', None)
    if skip_metric:
        modified_filters.pop('metric', None)
    if skip_instance:
        modified_filters.pop('instance', None)
    if subsample:
        modified_filters['instance'] = modified_filters.get('subsample', [])
    if unique_metric:
        modified_filters['metric'] = [modified_filters['unique_metric']]
    if unique_model:
        modified_filters['model'] = [modified_filters['unique_model']]
        
    
    # Calculate selectivity with modified filters
    number_of_model_filters = len(modified_filters['model']) if modified_filters.get('model') else NUMBER_OF_MODEL
    number_of_explainer_filters = len(modified_filters['explainer']) if modified_filters.get('explainer') else NUMBER_OF_EXPLAINER
    number_of_metric_filters = len(modified_filters['metric']) if modified_filters.get('metric') else NUMBER_OF_METRIC
    number_of_instance_filters = len(modified_filters['subsample']) if subsample else len(modified_filters['instance']) if modified_filters.get('instance') else NUMBER_OF_INSTANCE
    number_of_matching_lines = number_of_model_filters * number_of_explainer_filters * number_of_metric_filters * number_of_instance_filters 
    if mean_values:
        number_of_matching_lines = number_of_model_filters * number_of_explainer_filters * number_of_metric_filters
    elif inference_values:
        number_of_matching_lines = number_of_model_filters * number_of_explainer_filters * NUMBER_OF_GPUS * NUMBER_OF_BATCH
    if number_of_matching_lines > max_rows:
        print(number_of_matching_lines, number_of_instance_filters)
        st.warning(f"⚠️ Too many rows found, showing {max_rows:,}")
    # Decision logic
    
    if number_of_matching_lines < 5000000: # Threshold for good selectivity, it is arbitrary and can be adjusted
        # Good selectivity - use direct reading with PyArrow filters
        info = st.info("✨ Using optimized reading (good filter selectivity)")
        df = query_parquet_data_direct(parquet_path, modified_filters, max_rows, columns)
        info.empty()
        return df
    
    else:
        # Poor selectivity - use sequential reading with early stopping
        info = st.info("🔄 Using sequential reading (scanning for matches)")
        
        df =  query_parquet_data_sequential(parquet_path, modified_filters, max_rows, columns=columns)
        
        info.empty()
        return df

def query_parquet_data_direct(parquet_path, selected_filters, max_rows=10000, columns = []):
    """
    Direct reading with PyArrow filters for good selectivity.
    This function reads a Parquet file using PyArrow with filters applied.
    It is optimized for cases where the filters are selective enough to avoid reading the entire file.
    Args:
        parquet_path (str): Path to the Parquet file
        selected_filters (dict): Dictionary of selected filters with keys like 'model', 'explainer', 'metric', 'instance', etc.
        max_rows (int): Maximum number of rows to return
        columns (list): Specific columns to read from the Parquet file
    Returns:
        pd.DataFrame: Filtered DataFrame with the requested data
    """
    
    if not Path(parquet_path).exists():
        st.error(f"Parquet file not found: {parquet_path}")
        return pd.DataFrame()
    
    try:
        import pyarrow.parquet as pq
        
        # Create PyArrow filters
        filters = create_pyarrow_filters(selected_filters)
        
        with st.spinner("Reading filtered data..."):
            # Read with filters applied
            if filters:
                if columns != []:
                    table = pq.read_table(parquet_path, filters=filters, columns=columns)
                else:
                    table = pq.read_table(parquet_path, filters=filters)
                df = table.to_pandas()
            else:
                # Fallback to sequential if no PyArrow filters
                return query_parquet_data_sequential(selected_filters, max_rows, columns)
        
        # Apply label/prediction filter (can't be done at PyArrow level)
        label_pred_filter = selected_filters.get('label_prediction', 'Display all')
        df = apply_label_prediction_filter(df, label_pred_filter)
        
        # Limit rows if needed
        if len(df) > max_rows:
            df = df.sample(n = max_rows)
        
        return df
        
    except Exception as e:
        st.error(f"Error in direct reading: {e}")
        # Fallback to sequential approach
        return query_parquet_data_sequential(selected_filters, max_rows)

def query_parquet_data_sequential(parquet_path, selected_filters, max_rows=10000, chunk_size=10000, columns=[]):
    """
    Sequential reading of Parquet data with early stopping.
    This function reads a Parquet file in chunks and applies filters to each chunk.
    It is optimized for cases where the filters are not selective enough to read the entire file at once.
    Args:
        parquet_path (str): Path to the Parquet file
        selected_filters (dict): Dictionary of selected filters with keys like 'model', 'explainer', 'metric', 'instance', etc.
        max_rows (int): Maximum number of rows to return
        chunk_size (int): Size of each chunk to read from the Parquet file
        columns (list): Specific columns to read from the Parquet file
    Returns:
        pd.DataFrame: Filtered DataFrame with the requested data
    """

    if not Path(parquet_path).exists():
        st.error(f"Parquet file not found: {parquet_path}")
        return pd.DataFrame()
    
    try:
        import pyarrow.parquet as pq
        import time
        
        parquet_file = pq.ParquetFile(parquet_path)
        
        # Calculate total number of chunks
        total_rows = parquet_file.metadata.num_rows

        total_chunks = (total_rows + chunk_size - 1) // chunk_size
        
        collected_rows = []
        total_collected = 0
        chunks_processed = 0
        start_time = time.time()
        
        # Create progress elements
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Read sequentially through the file
        for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=columns if columns else None):
            chunks_processed += 1
            
            # Update progress bar
            progress_percentage = min(chunks_processed / total_chunks, 1.0)
            progress_bar.progress(progress_percentage)
            
            # Convert to pandas
            df_chunk = batch.to_pandas()
            
            # Apply filters
            df_chunk = apply_filters_to_dataframe(df_chunk, selected_filters)
            
            # Apply label/prediction filter
            label_pred_filter = selected_filters.get('label_prediction', 'Display all')
            df_chunk = apply_label_prediction_filter(df_chunk, label_pred_filter)
            
            # If this chunk has matching data
            if not df_chunk.empty:
                rows_needed = max_rows - total_collected
                
                # Take only what we need
                if len(df_chunk) > rows_needed:
                    df_chunk = df_chunk.head(rows_needed)
                
                collected_rows.append(df_chunk)
                total_collected += len(df_chunk)
                
                # Update status
                status_text.success(f"✅ Found {total_collected:,}/{max_rows:,} rows (chunk {chunks_processed:,}/{total_chunks:,})")
                
                # Stop when we have enough
                if total_collected >= max_rows:
                    progress_bar.progress(1.0)
                    elapsed = time.time() - start_time
                    status_text.success(f"🎉 Complete! Got {total_collected:,} rows in {elapsed:.1f}s")
                    break
            else:
                # Update status for empty chunks
                if chunks_processed % 50 == 0:  # Update less frequently to avoid spam
                    status_text.info(f"🔍 Searching... chunk {chunks_processed:,}/{total_chunks:,} - found {total_collected:,} rows")
        
        # Clean up after a moment
        time.sleep(1.5)
        progress_bar.empty()
        status_text.empty()
        
        # Combine results
        if collected_rows:
            result_df = pd.concat(collected_rows, ignore_index=True)
        else:
            result_df = pd.DataFrame()
            st.warning("No data found matching your filters")
        
        return result_df
        
    except Exception as e:
        st.error(f"Error reading parquet file: {e}")
        return pd.DataFrame()
    
def create_pyarrow_filters(selected_filters):
    """
    Create PyArrow filters based on selected filters from the UI.
    This function converts the selected filters into a format suitable for PyArrow filtering.
    Args:
        selected_filters (dict): Dictionary of selected filters with keys like 'model', 'explainer', 'metric', 'instance', etc.
    Returns:
        list: List of PyArrow filters to apply
    Example:
        >>> selected_filters = {
        ...     'model': ['MLPMixer', 'ConvNeXtV2'],
        ...     'explainer': ['IntegratedGradients'],
        ...     'metric': ['Sparseness'],
        ...     'instance': [1, 2, 3],
        ...     'activation': 'softmax'
        ... }
        >>> filters = create_pyarrow_filters(selected_filters)
        >>> print(filters)
        [('model', 'in', ['MLPMixer', 'ConvNeXtV2']), ('explainer', 'in', ['IntegratedGradients']), ('metric', 'in', ['Sparseness']), ('instance', 'in', [1, 2, 3]), ('activation', '=', 'softmax')]
    """
    filters = []
    
    for key in ['model', 'activation', 'explainer', 'metric', 'instance']:
        if selected_filters.get(key):
            value = selected_filters[key]
            
            # Convert to list if needed
            if isinstance(value, str):
                filter_list = [value]
            elif isinstance(value, (int, float)):
                filter_list = [value]
            elif hasattr(value, '__iter__'):
                filter_list = list(value)
            else:
                filter_list = [value]
            
            # Skip empty filters
            if not filter_list or (len(filter_list) == 1 and not filter_list[0]):
                continue
            
            # Convert instances to integers
            if key == 'instance':
                converted_list = []
                for inst in filter_list:
                    try:
                        converted_list.append(int(inst))
                    except (ValueError, TypeError):
                        converted_list.append(inst)
                filter_list = converted_list
            
            # Create PyArrow filter
            if len(filter_list) == 1:
                filters.append((key, '=', filter_list[0]))
            else:
                filters.append((key, 'in', filter_list))
    
    return filters

def apply_filters_to_dataframe(df, selected_filters):
    """
    Apply selected filters to a DataFrame.
    This function filters the DataFrame based on the selected filters from the UI.
    Args:
        df (pd.DataFrame): Input DataFrame to filter
        selected_filters (dict): Dictionary of selected filters with keys like 'model', 'explainer', 'metric', 'instance', etc.
    Returns:
        pd.DataFrame: Filtered DataFrame based on the selected filters
    """
    
    if df.empty:
        return df
    
    # Model filter
    if selected_filters.get('model') and len(selected_filters['model']) > 0:
        df = df[df['model'].isin(selected_filters['model'])]
    
    # Activation filter
    if selected_filters.get('activation') and len(selected_filters['activation']) > 0:
        df = df[df['activation'] == selected_filters['activation']]
    
    # Explainer filter
    if selected_filters.get('explainer') and len(selected_filters['explainer']) > 0:
        df = df[df['explainer'].isin(selected_filters['explainer'])]
    
    # Metric filter
    if selected_filters.get('metric') and len(selected_filters['metric']) > 0:
        df = df[df['metric'].isin(selected_filters['metric'])]
    
    # Instance filter
    if selected_filters.get('instance') and len(selected_filters['instance']) > 0:
        instances = []
        for inst in selected_filters['instance']:
            try:
                instances.append(int(inst))
            except (ValueError, TypeError):
                instances.append(inst)
        df = df[df['instance'].isin(instances)]
    
    return df


def clean_all_filters():
    """Reset all filter selections in session state to default values.

    Clears all selected filters in the session state, effectively resetting
    the UI to its initial state. This allows users to start fresh without
    any previously applied filters.

    Args:
        None: Directly modifies st.session_state.selected_filters.

    Returns:
        None: Updates st.session_state.selected_filters to default values.

    Example:
        >>> clean_all_filters()
        >>> # Resets all filters to their default state
    """
    st.session_state.selected_filters['model'] = []
    st.session_state.selected_filters['activation'] = 'softmax'
    st.session_state.selected_filters['explainer'] = []
    st.session_state.selected_filters['metric'] = []
    st.session_state.selected_filters['label_prediction'] = "Display All"
    st.session_state.selected_filters['instance'] = []