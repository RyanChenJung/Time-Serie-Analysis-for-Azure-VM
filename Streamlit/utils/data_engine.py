
import pandas as pd
import numpy as np
from io import StringIO

DEFAULT_DATA_PATH = 'final_deep_readings_20vms.csv'

def load_data(uploaded_file=None):
    """
    Loads data from the default CSV or an uploaded file.

    Args:
        uploaded_file: A file-like object from st.file_uploader.

    Returns:
        A pandas DataFrame.
    """
    if uploaded_file is not None:
        # To handle InMemoryUploadedFile from streamlit
        string_data = StringIO(uploaded_file.getvalue().decode("utf-8"))
        df = pd.read_csv(string_data)
    else:
        df = pd.read_csv(DEFAULT_DATA_PATH)

    # Validate required columns
    required_columns = {'timestamp', 'vm_id', 'avg_cpu', 'max_cpu', 'min_cpu'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV must contain the following columns: {required_columns}")

    return df

def apply_temporal_anchor(df, start_date, start_time):
    """
    Applies a user-defined temporal anchor to the dataframe.
    The 'timestamp' column is converted from integer offsets to datetime objects.

    Args:
        df (pd.DataFrame): The input dataframe with an integer 'timestamp' column.
        start_date (datetime.date): The user-defined start date.
        start_time (datetime.time): The user-defined start time.

    Returns:
        pd.DataFrame: The dataframe with a new 'timestamp_dt' column.
    """
    start_datetime = pd.to_datetime(f"{start_date} {start_time}")
    # Timestamps are raw second offsets (each step = 300 seconds = 5 minutes).
    # Use unit='s' to convert them directly — do NOT multiply by 5 again.
    df['timestamp_dt'] = start_datetime + pd.to_timedelta(df['timestamp'], unit='s')
    return df

def aggregate_and_interpolate(df, granularity='5-min'):
    """
    Aggregates data to the specified granularity and interpolates missing values.

    CRITICAL: The full_index is built *per VM* from each group's own min/max.
    A global full_index spanning all VMs causes catastrophic memory bloat:
    each sparse VM gets reindexed against the entire fleet's time range,
    producing ~172K NaN rows per VM before dropna — causing hangs.

    Args:
        df (pd.DataFrame): The input dataframe with 'timestamp_dt'.
        granularity (str): '5-min' (native) or '1-hour' (aggregated).

    Returns:
        pd.DataFrame: The processed dataframe with 'timestamp_dt' column.
    """
    metric_cols = ['avg_cpu', 'max_cpu', 'min_cpu']

    # Sort by timestamp for correct interpolation
    df = df.set_index('timestamp_dt').sort_index()

    if granularity == '1-hour':
        agg_rules = {'avg_cpu': 'mean', 'max_cpu': 'max', 'min_cpu': 'min'}

        processed_dfs = []
        for vm_id, group in df.groupby('vm_id'):
            resampled = group.resample('h').agg(agg_rules)

            # Per-VM hourly index — tight to this VM's own window only
            vm_full_index = pd.date_range(
                start=resampled.index.min(),
                end=resampled.index.max(),
                freq='h'
            )
            resampled = resampled.reindex(vm_full_index)
            resampled['vm_id'] = vm_id  # fill after reindex

            resampled[metric_cols] = resampled[metric_cols].interpolate(method='linear')
            resampled.dropna(subset=metric_cols, inplace=True)
            processed_dfs.append(resampled)

        if not processed_dfs:
            return pd.DataFrame()

        final_df = pd.concat(processed_dfs)
        final_df.index.name = 'timestamp_dt'  # name the DatetimeIndex correctly
        final_df = final_df.reset_index()      # promotes index to column

    else:  # '5-min' native
        processed_dfs = []
        for vm_id, group in df.groupby('vm_id'):
            # Per-VM 5-min index — tight to this VM's own window only
            vm_full_index = pd.date_range(
                start=group.index.min(),
                end=group.index.max(),
                freq='5min'
            )
            reindexed = group.reindex(vm_full_index)
            reindexed['vm_id'] = vm_id  # fill after reindex

            reindexed[metric_cols] = reindexed[metric_cols].interpolate(method='linear')
            reindexed.dropna(subset=metric_cols, inplace=True)
            processed_dfs.append(reindexed)

        if not processed_dfs:
            return pd.DataFrame()

        final_df = pd.concat(processed_dfs)
        final_df.index.name = 'timestamp_dt'  # name the DatetimeIndex correctly
        final_df = final_df.reset_index()      # promotes index to column

    return final_df.drop(columns=['timestamp'], errors='ignore')

import os
import pickle

CACHE_DIR = 'cache'
DEMO_CACHE_PATH = os.path.join(CACHE_DIR, 'demo_results.pkl')

def ensure_cache_dir():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def save_demo_cache(data):
    ensure_cache_dir()
    # Prune heavy results for caching
    pruned_data = {}
    for vm_id, diag in data.items():
        pruned_data[vm_id] = {
            'is_eligible': diag['is_eligible'],
            'adf_p': diag['adf_p'],
            'ljung_p': diag['ljung_p'],
            'detected_seasonality': diag['detected_seasonality'],
            'cv_results': None,      # Always explicitly None in cache
            'holdout_metrics': None  # Always explicitly None in cache
        }
    with open(DEMO_CACHE_PATH, 'wb') as f:
        pickle.dump(pruned_data, f)

def load_demo_cache():
    if os.path.exists(DEMO_CACHE_PATH):
        try:
            with open(DEMO_CACHE_PATH, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    return None

def is_demo_data(df):
    """Checks if the dataframe corresponds to the demo dataset."""
    expected_vms = 20
    # Simple check: number of VMs and specific column names
    if 'vm_id' in df.columns and df['vm_id'].nunique() == expected_vms:
        return True
    return False
