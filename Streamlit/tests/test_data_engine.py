
import pandas as pd
import pytest
from datetime import date, time
from utils.data_engine import apply_temporal_anchor, aggregate_and_interpolate

@pytest.fixture
def sample_df():
    """
    Provides a sample dataframe for testing.

    Timestamps are raw seconds (each 5-min step = 300s), matching the real CSV format.
      0s, 300s, 600s  -> 00:00, 00:05, 00:10  (first hour)
      3300s, 3600s, 3900s -> 00:55, 01:00, 01:05 (straddle 1-hour boundary)
    """
    data = {
        'timestamp': [0, 300, 600, 3300, 3600, 3900],
        'vm_id': ['vm-1', 'vm-1', 'vm-1', 'vm-1', 'vm-1', 'vm-1'],
        'avg_cpu': [10, 12, 11, 20, 22, 21],
        'max_cpu': [15, 18, 16, 25, 30, 28],
        'min_cpu': [5, 6, 5, 10, 11, 10]
    }
    return pd.DataFrame(data)


def test_temporal_anchor(sample_df):
    """
    Tests if the temporal anchor correctly maps the first timestamp (0s) to T0.
    """
    start_date = date(2026, 2, 23)
    start_time = time(10, 0)

    df = apply_temporal_anchor(sample_df, start_date, start_time)

    # Timestamp 0s -> exactly the start datetime
    expected_first_timestamp = pd.to_datetime("2026-02-23 10:00:00")
    assert df['timestamp_dt'].iloc[0] == expected_first_timestamp

    # Timestamp 300s -> T0 + 5 minutes
    expected_second = pd.to_datetime("2026-02-23 10:05:00")
    assert df['timestamp_dt'].iloc[1] == expected_second


def test_hourly_aggregation(sample_df):
    """
    Tests if hourly aggregation correctly applies max/mean/min rules.

    With second-unit timestamps:
      Hour 00:00: timestamps 0s, 300s, 600s (00:00, 00:05, 00:10)
        max_cpu values: 15, 18, 16 -> max = 18
      Hour 00:55 falls in first hour bucket (resample 'h' rounds down to 00:00)
        Actually 3300s = 55min -> still in 00:00 bucket
        So first-hour max_cpu = max(15, 18, 16, 25) = 25
      Hour 01:00: timestamps 3600s, 3900s
        max_cpu: 30, 28 -> max = 30
    """
    start_date = date(2026, 1, 1)
    start_time = time(0, 0)

    df_anchored = apply_temporal_anchor(sample_df, start_date, start_time)
    df_hourly = aggregate_and_interpolate(df_anchored, granularity='1-hour')

    # First hour (00:00): timestamps 0s/300s/600s/3300s are all in 00:xx
    # max_cpu = max(15, 18, 16, 25) = 25
    expected_max_cpu_first_hour = 25
    actual_max_cpu_first_hour = df_hourly['max_cpu'].iloc[0]
    assert actual_max_cpu_first_hour == expected_max_cpu_first_hour

    # avg_cpu first hour: mean(10, 12, 11, 20) = 13.25
    assert abs(df_hourly['avg_cpu'].iloc[0] - 13.25) < 0.01

    # min_cpu first hour: min(5, 6, 5, 10) = 5
    assert df_hourly['min_cpu'].iloc[0] == 5


def test_interpolation_5min(sample_df):
    """
    Tests if 5-min granularity correctly interpolates a missing value.

    Timestamps: 0s (00:00) and 600s (00:10).
    Missing step: 300s (00:05).
    avg_cpu: [10, _, 20]. Interpolated middle = 15.
    """
    data = {
        'timestamp': [0, 600],   # 0s = 00:00, 600s = 00:10 (gap at 300s = 00:05)
        'vm_id': ['vm-1', 'vm-1'],
        'avg_cpu': [10, 20],
        'max_cpu': [10, 20],
        'min_cpu': [10, 20]
    }
    df = pd.DataFrame(data)

    df_anchored = apply_temporal_anchor(df, date(2026, 1, 1), time(0, 0))
    df_interpolated = aggregate_and_interpolate(df_anchored, granularity='5-min')

    # Should have 3 rows: 00:00, 00:05, 00:10
    assert len(df_interpolated) == 3

    # Interpolated value at 00:05 should be 15.0
    assert df_interpolated['avg_cpu'].iloc[1] == 15.0
