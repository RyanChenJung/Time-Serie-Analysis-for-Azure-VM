import pytest
import pandas as pd
import numpy as np
from utils.baselines import fit_naive, fit_snaive, fit_drift, fit_moving_average, expanding_window_cv

def test_baselines_logic():
    train = pd.Series([10, 11, 12, 13, 14, 15])
    
    # Naive: last value
    n_fc = fit_naive(train, 3)
    assert np.all(n_fc == 15)
    
    # SNaive: cycle repeat
    # Let's say seasonality is 3: [10, 11, 12, 13, 14, 15] -> cycle is [13, 14, 15]
    sn_fc = fit_snaive(train, 4, 3)
    assert np.array_equal(sn_fc, [13, 14, 15, 13])
    
    # Drift: slope
    # y1=10, yt=15, t=6. drift = (15-10)/(6-1) = 1.
    # forecasts: 16, 17, 18
    d_fc = fit_drift(train, 3)
    assert np.allclose(d_fc, [16, 17, 18])
    
    # MA: mean of window
    ma_fc = fit_moving_average(train, 3, window=2)
    assert np.all(ma_fc == 14.5)

def test_expanding_window_cv():
    np.random.seed(42)
    series = pd.Series(np.add(np.arange(100), np.random.normal(0, 1, 100))) # trend
    
    results = expanding_window_cv(series, seasonality=1)
    
    assert not results.empty
    assert 'Model' in results.columns
    assert 'RMSE' in results.columns
    assert 'is_winner' in results.columns
    
    # For a strong trend, Drift or Naive should arguably do better than random
    assert results['is_winner'].sum() >= 1
