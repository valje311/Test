import numpy as np
import pandas as pd
from typing import Union, List
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from typing import Tuple

def granger_causality_test(data: pd.DataFrame, max_lag: int = 5, verbose: bool = True) -> None:
    """
    Perform Granger causality test on pairs of time series in a DataFrame.
    
    Args:
        data: DataFrame containing time series data with columns as series names
        max_lag: Maximum number of lags to consider for the test
        verbose: Whether to print the results
    
    Returns:
        None
    """
    columns = data.columns
    n = len(columns)
    
    for i in range(n):
        for j in range(i + 1, n):
            x = data[columns[i]]
            y = data[columns[j]]
            result = grangercausalitytests(np.column_stack((x, y)), max_lag, verbose=verbose)
            if verbose:
                print(f"Granger causality test between {columns[i]} and {columns[j]}:")
                for lag, res in result.items():
                    print(f"Lag {lag}: F-statistic = {res[0]['ssr_ftest'][0]}, p-value = {res[0]['ssr_ftest'][1]}")


def calculate_autocorrelation(data: Union[pd.Series, List[float], np.ndarray], config) -> np.ndarray:
    """
    Calculate and optionally plot the autocorrelation function for a time series.
    
    Args:
        data: Time series data as pandas Series, list, or numpy array
        max_lags: Maximum number of lags to calculate
        plot: Whether to plot the autocorrelation function
    
    Returns:
        numpy array containing autocorrelation values for each lag
    """
    # Convert input to numpy array
    if isinstance(data, pd.Series):
        series = data.values
    else:
        series = np.array(data)
    
    # Remove mean from series
    series = series - np.mean(series)
    
    # Calculate variance
    variance = np.var(series)
    
    # Initialize autocorrelation array
    autocorr = np.zeros(int(config['Autocorrelation']['MaxLag']) + 1)
    n = len(series)
    
    # Calculate autocorrelation for each lag
    for lag in range(int(config['Autocorrelation']['MaxLag']) + 1):
        # Calculate covariance
        cov = np.sum((series[lag:] * series[:(n-lag)])) / (n - lag)
        autocorr[lag] = cov / variance
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(autocorr)), autocorr)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.axhline(y=1.96/np.sqrt(n), color='r', linestyle='--')
    plt.axhline(y=-1.96/np.sqrt(n), color='r', linestyle='--')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Function')
    plt.savefig('Autocorrelation.png')
    plt.close()
    
    return autocorr

def calculate_bollinger_bands(data: Union[pd.Series, List[float], np.ndarray],
                            window: int = 20,
                            num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands for a time series.
    
    Args:
        data: Time series data as pandas Series, list, or numpy array
        window: Moving average window size (default: 20)
        num_std: Number of standard deviations for bands (default: 2.0)
    
    Returns:
        Tuple of (middle_band, upper_band, lower_band) as numpy arrays
    """
    # Convert input to numpy array
    if isinstance(data, pd.Series):
        series = data.values
    else:
        series = np.array(data)
    
    # Calculate middle band (simple moving average)
    middle_band = np.zeros(len(series))
    for i in range(window - 1, len(series)):
        middle_band[i] = np.mean(series[i - window + 1:i + 1])
    
    # Calculate standard deviation
    std = np.zeros(len(series))
    for i in range(window - 1, len(series)):
        std[i] = np.std(series[i - window + 1:i + 1])
    
    # Calculate upper and lower bands
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    if plt:
        plt.figure(figsize=(12, 6))
        plt.plot(series, label='Price', color='blue', alpha=0.5)
        plt.plot(middle_band, label='Middle Band', color='red')
        plt.plot(upper_band, label='Upper Band', color='gray', linestyle='--')
        plt.plot(lower_band, label='Lower Band', color='gray', linestyle='--')
        plt.legend()
        plt.title('Bollinger Bands')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.show()
    
    return middle_band, upper_band, lower_band