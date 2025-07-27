import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import configparser
from tqdm import tqdm
import pandas as pd

def getVolumeWeightedAveragePrice(df: pd.DataFrame, config: configparser.ConfigParser):
    """
    Calculate the Volume Weighted Average Price (VWAP) for a DataFrame.
    
    Args:
        df: DataFrame containing 'Price' and 'Volume' columns.
        config: ConfigParser object with database settings.
    
    Returns:
        DataFrame with an additional 'VWAP' column.
    """
    if config['SQL']['DataColName'] not in df.columns or config['SQL']['VolumeColName'] not in df.columns:
        raise ValueError("DataFrame must contain '{config['SQL']['DataColName']}' and '{config['SQL']['VolumeColName']}' columns.")
    
    vwap = (df[config['SQL']['DataColName']] * df[config['SQL']['VolumeColName']]).cumsum() / df[config['SQL']['VolumeColName']].cumsum()
    df['VWAP'] = vwap
    return df

def getLogReturns(dx):
    """
    Calculate the log returns of a given list of prices.

    Parameters:
    dx (list): A list of prices.

    Returns:
    list: A list of log returns.
    """

    if len(dx) < 2:
        return []

    # Calculate log returns
    log_returns = np.log(np.array(dx[1:]) / np.array(dx[:-1]))
    
    return log_returns.tolist()

def getDifferences(dx):
    """
    Calculate the differences of a given list of prices.

    Parameters:
    dx (list): A list of prices.

    Returns:
    list: A list of differences.
    """
    if len(dx) < 2:
        return []

    # Calculate differences
    differences = np.array(dx[1:]) - np.array(dx[:-1])
    
    return differences.tolist()

def getSimpleReturns(dx):
    """
    Calculate the simple returns of a given list of prices.

    Parameters:
    dx (list): A list of prices.

    Returns:
    list: A list of simple returns.
    """
    if len(dx) < 2:
        return []

    # Calculate simple returns
    simple_returns = np.array(dx[1:]) / np.array(dx[:-1]) - 1
    
    return simple_returns.tolist()

def perona_malik_smoothing(time: np.array, data: np.array, config: configparser.ConfigParser):
    """
    Applies Perona-Malik diffusion to smooth 1D data while preserving edges.
    """
    smoothed = data.copy()
    padded = np.pad(smoothed, (1, 1), 'edge')
    assert int(config['Perona-Malik']['Iterations']) > 0, "Iterations must be a positive integer"
    assert float(config['Perona-Malik']['TimeStep']) > 0, "Time step must be a positive float"
    assert float(config['Perona-Malik']['Kappa']) > 0, "Kappa must be a positive float"
    assert float(config['Perona-Malik']['TimeStep']) < 0.25, "Time step must be less than 0.25 for stability"

    iterations = int(config['Perona-Malik']['Iterations'])
    with tqdm(total=iterations, desc='Perona-Malik Smoothing Progress') as pbar:
        for _ in range(iterations):
            diff_left = padded[1:-1] - padded[:-2]
            diff_right = padded[2:] - padded[1:-1]
            c_left = np.exp(-(diff_left / float(config['Perona-Malik']['Kappa'])) ** 2)
            c_right = np.exp(-(diff_right / float(config['Perona-Malik']['Kappa'])) ** 2)
            smoothed += float(config['Perona-Malik']['TimeStep']) * (c_right * diff_right - c_left * diff_left)
            padded[1:-1] = smoothed
            pbar.update(1)

    candle_col = config['DEFAULT']['DataColName']
    project_root = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(project_root, 'Plots', config['SQL']['TableName'])
    os.makedirs(plots_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(time, data, 'b-', label=candle_col, alpha=0.8)
    plt.plot(time, smoothed, 'r-', label='Smoothed ' + candle_col, alpha=0.8)
    plt.title(candle_col + ' Chart')
    plt.xlabel('Time')
    plt.ylabel(candle_col)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'Perona_Malik_smoothing.png'))
    plt.close()
    return smoothed