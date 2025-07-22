import numpy as np
import pandas as pd
from typing import Union, List, cast
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Für 3D-Plots
from mpl_toolkits.mplot3d.axes3d import Axes3D as Axes3DType
from statsmodels.tsa.stattools import grangercausalitytests
from typing import Tuple
import os
from tqdm import tqdm

def friedman_diaconis_bins(data: np.ndarray) -> int:
    """Estimate the optimal number of histogram bins using the Friedman-Diaconis rule."""
    data = np.asarray(data)
    n = len(data)
    if n < 2:
        return 1
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / np.cbrt(n)
    if bin_width == 0:
        return 1
    num_bins = int(np.ceil((data.max() - data.min()) / bin_width))
    return max(1, num_bins)

def calculate_mutual_information(series: np.ndarray, delay: int, config, plots_dir=None) -> float:
    """
    Calculates the mutual information between a time series and its delayed version.

    Args:
        series (np.ndarray): The input time series.
        delay (int): The time delay (tau).
        config: ConfigDict object with get() method

    Returns:
        float: The mutual information value.
    """
    if delay >= len(series):
        return 0.0

    # Ensure series is a proper numpy array
    series = np.asarray(series, dtype=float)

    # Create delayed copies
    series_original = series[:-delay]
    series_delayed = series[delay:]

    # Joint histogram
    bins_x = config.getint('Mutual Information', 'NumBinsX', fallback=275)
    bins_y = config.getint('Mutual Information', 'NumBinsY', fallback=350)
    binning = (bins_x, bins_y)
    
    if config.getboolean('Mutual Information', 'UseFriedmanDiaconis', fallback=False):
        # Use Friedman-Diaconis rule to determine the number of bins
        num_bins_x = friedman_diaconis_bins(series_original)
        num_bins_y = friedman_diaconis_bins(series_delayed)
        binning = (num_bins_x, num_bins_y)
    
    hist_joint, _, _ = np.histogram2d(series_original, series_delayed, bins=binning, density=True)
    
    # Plot and save the joint histogram as a heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(
        hist_joint.T,  # transpose for correct orientation
        origin='lower',
        aspect='auto',
        cmap='viridis'
    )
    plt.colorbar(label='Density')
    plt.xlabel('Original Series')
    plt.ylabel('Delayed Series')
    plt.title('Joint Histogram (2D Density)')

    # Save to the Plots directory
    if plots_dir is None:
        project_root = os.path.dirname(os.path.abspath(__file__))
        table_name = config.get('SQL', 'TableName', fallback='default')
        plots_dir = os.path.join(project_root, 'Plots', table_name)
        os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'Joint_Histogram' + str(delay) + '.png'))
    plt.close()

    # Marginal histograms
    hist_original, _ = np.histogram(series_original, bins=binning[0], density=True)
    hist_delayed, _ = np.histogram(series_delayed, bins=binning[1], density=True)

    # Convert histograms to probabilities (normalize to sum to 1)
    # Add a small epsilon to avoid log(0)
    p_joint = hist_joint / np.sum(hist_joint) + np.finfo(float).eps
    p_original = hist_original / np.sum(hist_original) + np.finfo(float).eps
    p_delayed = hist_delayed / np.sum(hist_delayed) + np.finfo(float).eps

    # Calculate mutual information
    # MI(X;Y) = sum(p(x,y) * log(p(x,y) / (p(x)*p(y))))
    
    # Outer product to get p(x) * p(y) for all pairs
    p_outer = np.outer(p_original, p_delayed) + np.finfo(float).eps
    
    mutual_info = np.sum(p_joint * np.log(p_joint / p_outer))

    return float(mutual_info)

def find_optimal_time_delay(data: Union[pd.Series, List[float], np.ndarray], config, plots_dir=None) -> int:
    """
    Finds the optimal time delay (tau) using the first minimum of the mutual information.

    Args:
        data (Union[pd.Series, List[float], np.ndarray]): The time series data.
        config: ConfigDict object with get() method

    Returns:
        int: The optimal time delay (tau).
    """
    if isinstance(data, pd.Series):
        series = data.values
    else:
        series = np.array(data)

    # Ensure proper numpy array
    series = np.asarray(series, dtype=float)

    max_lag = config.getint('Mutual Information', 'MaxLag', fallback=100)
    
    mutual_informations = []
    lags = range(1, max_lag + 1)

    print("Calculating Mutual Information for various lags...")
    with tqdm(total=len(lags), desc="Mutual Information") as pbar:
        for lag in lags:
            mi = calculate_mutual_information(series, lag, config, plots_dir)
            mutual_informations.append(mi)
            pbar.update(1)

    # Plot Mutual Information
    plt.figure(figsize=(12, 6))
    plt.plot(lags, mutual_informations, marker='o', linestyle='-')
    plt.xlabel('Time Lag (τ)')
    plt.ylabel('Mutual Information (MI)')
    plt.title('Mutual Information vs. Time Lag')
    plt.grid(True)

    project_root = os.path.dirname(os.path.abspath(__file__))
    if plots_dir is None:
        table_name = config.get('SQL', 'TableName', fallback='default')
        plots_dir = os.path.join(project_root, 'Plots', table_name)
        os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'Mutual_Information.png'))
    plt.close()

    # Find the first minimum
    optimal_tau = 1
    if len(mutual_informations) > 2: 
        for i in range(1, len(mutual_informations) - 1):
            if (mutual_informations[i] < mutual_informations[i-1] and 
                mutual_informations[i] < mutual_informations[i+1]):
                optimal_tau = lags[i]
                break
    
    if optimal_tau == 1 and len(mutual_informations) > 1 and mutual_informations[0] < mutual_informations[1]:
        pass
    elif optimal_tau == 1 and len(mutual_informations) > 1:
        pass

    print(f"Optimal time delay (tau) found using Mutual Information: {optimal_tau}")
    return optimal_tau


def calculate_false_nearest_neighbors(data: np.ndarray, tau: int, config, plots_dir=None) -> Tuple[np.ndarray, int]:
    """
    Calculates the percentage of false nearest neighbors (FNN) for various embedding dimensions.
    
    Args:
        data (np.ndarray): The input time series.
        tau (int): The time delay.
        config: ConfigDict object with get() method
        
    Returns:
        Tuple[np.ndarray, int]: A tuple containing:
            - fnn_percentages (np.ndarray): Array of FNN percentages for each dimension.
            - optimal_embedding_dim (int): The estimated optimal embedding dimension.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    n_points = len(data)
    max_dim = config.getint('False Nearest Neighbour', 'MaxDim', fallback=15)
    fnn_percentages = np.zeros(max_dim)
    
    print("Calculating False Nearest Neighbors...")
    with tqdm(total=max_dim, desc="False Nearest Neighbors") as pbar:
        for m in range(1, max_dim + 1):
            if (m - 1) * tau >= n_points:
                fnn_percentages[m-1:] = 100.0
                break

            # Create embedded vectors
            embedded_series = sliding_window_view(data, window_shape=m * tau)[::tau]
            
            # Adjust n_points for the embedded series
            n_embedded = len(embedded_series)
            
            false_neighbors_count = 0
            
            for i in range(n_embedded):
                current_point = embedded_series[i]
                
                # Find nearest neighbor in m-dimension
                min_dist_sq_m = np.inf
                nearest_neighbor_idx_m = -1
                
                for j in range(n_embedded):
                    if i == j:
                        continue
                    
                    dist_sq = np.sum((current_point - embedded_series[j])**2)
                    if dist_sq < min_dist_sq_m:
                        min_dist_sq_m = dist_sq
                        nearest_neighbor_idx_m = j
                
                if nearest_neighbor_idx_m == -1:
                    continue

                # Check if we can form (m+1)-dimensional points
                if (i + m * tau >= n_points) or \
                   (nearest_neighbor_idx_m + m * tau >= n_points):
                   continue

                # Get the (m+1)-th coordinate for current point and its neighbor
                x_m_plus_1 = data[i + m * tau]
                nn_m_plus_1 = data[nearest_neighbor_idx_m + m * tau]
                
                # Calculate distance in (m+1)-dimension
                dist_sq_m_plus_1 = min_dist_sq_m + (x_m_plus_1 - nn_m_plus_1)**2
                
                # FNN criteria
                rtol = config.getfloat('False Nearest Neighbour', 'RTol', fallback=15.0)
                atol = config.getfloat('False Nearest Neighbour', 'ATol', fallback=3.0)
                
                # RTol test
                if np.sqrt(dist_sq_m_plus_1) / np.sqrt(min_dist_sq_m) > rtol:
                    false_neighbors_count += 1
                # ATol test
                elif np.abs(x_m_plus_1 - nn_m_plus_1) / np.std(data) > atol:
                    false_neighbors_count += 1

            fnn_percentages[m-1] = (false_neighbors_count / n_embedded) * 100 if n_embedded > 0 else 0
            pbar.update(1)

    # Plot FNN percentages
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, max_dim + 1), fnn_percentages, marker='o', linestyle='-')
    plt.xlabel('Embedding Dimension (m)')
    plt.ylabel('Percentage of False Nearest Neighbors (%)')
    plt.title('False Nearest Neighbors Method')
    plt.grid(True)
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    if plots_dir is None:
        table_name = config.get('SQL', 'TableName', fallback='default')
        plots_dir = os.path.join(project_root, 'Plots', table_name)
        os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'False_Nearest_Neighbors.png'))
    plt.close()

    # Determine optimal embedding dimension
    optimal_embedding_dim = 1
    if len(fnn_percentages) > 0:
        min_fnn_percentage = np.min(fnn_percentages)
        optimal_embedding_dim = np.where(fnn_percentages == min_fnn_percentage)[0][0] + 1
        
        # Heuristic threshold
        threshold = 5.0
        for i, perc in enumerate(fnn_percentages):
            if perc <= threshold:
                optimal_embedding_dim = i + 1
                break
        
    print(f"Optimal embedding dimension (m) found using FNN: {optimal_embedding_dim}")
    return fnn_percentages, optimal_embedding_dim

def granger_causality_test(data: pd.DataFrame, max_lag: int = 5, verbose: bool = True) -> None:
    """
    Perform Granger causality test on pairs of time series in a DataFrame.
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


def calculate_autocorrelation(data: Union[pd.Series, List[float], np.ndarray], config, plots_dir=None) -> np.ndarray:
    """
    Calculate and optionally plot the autocorrelation function for a time series.
    
    Args:
        data: Time series data as pandas Series, list, or numpy array
        config: ConfigDict object with get() method
    
    Returns:
        numpy array containing autocorrelation values for each lag
    """
    # Convert input to numpy array
    if isinstance(data, pd.Series):
        series = data.values
    else:
        series = np.array(data)
    
    # Ensure proper numpy array
    series = np.asarray(series, dtype=float)
    
    # Remove mean from series
    series = series - np.mean(series)
    
    # Calculate variance
    variance = np.var(series)
    
    # Initialize autocorrelation array
    max_lag = config.getint('Autocorrelation', 'MaxLag', fallback=100)
    autocorr = np.zeros(max_lag + 1)
    n = len(series)
    
    # Calculate autocorrelation for each lag with progress bar
    with tqdm(total=max_lag + 1, desc="Calculating autocorrelation") as pbar:
        for lag in range(max_lag + 1):
            # Calculate covariance
            cov = np.sum((series[lag:] * series[:(n-lag)])) / (n - lag)
            autocorr[lag] = cov / variance
            pbar.update(1)
    
    # Create and save plot
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(autocorr)), autocorr)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.axhline(y=1.96/np.sqrt(n), color='r', linestyle='--')
    plt.axhline(y=-1.96/np.sqrt(n), color='r', linestyle='--')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Function')
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    if plots_dir is None:
        table_name = config.get('SQL', 'TableName', fallback='default')
        plots_dir = os.path.join(project_root, 'Plots', table_name)
        os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'Autocorrelation.png'))
    plt.close()
    
    return autocorr

def calculate_bollinger_bands(data: Union[pd.Series, List[float], np.ndarray],
                            window: int = 20,
                            num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands for a time series.
    """
    # Convert input to numpy array
    if isinstance(data, pd.Series):
        series = data.values
    else:
        series = np.array(data)
    
    # Ensure proper numpy array
    series = np.asarray(series, dtype=float)
    
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
    
    return middle_band, upper_band, lower_band

def TakenEmbedding(myReturns: Union[pd.Series, List[float], np.ndarray], plots_dir, config) -> None:
    """
    Perform Taken's embedding analysis.
    
    Args:
        myReturns: Time series data
        plots_dir: Directory to save plots
        config: ConfigDict object with get() method
    """
    # Convert to numpy array first
    if isinstance(myReturns, pd.Series):
        returns_array = myReturns.values
    else:
        returns_array = np.array(myReturns)
    
    # Ensure proper numpy array
    returns_array = np.asarray(returns_array, dtype=float)
    
    optimal_tau = 0
    optimal_embedding_dim = 0 
    
    if config.getboolean('Takens Embedding', 'UseUserDefinedParameters', fallback=False):
        optimal_tau = config.getint('Takens Embedding', 'TimeDelay', fallback=3)
        optimal_embedding_dim = config.getint('Takens Embedding', 'EmbeddingDim', fallback=3)
    else:
        optimal_tau = find_optimal_time_delay(returns_array, config, plots_dir)
        fnn_percentages, optimal_embedding_dim = calculate_false_nearest_neighbors(returns_array, optimal_tau, config, plots_dir)
        
        num_points_for_embedding = len(returns_array) - (optimal_embedding_dim - 1) * optimal_tau
        if num_points_for_embedding <= 0:
            print(f"Warning: Not enough data points ({len(returns_array)}) to form embedded vectors with tau={optimal_tau} and m={optimal_embedding_dim}.")
            print("Please consider a longer time series or smaller tau/m values in the config file.")
            delay_vectors = np.array([])
        else:
            delay_vectors = np.zeros((num_points_for_embedding, optimal_embedding_dim))
            for i in range(num_points_for_embedding):
                for j in range(optimal_embedding_dim):
                    delay_vectors[i, j] = returns_array[i + j * optimal_tau]

            if optimal_embedding_dim >= 2:
                plt.figure(figsize=(10, 8))
                plt.plot(delay_vectors[:, 0], delay_vectors[:, 1], 'b-', linewidth=0.5, alpha=0.7)
                plt.xlabel(f"Returns at t")
                plt.ylabel(f"Returns at t + {optimal_tau}")
                plt.title(f"2D Phase Space Reconstruction (m={optimal_embedding_dim}, τ={optimal_tau})")
                plt.grid(True)
                plt.savefig(os.path.join(plots_dir, 'Phase_Space_2D_Reconstruction.png'))
                plt.close()
                print(f"Saved 2D phase space reconstruction plot to {os.path.join(plots_dir, 'Phase_Space_2D_Reconstruction.png')}")

            if optimal_embedding_dim >= 3:
                fig = plt.figure(figsize=(10, 8))
                ax = cast(Axes3DType, fig.add_subplot(111, projection='3d'))
                ax.plot(delay_vectors[:, 0], delay_vectors[:, 1], delay_vectors[:, 2], 'b-', linewidth=0.5, alpha=0.7)
                ax.set_xlabel(f"Returns at t")
                ax.set_ylabel(f"Returns at t + {optimal_tau}")
                ax.set_zlabel(f"Returns at t + {2 * optimal_tau}")
                ax.set_title(f"3D Phase Space Reconstruction (m={optimal_embedding_dim}, τ={optimal_tau})")
                plt.savefig(os.path.join(plots_dir, 'Phase_Space_3D_Reconstruction.png'))
                plt.close()
                print(f"Saved 3D phase space reconstruction plot to {os.path.join(plots_dir, 'Phase_Space_3D_Reconstruction.png')}")