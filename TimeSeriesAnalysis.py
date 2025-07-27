import numpy as np
import pandas as pd
from typing import Union, List
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from typing import Tuple
import os
from tqdm import tqdm
import configparser

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

def calculate_mutual_information(series: np.ndarray, delay: int, config: configparser.ConfigParser) -> float:
    """
    Calculates the mutual information between a time series and its delayed version.

    Args:
        series (np.ndarray): The input time series.
        delay (int): The time delay (tau).
        NumBins (int): The number of bins for histogram estimation.

    Returns:
        float: The mutual information value.
    """
    if delay >= len(series):
        return 0.0

    # Create delayed copies
    series_original = series[:-delay]
    series_delayed = series[delay:]

    # Joint histogram
    binning = (int(config['Mutual Information']['NumBinsX']), int(config['Mutual Information']['NumBinsY']))
    if config['Mutual Information']['UseFriedmanDiaconis'] == 'True':
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
        # extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap='viridis'
    )
    plt.colorbar(label='Density')
    plt.xlabel('Original Series')
    plt.ylabel('Delayed Series')
    plt.title('Joint Histogram (2D Density)')

    # Save to the Plots directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(project_root, 'Plots', config['SQL']['TableName'])
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

    return mutual_info

def find_optimal_time_delay(data: Union[pd.Series, List[float], np.ndarray], config: configparser.ConfigParser) -> int:
    """
    Finds the optimal time delay (tau) using the first minimum of the mutual information.

    Args:
        data (Union[pd.Series, List[float], np.ndarray]): The time series data.
        config (configparser.ConfigParser): Configuration object with 'MaxLag' setting.

    Returns:
        int: The optimal time delay (tau).
    """
    if isinstance(data, pd.Series):
        series = data.values
    else:
        series = np.array(data)

    max_lag = int(config['Mutual Information']['MaxLag'])
    
    mutual_informations = []
    lags = range(1, max_lag + 1)

    print("Calculating Mutual Information for various lags...")
    with tqdm(total=len(lags), desc="Mutual Information") as pbar:
        for lag in lags:
            mi = calculate_mutual_information(series, lag, config)
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
    plots_dir = os.path.join(project_root, 'Plots', config['SQL']['TableName'])
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'Mutual_Information.png'))
    plt.close()

    # Find the first minimum
    optimal_tau = 1
    # Ensure there are at least 3 points to check for a minimum (prev, current, next)
    if len(mutual_informations) > 2: 
        for i in range(1, len(mutual_informations) - 1):
            if (mutual_informations[i] < mutual_informations[i-1] and 
                mutual_informations[i] < mutual_informations[i+1]):
                optimal_tau = lags[i]
                break # Found the first minimum
        # If no minimum is found, often the first minimum is chosen when it starts to level off or simply the first point (lag 1) if MI continuously decreases.
        # However, the strict definition implies a dip. If it just keeps decreasing, a common heuristic is to pick the first lag where MI drops significantly or
        # the lag where MI is significantly below its initial value. For simplicity and adhering to "first minimum", we'll stick to the strict dip.
        # If no strict minimum is found, it means MI is likely monotonically decreasing or has complex behavior.
        # In such cases, taking a point where the curve flattens out or where the decay is significant might be an alternative.
        # For now, if no minimum is found, it will default to 1 as initialized.
    
    if optimal_tau == 1 and len(mutual_informations) > 1 and mutual_informations[0] < mutual_informations[1]:
        # This handles the case where the first value is already a minimum (i.e., MI starts increasing immediately)
        # This is rare but technically possible.
        pass
    elif optimal_tau == 1 and len(mutual_informations) > 1:
        # If optimal_tau is still 1, and no clear minimum was found, and MI is not decreasing,
        # it might imply the optimal_tau is indeed 1 if the very first drop is significant.
        # This part is more heuristic. The standard is the first local minimum.
        pass

    print(f"Optimal time delay (tau) found using Mutual Information: {optimal_tau}")
    return optimal_tau

def calculate_false_nearest_neighbors(data: np.ndarray, tau: int, config: configparser.ConfigParser) -> Tuple[np.ndarray, int]:
    """
    Calculates the percentage of false nearest neighbors (FNN) for various embedding dimensions.
    
    Args:
        data (np.ndarray): The input time series.
        tau (int): The time delay.
        MaxDim (int): The maximum embedding dimension to test.
        RTol (float): Tolerance for distance increase (ratio test).
        ATol (float): Tolerance for distance increase (absolute test).
        
    Returns:
        Tuple[np.ndarray, int]: A tuple containing:
            - fnn_percentages (np.ndarray): Array of FNN percentages for each dimension.
            - optimal_embedding_dim (int): The estimated optimal embedding dimension.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    n_points = len(data)
    fnn_percentages = np.zeros(int(config['False Nearest Neighbour']['MaxDim']))
    
    print("Calculating False Nearest Neighbors...")
    with tqdm(total=int(config['False Nearest Neighbour']['MaxDim']), desc="False Nearest Neighbors") as pbar:
        for m in range(1, int(config['False Nearest Neighbour']['MaxDim']) + 1):
            if (m - 1) * tau >= n_points:
                fnn_percentages[m-1:] = 100.0 # Cannot form embedding for higher dimensions
                break

            # Create embedded vectors
            # Each row is an m-dimensional vector: [x(t), x(t+tau), ..., x(t+(m-1)tau)]
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
                
                if nearest_neighbor_idx_m == -1: # No neighbor found (e.g., n_embedded = 1)
                    continue

                # Check if we can form (m+1)-dimensional points
                if (i + m * tau >= n_points) or \
                   (nearest_neighbor_idx_m + m * tau >= n_points):
                   # Cannot check in m+1 dimension, treat as not an FNN for now, or skip
                   continue

                # Get the (m+1)-th coordinate for current point and its neighbor
                x_m_plus_1 = data[i + m * tau]
                nn_m_plus_1 = data[nearest_neighbor_idx_m + m * tau]
                
                # Calculate distance in (m+1)-dimension
                dist_sq_m_plus_1 = min_dist_sq_m + (x_m_plus_1 - nn_m_plus_1)**2
                
                # FNN criteria
                # RTol test: Check if the relative increase in distance is large
                if np.sqrt(dist_sq_m_plus_1) / np.sqrt(min_dist_sq_m) > float(config['False Nearest Neighbour']['RTol']):
                    false_neighbors_count += 1
                # ATol test: Check if the absolute increase in distance is large relative to the attractor size
                # This part needs an estimate of the attractor's diameter.
                # A common heuristic is to use the standard deviation of the time series as a proxy for attractor size.
                # For more rigorous implementation, compute the actual diameter of the embedded attractor.
                # Here, we'll use a simpler approximation if RTol fails.
                elif np.abs(x_m_plus_1 - nn_m_plus_1) / np.std(data) > float(config['False Nearest Neighbour']['ATol']):
                    false_neighbors_count += 1

            fnn_percentages[m-1] = (false_neighbors_count / n_embedded) * 100 if n_embedded > 0 else 0
            pbar.update(1)

    # Plot FNN percentages
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, int(config['False Nearest Neighbour']['MaxDim']) + 1), fnn_percentages, marker='o', linestyle='-')
    plt.xlabel('Embedding Dimension (m)')
    plt.ylabel('Percentage of False Nearest Neighbors (%)')
    plt.title('False Nearest Neighbors Method')
    plt.grid(True)
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(project_root, 'Plots', config['SQL']['TableName'])
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'False_Nearest_Neighbors.png'))
    plt.close()

    # Determine optimal embedding dimension (first minimum or where percentage drops below a threshold)
    optimal_embedding_dim = 1
    # Find the first dimension where FNN percentage is below a small threshold (e.g., 5-10%)
    # Or, where the percentage significantly drops and levels off.
    # A simple approach is the first dimension where FNN is minimal.
    if len(fnn_percentages) > 0:
        min_fnn_percentage = np.min(fnn_percentages)
        # Find the first occurrence of this minimum
        optimal_embedding_dim = np.where(fnn_percentages == min_fnn_percentage)[0][0] + 1
        
        # Heuristic: If min_fnn_percentage is still high, or if it stabilizes after an initial drop
        # You might want to define a threshold (e.g., 1% or 5%)
        threshold = 5.0 # Example threshold
        for i, perc in enumerate(fnn_percentages):
            if perc <= threshold:
                optimal_embedding_dim = i + 1
                break
        
    print(f"Optimal embedding dimension (m) found using FNN: {optimal_embedding_dim}")
    return fnn_percentages, optimal_embedding_dim

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
        config: Configuration dictionary containing MaxLag and other settings
    
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
    max_lag = int(config['Autocorrelation']['MaxLag'])
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
    plots_dir = os.path.join(project_root, 'Plots', config['SQL']['TableName'])
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'Autocorrelation.png'))
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

def TakenEmbedding(myReturns: Union[pd.Series, List[float], np.ndarray], plots_dir, config: configparser.ConfigParser) -> None:
    optimal_tau = 0
    optimal_embedding_dim = 0 
    if config['Takens Embedding']['UseUserDefinedParameters'] == 'True':
        optimal_tau = int(config['Takens Embedding']['TimeDelay'])
        optimal_embedding_dim = int(config['Takens Embedding']['EmbeddingDim'])
    else:
        optimal_tau = find_optimal_time_delay(myReturns, config)
        fnn_percentages, optimal_embedding_dim = calculate_false_nearest_neighbors(myReturns, optimal_tau, config)
    num_points_for_embedding = len(myReturns) - (optimal_embedding_dim - 1) * optimal_tau
    if num_points_for_embedding <= 0:
        print(f"Warning: Not enough data points ({len(myReturns)}) to form embedded vectors with tau={optimal_tau} and m={optimal_embedding_dim}.")
        print("Please consider a longer time series or smaller tau/m values in the config file.")
        delay_vectors = np.array([]) # Empty array if cannot embed
    else:
        delay_vectors = np.zeros((num_points_for_embedding, optimal_embedding_dim))
        for i in range(num_points_for_embedding):
            for j in range(optimal_embedding_dim):
                delay_vectors[i, j] = myReturns[i + j * optimal_tau]

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
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(delay_vectors[:, 0], delay_vectors[:, 1], delay_vectors[:, 2], 'b-', linewidth=0.5, alpha=0.7)
            ax.set_xlabel(f"Returns at t")
            ax.set_ylabel(f"Returns at t + {optimal_tau}")
            ax.set_zlabel(f"Returns at t + {2 * optimal_tau}")
            ax.set_title(f"3D Phase Space Reconstruction (m={optimal_embedding_dim}, τ={optimal_tau})")
            plt.savefig(os.path.join(plots_dir, 'Phase_Space_3D_Reconstruction.png'))
            plt.close()
            print(f"Saved 3D phase space reconstruction plot to {os.path.join(plots_dir, 'Phase_Space_3D_Reconstruction.png')}")
    return optimal_tau