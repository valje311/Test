import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

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

import numpy as np

def perona_malik_smoothing(time, data, config):
    """
    Applies Perona-Malik diffusion to smooth 1D data while preserving edges.
    
    Args:
        time: Time array corresponding to the data
        data: Input 1D numpy array
        Iterations: Number of iterations for the diffusion process
        TimeStep: Time step size (should be < 0.25 for stability)
        Kappa: Diffusion constant, controls the sensitivity to edges
        
    Returns:
        Smoothed data array
    """
    from tqdm import tqdm

    # Convert input to numpy array if needed
    data = np.array(data, dtype=float)
    smoothed = data.copy()
    
    # Pad array to handle boundaries
    padded = np.pad(smoothed, (1, 1), 'edge')
    iterations = int(config['Perona-Malik']['Iterations'])
    with tqdm(total=iterations, desc='Perona-Malik Smoothing Progress') as pbar:
        for _ in range(iterations):
            # Calculate gradients
            diff_left = padded[1:-1] - padded[:-2]  # backward difference
            diff_right = padded[2:] - padded[1:-1]  # forward difference
            
            # Calculate diffusion coefficients
            c_left = np.exp(-(diff_left/float(config['Perona-Malik']['Kappa']))**2)
            c_right = np.exp(-(diff_right/float(config['Perona-Malik']['Kappa']))**2)
            
            # Update the signal
            smoothed += float(config['Perona-Malik']['TimeStep']) * (c_right * diff_right - c_left * diff_left)
            
            # Update padded array for next iteration
            padded[1:-1] = smoothed
            
            # Update progress bar
            pbar.update(1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(time, data, 'b-', label=config['SQL']['DataColName'], alpha=0.8)
    plt.plot(time, smoothed, 'r-', label='Smoothed ' + config['SQL']['DataColName'], alpha=0.8)
    plt.title(config['SQL']['DataColName'] + ' Chart')
    plt.xlabel('Time')
    plt.ylabel(config['SQL']['DataColName'])
    plt.grid(True)
    plt.legend()

    # Get and print the absolute paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(project_root, 'Plots', config['SQL']['tableName'])
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'Perona_Malik_smoothing.png'))
    plt.close()
    return smoothed    