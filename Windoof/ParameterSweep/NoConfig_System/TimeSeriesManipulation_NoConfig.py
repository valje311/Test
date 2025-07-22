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

def perona_malik_smoothing(time, data, config, plots_dir=None):
    """
    Applies Perona-Malik diffusion to smooth 1D data while preserving edges.
    
    Args:
        time: Time array corresponding to the data
        data: Input 1D numpy array
        config: ConfigDict object with get() method
        plots_dir: Directory to save plots
        
    Returns:
        Smoothed data array
    """
    from tqdm import tqdm

    # Convert input to numpy array if needed
    data = np.array(data, dtype=float)
    smoothed = data.copy()
    
    # Pad array to handle boundaries
    padded = np.pad(smoothed, (1, 1), 'edge')
    iterations = config.getint('Perona-Malik', 'Iterations', fallback=200)
    
    with tqdm(total=iterations, desc='Perona-Malik Smoothing Progress') as pbar:
        for _ in range(iterations):
            # Calculate gradients
            diff_left = padded[1:-1] - padded[:-2]  # backward difference
            diff_right = padded[2:] - padded[1:-1]  # forward difference
            
            # Calculate diffusion coefficients
            kappa = config.getfloat('Perona-Malik', 'Kappa', fallback=2.0)
            c_left = np.exp(-(diff_left/kappa)**2)
            c_right = np.exp(-(diff_right/kappa)**2)
            
            # Update the signal
            time_step = config.getfloat('Perona-Malik', 'TimeStep', fallback=0.1)
            smoothed += time_step * (c_right * diff_right - c_left * diff_left)
            
            # Update padded array for next iteration
            padded[1:-1] = smoothed
            
            # Update progress bar
            pbar.update(1)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    data_col_name = config.get('SQL', 'DataColName', fallback='Close')
    plt.plot(time, data, 'b-', label=data_col_name, alpha=0.8)
    plt.plot(time, smoothed, 'r-', label='Smoothed ' + data_col_name, alpha=0.8)
    plt.title(data_col_name + ' Chart')
    plt.xlabel('Time')
    plt.ylabel(data_col_name)
    plt.grid(True)
    plt.legend()

    # Use provided plots_dir or create default one
    if plots_dir is None:
        project_root = os.path.dirname(os.path.abspath(__file__))
        table_name = config.get('SQL', 'TableName', fallback='default')
        plots_dir = os.path.join(project_root, 'Plots', table_name)
        os.makedirs(plots_dir, exist_ok=True)
    
    plt.savefig(os.path.join(plots_dir, 'Perona_Malik_smoothing.png'))
    plt.close()
    return smoothed
