def getLogReturns(dx):
    """
    Calculate the log returns of a given list of prices.

    Parameters:
    dx (list): A list of prices.

    Returns:
    list: A list of log returns.
    """
    import numpy as np

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
    import numpy as np
    differences = np.array(dx[1:]) - np.array(dx[:-1])
    
    return differences.tolist()