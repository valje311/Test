from tqdm import tqdm
import pandas as pd
import mplfinance as mpf
import os

def createCandleDataFrame(df, config, plots_dir=None) -> pd.DataFrame:
    """
    Creates a candlestick DataFrame from raw trading data and saves a candlestick plot.
    
    Args:
        df (DataFrame): Raw trading data with timestamp and price columns
        config (dict): Configuration dictionary
        
    Returns:
        DataFrame: OHLCV candlestick data with minute-based candles
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    timestamp_col = config['SQL']['timeColName']
    
    with tqdm(total=5, desc="Creating candles") as pbar:
        # Convert millisecond timestamps to datetime
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='ms')
        pbar.update(1)
        
        # Sort by timestamp
        df = df.sort_values(by=timestamp_col)
        pbar.update(1)
        
        # Create minute-based timestamp for grouping
        df['group_timestamp'] = df[timestamp_col].dt.floor(config['Candle']['timeGranularity'])
        pbar.update(1)
        
        # Group and calculate OHLCV
        candles = df.groupby('group_timestamp').agg({
            'Close': ['first', 'max', 'min', 'last'],
            'Volume': 'sum'
        })
        
        # Flatten column names
        candles.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        candles = candles.reset_index()
        candles = candles.rename(columns={'group_timestamp': timestamp_col})
        
        # Ensure timestamps are timezone-naive
        if candles[timestamp_col].dt.tz is not None:
            candles[timestamp_col] = candles[timestamp_col].dt.tz_localize(None)
        pbar.update(1)
        
        # Create and save candlestick plot
        # Set index to timestamp for mplfinance
        plot_df = candles.set_index(timestamp_col)
        
        # Use provided plots_dir or create default one
        if plots_dir is None:
            project_root = os.path.dirname(os.path.abspath(__file__))
            plots_dir = os.path.join(project_root, 'Plots', config['SQL']['tableName'])
            os.makedirs(plots_dir, exist_ok=True)
        
        # Create candlestick plot
        mpf.plot(
            plot_df,
            type='candle',
            volume=True,
            title=f"{config['SQL']['tableName']} Candlestick Chart",
            savefig=os.path.join(plots_dir, 'candlestick_chart.png'),
            style='charles',
            figsize=(12, 8),
            warn_too_much_data=len(plot_df) + 1  # Suppress warning
        )
        pbar.update(1)
    
    return candles