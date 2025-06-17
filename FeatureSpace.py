import pandas as pd
import configparser

config = configparser.ConfigParser()
config.read('MyConfig')

def createCandleDataFrame(df) -> pd.DataFrame:
    """
    Creates a candlestick DataFrame from raw trading data.
    
    Args:
        df (DataFrame): Raw trading data with timestamp and price columns
        
    Returns:
        DataFrame: OHLCV candlestick data with minute-based candles
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    timestamp_col = config['SQL']['timeColName']
    
    # Convert millisecond timestamps to datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='ms')
    
    # Sort by timestamp
    df = df.sort_values(by=timestamp_col)
    
    # Create minute-based timestamp for grouping
    df['group_timestamp'] = df[timestamp_col].dt.floor(config['Candle']['timeGranularity'])
    
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
    
    return candles
