import pandas as pd
import configparser

config = configparser.ConfigParser()
config.read('MyConfig')

def createCandleDataFrame(x_df, timeWindow):
    """
    Convert tick data to OHLCV (Open, High, Low, Close, Volume) candlesticks
    
    Args:
        df: DataFrame with columns ['Timestamp', 'Close', 'Volume']
        timeWindow: Candlestick timeframe in minutes
    """
    # Convert timestamp to datetime
    x_df[config['SQL']['timeColName']] = pd.to_datetime(x_df[config['SQL']['timeColName']], unit='ms')
    
    # Resample data to specified timeframe
    resampled = x_df.set_index(config['SQL']['timeColName']).resample(f'{timeWindow}T')
    
    candles = pd.DataFrame({
        'Open': resampled['Close'].first(),
        'High': resampled['Close'].max(),
        'Low': resampled['Close'].min(),
        'Close': resampled['Close'].last(),
        'Volume': resampled['Volume'].sum()
    }).dropna()
    
    return candles