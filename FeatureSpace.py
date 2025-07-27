from tqdm import tqdm
import pandas as pd
import mplfinance as mpf
import os, sys
import configparser
from sqlalchemy import create_engine
import TimeSeriesManipulation
import re

def loadTicks(config: configparser.ConfigParser):
    """
    Load tick data from the database and return a DataFrame.
    
    Args:
        config: ConfigParser object with database settings.
    
    Returns:
        DataFrame containing tick data.
    """

    CONNECTION_STRING = f"mysql+mysqlconnector://{config['SQL']['UserName']}:{config['SQL']['Password']}@localhost:{config['SQL']['Port']}/{config['SQL']['Database']}"
    engine = create_engine(CONNECTION_STRING)

    query = f"""
        SELECT {config['SQL']['TimeColName']}, {config['SQL']['DataColName']}, {config['SQL']['VolumeColName']}
        FROM {config['SQL']['TableName']}
        ORDER BY {config['SQL']['TimeColName']}
    """
    
    df = pd.read_sql(query, engine)
    df[config['SQL']['TimeColName']] = pd.to_datetime(df[config['SQL']['TimeColName']], unit='ms')
    if config['DEFAULT']['SmoothTicks'] == 'True':
        if config['DEFAULT']['SmoothTicksMethod'] == 'Perona-Malik':
            print("Smoothing data using Perona-Malik method...")
            df[config['SQL']['DataColName']] = TimeSeriesManipulation.perona_malik_smoothing(df[config['SQL']['TimeColName']], df[config['SQL']['DataColName']].tolist(), config)
        else:
            raise NotImplementedError("Smoothing method for ticks not implemented.")
    if config['DEFAULT']['UseVolumeWeightedPrice'] == 'True':
        if config['SQL']['VolumeColName'] not in df.columns:
            raise ValueError("DataFrame must contain 'Volume' column for VWAP calculation.")
        df = getVolumeWeightedAveragePrice(df, config)
    return df

def interval_to_seconds(interval_str):
    """
    Converts a pandas-style interval string (e.g. '1min', '5min', '1H', '1D') to seconds.
    """
    match = re.match(r'(\d+)\s*(min|h|d|s|m)', interval_str.lower())
    if not match:
        raise ValueError(f"Invalid interval format: {interval_str}")
    value, unit = match.groups()
    value = int(value)
    if unit in ['min', 'm']:
        return value * 60
    elif unit == 'h':
        return value * 3600
    elif unit == 'd':
        return value * 86400
    elif unit == 's':
        return value
    else:
        raise ValueError(f"Unknown time unit: {unit}")

def loadCandles(config: configparser.ConfigParser):
    CONNECTION_STRING = f"mysql+mysqlconnector://{config['SQL']['UserName']}:{config['SQL']['Password']}@localhost:{config['SQL']['Port']}/{config['SQL']['Database']}"
    engine = create_engine(CONNECTION_STRING)

    candle_interval = interval_to_seconds(config['Candle']['TimeGranularity'])
    if config['DEFAULT']['SmoothTicks'] == 'False':
        candles_query = f"""
            SELECT 
                FROM_UNIXTIME(FLOOR({config['SQL']['TimeColName']} / 1000 / {candle_interval}) * {candle_interval}) AS {config['Candle']['CandleTimeColName']},
                SUBSTRING_INDEX(GROUP_CONCAT({config['SQL']['DataColName']} ORDER BY {config['SQL']['TimeColName']} ASC), ',', 1) AS Open,
                MAX({config['SQL']['DataColName']}) AS High,
                MIN({config['SQL']['DataColName']}) AS Low,
                SUBSTRING_INDEX(GROUP_CONCAT({config['SQL']['DataColName']} ORDER BY {config['SQL']['TimeColName']} DESC), ',', 1) AS Close,
                SUM({config['SQL']['VolumeColName']}) AS Volume
            FROM {config['SQL']['TableName']}
            GROUP BY {config['Candle']['CandleTimeColName']}
            ORDER BY {config['Candle']['CandleTimeColName']}
        """
        result = pd.read_sql(candles_query, engine)
    else:
        result = loadTicks(config)
        dateTimes = pd.to_datetime(result[config['SQL']['TimeColName']], unit='ms')
        dateTimes = dateTimes.dt.floor(candle_interval)
        result = result.groupby(dateTimes).agg( open=(config['SQL']['DataColName'], 'first'), high=(config['SQL']['DataColName'], 'max'), low=(config['SQL']['DataColName'], 'min'), close=(config['SQL']['DataColName'], 'last'), volume=(config['SQL']['VolumeColName'], 'sum')).reset_index()
    
    if config['DEFAULT']['SmoothCandles'] == 'True':
        if config['DEFAULT']['SmoothCandlesMethod'] == 'Perona-Malik':
            print("Smoothing candles using Perona-Malik method...")
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in result.columns:
                    result[col] = TimeSeriesManipulation.perona_malik_smoothing(result[config['Candle']['CandleTimeColName']], result[col].tolist(), config)
        else:
            raise NotImplementedError("Smoothing method for candles not implemented.")
    return result

def createCandleDataFrame(df, config: configparser.ConfigParser) -> pd.DataFrame:
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
    
    timestamp_col = config['SQL']['TimeColName']
    
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
            config['SQL']['DataColName']: ['first', 'max', 'min', 'last'],
            config['SQL']['VolumeColName']: 'sum'
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
        
        # Create plots directory if it doesn't exist
        project_root = os.path.dirname(os.path.abspath(__file__))
        plots_dir = os.path.join(project_root, 'Plots', config['SQL']['TableName'])
        os.makedirs(plots_dir, exist_ok=True)
        
        # Create candlestick plot
        mpf.plot(
            plot_df,
            type='candle',
            volume=True,
            title=f"{config['SQL']['TableName']} Candlestick Chart",
            savefig=os.path.join(plots_dir, 'candlestick_chart.png'),
            style='charles',
            figsize=(12, 8)
        )
        pbar.update(1)
    
    return candles

def loadData(config: configparser.ConfigParser) -> pd.DataFrame:
    candles = loadCandles(config)
    if candles.empty:
        raise ValueError("No data found in the database.")

    candle_col = config['DEFAULT']['DataColName']
    if candles[candle_col].isnull().any():
        raise ValueError(f"NaN in '{candle_col}'! Please check the data base.")

    if config['DEFAULT']['UseLogReturns'] == 'True':
        values = pd.to_numeric(candles[candle_col], errors='raise').tolist()
        log_returns = TimeSeriesManipulation.getLogReturns(values)
        candles['LogReturns'] = pd.Series(log_returns, index=candles.index[1:])
        if config['DEFAULT']['SmoothLogReturns'] == 'True':
            if config['DEFAULT']['SmoothLogReturnsMethod'] == 'Perona-Malik':
                candles['LogReturns'] = TimeSeriesManipulation.perona_malik_smoothing(
                    candles[config['Candle']['CandleTimeColName']],
                    candles['LogReturns'],
                    config
                )
            else:
                raise NotImplementedError("Smoothing method for log returns not implemented.")
    elif config['DEFAULT']['UseSimpleReturns'] == 'True':
        values = pd.to_numeric(candles[candle_col], errors='raise').tolist()
        simple_returns = TimeSeriesManipulation.getSimpleReturns(values)
        candles['SimpleReturns'] = pd.Series(simple_returns, index=candles.index[1:])
        if config['DEFAULT']['SmoothSimpleReturns'] == 'True':
            if config['DEFAULT']['SmoothSimpleReturnsMethod'] == 'Perona-Malik':
                candles['SimpleReturns'] = TimeSeriesManipulation.perona_malik_smoothing(
                    candles[config['Candle']['CandleTimeColName']],
                    candles['SimpleReturns'],
                    config
                )
            else:
                raise NotImplementedError("Smoothing method for simple returns not implemented.")
    return candles