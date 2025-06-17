from datetime import datetime
import configparser
from sqlalchemy import create_engine, text
import pandas as pd
from typing import Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import FeatureSpace
import TimeSeriesAnalysis
import TimeSeriesManipulation

config = configparser.ConfigParser()
config.read('MyConfig')

def load_tick_data(start_date: datetime, end_date: datetime, config: configparser.ConfigParser) -> pd.DataFrame:
    """
    Load tick data from MySQL database for a given time period.
    
    Args:
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        config: ConfigParser object with database settings
    
    Returns:
        pandas DataFrame with tick data
    """
    # Create database connection
    CONNECTION_STRING = f"mysql+mysqlconnector://{config['SQL']['Username']}:{config['SQL']['Password']}@localhost:{config['SQL']['Port']}/{config['SQL']['Database']}"
    engine = create_engine(CONNECTION_STRING)
    
    # Prepare query
    features = [config['SQL']['timeColName'], 'Close', 'Volume']
    columns = ", ".join(features)
    query = text(f"""
        SELECT {columns}
        FROM {config['SQL']['TableName']}
        WHERE Timestamp BETWEEN :start AND :end
        ORDER BY Timestamp
    """)
    
    with engine.connect() as conn:
        # Convert dates to milliseconds timestamp
        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)
        
        df = pd.read_sql(
            query,
            conn,
            params={'start': start_ms, 'end': end_ms}
        )
    
    return df

def get_data_boundaries(config: configparser.ConfigParser) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get first and last entries from the database.
    
    Args:
        config: ConfigParser object with database settings
    
    Returns:
        Tuple of (first_entry, last_entry) as pandas DataFrames
    """
    CONNECTION_STRING = f"mysql+mysqlconnector://{config['SQL']['Username']}:{config['SQL']['Password']}@localhost:{config['SQL']['Port']}/{config['SQL']['Database']}"
    engine = create_engine(CONNECTION_STRING)
    
    features = [config['SQL']['TimeColName'], config['SQL']['DataColName'], 'Volume']
    columns = ", ".join(features)
    
    first_query = text(f"""
        SELECT {columns}
        FROM {config['SQL']['TableName']}
        ORDER BY Timestamp ASC
        LIMIT 1
    """)
    
    last_query = text(f"""
        SELECT {columns}
        FROM {config['SQL']['TableName']}
        ORDER BY Timestamp DESC
        LIMIT 1
    """)
    
    with engine.connect() as conn:
        first_entry = pd.read_sql(first_query, conn)
        last_entry = pd.read_sql(last_query, conn)
        
        # Convert timestamp from milliseconds to datetime
        for df in [first_entry, last_entry]:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    
    return first_entry, last_entry


startDateTime = datetime.strptime(config['SQL']['StartDate'], '%d.%m.%Y %H:%M:%S')
endDateTime = datetime.strptime(config['SQL']['EndDate'], '%d.%m.%Y %H:%M:%S')
print("Loading data from database...")
df = load_tick_data(startDateTime, endDateTime, config)
print("Data loaded successfully.")
candles = FeatureSpace.createCandleDataFrame(df, config)
smoothed = TimeSeriesManipulation.perona_malik_smoothing(candles[config['SQL']['TimeColName']], candles[config['SQL']['DataColName']].tolist(), config)

get_log_returns = TimeSeriesManipulation.getLogReturns(smoothed.tolist())
calculate_autocorr = TimeSeriesAnalysis.calculate_autocorrelation(get_log_returns, config)

plt.figure()
plt.xlabel('Time')
plt.ylabel('log Returns')
plt.title('Lag visualization of log Returns')
plt.plot(candles[config['SQL']['TimeColName']][1:], get_log_returns, label='Log Returns', color='blue')
plt.plot(candles[config['SQL']['TimeColName']][1:-int(config['Autocorrelation']['Lag'])], get_log_returns[int(config['Autocorrelation']['Lag']):], label='Log Returns with lag', color='orange')

project_root = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(project_root, 'Plots', config['SQL']['TableName'])
os.makedirs(plots_dir, exist_ok=True)
plt.savefig(os.path.join(plots_dir, 'LogReturns.png'))
plt.close()

#differences = TimeSeriesManipulation.getDifferences(df['Volume'].tolist())
#bollinger_bands_middle, bollinger_bands_high, bollinger_bands_low = TimeSeriesAnalysis.calculate_bollinger_bands(df['Close'], window=20, num_std=2)
#df = df.iloc[1:].reset_index(drop=True)
# while bollinger_bands_middle[0] == 0.0:
#     df = df.iloc[1:].reset_index(drop=True)
#     bollinger_bands_middle = np.delete(bollinger_bands_middle, 0)
#     bollinger_bands_high = np.delete(bollinger_bands_high, 0)
#     bollinger_bands_low = np.delete(bollinger_bands_low, 0)
#     get_log_returns = np.delete(get_log_returns, 0)
#     differences = np.delete(differences, 0)
# df['Bollinger_Middle'] = TimeSeriesManipulation.getLogReturns(bollinger_bands_middle)
# df['Bollinger_High'] = TimeSeriesManipulation.getLogReturns(bollinger_bands_high)
# df['Bollinger_Low'] = TimeSeriesManipulation.getLogReturns(bollinger_bands_low)
# df['Close'] = get_log_returns
# df['Volume'] = differences

#print("autocorrelation:", calculate_autocorr)
# granger_causality = TimeSeriesAnalysis.granger_causality_test(df, max_lag=100, verbose=True)