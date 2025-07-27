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
import LSTM

config = configparser.ConfigParser()
config.read('MyConfig.ini')

def xor(a: bool, b: bool) -> bool:
    """
    Perform exclusive OR operation on two boolean values.
    
    Args:
        a: First boolean value
        b: Second boolean value
    
    Returns:
        True if exactly one of a or b is True, otherwise False
    """
    return (a and not b) or (not a and b)

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
    CONNECTION_STRING = f"mysql+mysqlconnector://{config['SQL']['UserName']}:{config['SQL']['Password']}@localhost:{config['SQL']['Port']}/{config['SQL']['Database']}"
    engine = create_engine(CONNECTION_STRING)
    
    # Prepare query
    features = [config['SQL']['TimeColName'], 'Close', 'Volume']
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
    CONNECTION_STRING = f"mysql+mysqlconnector://{config['SQL']['UserName']}:{config['SQL']['Password']}@localhost:{config['SQL']['Port']}/{config['SQL']['Database']}"
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


assert xor(config['DEFAULT']['UseLogReturns'] =='True', config['DEFAULT']['UseSimpleReturns'] == 'True'), "Exactly one of UseLogReturns or UseSimpleReturns must be True."
startDateTime = datetime.strptime(config['SQL']['StartDate'], '%d.%m.%Y %H:%M:%S')
endDateTime = datetime.strptime(config['SQL']['EndDate'], '%d.%m.%Y %H:%M:%S')
print("Loading data from database...")
df = load_tick_data(startDateTime, endDateTime, config)
print("Data loaded successfully.")
project_root = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(project_root, 'Plots', config['SQL']['TableName'])
os.makedirs(plots_dir, exist_ok=True)
if config['DEFAULT']['SmoothTicks'] == 'True':
    if config['DEFAULT']['SmoothTicksMethod'] == 'Perona-Malik':
        print("Smoothing data using Perona-Malik method...")
        df[config['SQL']['DataColName']] = TimeSeriesManipulation.perona_malik_smoothing(df[config['SQL']['TimeColName']], df[config['SQL']['DataColName']].tolist(), config)

candles = FeatureSpace.createCandleDataFrame(df, config)
if config['DEFAULT']['SmoothCandles'] == 'True':
    if config['DEFAULT']['SmoothCandlesMethod'] == 'Perona-Malik':
        print("Smoothing candles using Perona-Malik method...")
        candles[config['SQL']['DataColName']] = TimeSeriesManipulation.perona_malik_smoothing(candles[config['SQL']['TimeColName']], candles[config['SQL']['DataColName']].tolist(), config)

myReturns=None
if config['DEFAULT']['UseLogReturns'] == 'True':
    myReturns = TimeSeriesManipulation.getLogReturns(candles[config['SQL']['DataColName']].tolist())
    if config['DEFAULT']['SmoothLogReturns'] == 'True':
        if config['DEFAULT']['SmoothLogReturnsMethod'] == 'Perona-Malik':
            print("Smoothing log returns using Perona-Malik method...")
            myReturns = TimeSeriesManipulation.perona_malik_smoothing(candles[config['SQL']['TimeColName']], myReturns, config)
elif config['DEFAULT']['UseSimpleReturns'] == 'True':
    myReturns = TimeSeriesManipulation.getSimpleReturns(candles[config['SQL']['DataColName']].tolist())
    if config['DEFAULT']['SmoothSimpleReturns'] == 'True':
        if config['DEFAULT']['SmoothSimpleReturnsMethod'] == 'Perona-Malik':
            print("Smoothing simple returns using Perona-Malik method...")
            myReturns = TimeSeriesManipulation.perona_malik_smoothing(candles[config['SQL']['TimeColName']], myReturns, config)

calculate_autocorr = TimeSeriesAnalysis.calculate_autocorrelation(myReturns, config)
tau = TimeSeriesAnalysis.TakenEmbedding(myReturns, plots_dir, config)

plt.figure()
plt.xlabel('Time')
plt.ylabel('log Returns')
plt.title('Lag visualization of log Returns')
plt.plot(candles[config['SQL']['TimeColName']][1:], myReturns, label='Log Returns', color='blue')
plt.plot(candles[config['SQL']['TimeColName']][1:-int(config['Autocorrelation']['Lag'])], myReturns[int(config['Autocorrelation']['Lag']):], label='Log Returns with lag', color='orange')

plt.savefig(os.path.join(plots_dir, 'LogReturns.png'))
plt.close()

LSTM.trainLSTM(tau, config)

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