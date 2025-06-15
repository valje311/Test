import pandas as pd
import configparser
from sqlalchemy import create_engine
import mysql.connector

config = configparser.ConfigParser()
config.read('MyConfig')

CONNECTION_STRING = f"mysql+mysqlconnector://{config['SQL']['username']}:{config['SQL']['password']}@localhost:{config['SQL']['port']}/{config['SQL']['database']}"
engine = create_engine(CONNECTION_STRING)

for chunk in pd.read_csv(config['CSV']['file_path'], chunksize=int(config['CSV']['chunk_size']), sep=";"): 
    chunk['Timestamp'] = pd.to_datetime(chunk['Timestamp'], format='%Y.%m.%d %H:%M:%S').astype(int) // 10**6
    chunk.to_sql(config['SQL']['table_name'], engine, if_exists='append', index=False)
