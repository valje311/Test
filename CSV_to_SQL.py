import pandas as pd
import configparser
from sqlalchemy import create_engine
import mysql.connector

config = configparser.ConfigParser()
config.read('MyConfig.ini')

CONNECTION_STRING = f"mysql+mysqlconnector://{config['SQL']['username']}:{config['SQL']['password']}@localhost:{config['SQL']['port']}/{config['SQL']['database']}"
engine = create_engine(CONNECTION_STRING)

for chunk in pd.read_csv(config['CSV']['filepath'], chunksize=int(config['CSV']['chunksize']), sep=";"): 
    chunk['Timestamp'] = pd.to_datetime(chunk['Timestamp'], format='%Y.%m.%d %H:%M:%S.%f').astype('int64') // 10**6
    chunk.to_sql(config['SQL']['tablename'], engine, if_exists='append', index=False)
