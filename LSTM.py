import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

def build_lstm(window_size):
    model = Sequential([
        LSTM(32, stateful=False, batch_input_shape=(None, window_size, 1)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def trainLSTM(config):
    CONNECTION_STRING = f"mysql+mysqlconnector://{config['SQL']['Username']}:{config['SQL']['Password']}@localhost:{config['SQL']['Port']}/{config['SQL']['Database']}"
    engine = create_engine(CONNECTION_STRING)

    lookback = config['LSTM']['LookbackWindow']
    epochs = config['LSTM']['Epochs']

    model = build_lstm(lookback)
    scaler = MinMaxScaler()

    # Get all unique days
    dates_query = f"SELECT DISTINCT DATE(FROM_UNIXTIME({config['SQL']['TimeColName']}/1000)) as day FROM {config['SQL']['TableName']} ORDER BY day"
    dates = pd.read_sql(dates_query, engine)['day'].tolist()
    
    # Split into train,validation and test sets
    testSetSize = (1 - float(config['LSTM']['ValidationSetFraction']) - float(config['LSTM']['TrainingSetFraction'])) * len(dates)
    testSet = set(random.sample(dates, testSetSize))
    filteredDates = [date for date in dates if date not in testSet]
    usedSplits= set()
    split = int(float(config['LSTM']['TrainingSetFraction']) * len(dates))
    bitmask = np.array([1] * split + [0] * (len(filteredDates) - split), dtype=bool)
    for i in range(min(int(config['LSTM']['NumPermutations']), math.comb(len(filteredDates), split))):
        while True:
            newBitMask = tuple(np.random.shuffle(bitmask))
            if newBitmask not in usedSplits:
                usedSplits.add(newBitmask)
                break
        trainDates, valDates = [filteredDates[i] for i in range(len(filteredDates)) if newBitMask[i]], [filteredDates[i] for i in range(len(filteredDates)) if not newBitMask[i]]
    

        for epoch in range(config['LSTM']['Epochs']):
            print(f"Epoch {epoch+1}/{config['LSTM']['Epochs']}")

            # --- TRAINING ---
            for day in tqdm(trainDates, desc="Training Days"):
                query = f"""
                    SELECT {config['SQL']['TimeColName']}, {config['SQL']['DataColName']}
                    FROM {config['SQL']['TableName']}
                    WHERE DATE(FROM_UNIXTIME({config['SQL']['TimeColName']}/1000)) = '{day}'
                    ORDER BY {config['SQL']['TimeColName']}
                """
                df = pd.read_sql(query, engine)
                if len(df) <= lookback:
                    continue

                prices = df[config['SQL']['DataColName']].values.reshape(-1, 1)
                prices_scaled = scaler.fit_transform(prices)
                X, y = [], []
                for i in range(len(prices_scaled) - lookback):
                    X.append(prices_scaled[i:i+lookback])
                    y.append(prices_scaled[i+lookback])
                X, y = np.array(X), np.array(y)
                model.fit(X, y, epochs=1, batch_size=32, verbose=0, shuffle=False)

            # --- VALIDATION ---
            val_losses = []
            for day in tqdm(valDates, desc="Validation Days"):
                query = f"""
                    SELECT {config['SQL']['TimeColName']}, {config['SQL']['DataColName']}
                    FROM {config['SQL']['TableName']}
                    WHERE DATE(FROM_UNIXTIME({config['SQL']['TimeColName']}/1000)) = '{day}'
                    ORDER BY {config['SQL']['TimeColName']}
                """
                df = pd.read_sql(query, engine)
                if len(df) <= lookback:
                    continue

                prices = df[config['SQL']['DataColName']].values.reshape(-1, 1)
                prices_scaled = scaler.fit_transform(prices)
                X, y = [], []
                for i in range(len(prices_scaled) - lookback):
                    X.append(prices_scaled[i:i+lookback])
                    y.append(prices_scaled[i+lookback])
                X, y = np.array(X), np.array(y)
                loss = model.evaluate(X, y, verbose=0)
                val_losses.append(loss)
            if val_losses:
                print(f"Validation loss after epoch {epoch+1}: {np.mean(val_losses):.6f}")

        fileName = config['SQL']['TableName']+"_lstm_model_"+str(i)+".h5"
        model.save(fileName)
        print("Model saved as ", fileName)