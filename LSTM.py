import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import random
import math
import configparser
import FeatureSpace
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

def build_lstm(window_size):
    model = Sequential([
        LSTM(1, stateful=False, input_shape=(window_size, 1)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def trainLSTM(tau: int, config: configparser.ConfigParser):
    lookback = int(config['LSTM']['LookbackWindow'])
    if lookback <= 0:
        assert tau > 0, "Lookback window must be greater than 0 if not specified in config."
        lookback = tau
    model = build_lstm(lookback)
    scaler = MinMaxScaler()

    data = FeatureSpace.loadData(config)
    print(data.head())
    dateTimes = pd.to_datetime(data[config['Candle']['CandleTimeColName']])
    data['Period'] = dateTimes.dt.to_period(config['LSTM']['SplitPeriod'])
    uniquePeriods = list(data['Period'].unique())
    numPeriods = len(uniquePeriods)
    np.random.shuffle(uniquePeriods)

    # Split into train,validation and test sets
    assert float(config['LSTM']['ValidationSetFraction']) <= 1.0, "Validation set fraction must be less than or equal to 1.0"
    assert float(config['LSTM']['ValidationSetFraction']) >= 0.0, "Validation set fraction must be greater than or equal to 0.0"
    assert float(config['LSTM']['TrainingSetFraction']) <= 1.0, "Training set fraction must be less than or equal to 1.0"
    assert float(config['LSTM']['TrainingSetFraction']) >= 0.0, "Training set fraction must be greater than or equal to 0.0"
    assert float(config['LSTM']['ValidationSetFraction']) + float(config['LSTM']['TrainingSetFraction']) <= 1.0, "Validation and training set fractions must sum to less than or equal to 1.0"
    
    testSetSize = int((1 - float(config['LSTM']['ValidationSetFraction']) - float(config['LSTM']['TrainingSetFraction'])) * numPeriods)
    testSet = set(random.sample(uniquePeriods, testSetSize))
    filteredPeriods = [period for period in uniquePeriods if period not in testSet]
    usedSplits= set()
    split = int(float(config['LSTM']['TrainingSetFraction']) * numPeriods)
    bitmask = np.array([1] * split + [0] * (len(filteredPeriods) - split), dtype=bool)
    best_val_loss = float('inf')
    best_model_file = None
    val_losses_all = []
    for i in range(min(int(config['LSTM']['NumPermutations']), math.comb(len(filteredPeriods), split))):
        while True:
            np.random.shuffle(bitmask)
            newBitMask = tuple(bitmask)
            if newBitMask not in usedSplits:
                usedSplits.add(newBitMask)
                break
        trainPeriods = [filteredPeriods[j] for j in range(len(filteredPeriods)) if newBitMask[j]]
        valPeriods = [filteredPeriods[j] for j in range(len(filteredPeriods)) if not newBitMask[j]]

        train_df = data[data['Period'].isin(trainPeriods)]
        val_df = data[data['Period'].isin(valPeriods)]

        for epoch in range(int(config['LSTM']['Epochs'])):
            print(f"Epoch {epoch+1}/{config['LSTM']['Epochs']}")

            # --- TRAINING ---
            for period in tqdm(trainPeriods, desc="Training Periods"):
                df = train_df[train_df['Period'] == period]
                df = df.sort_values(by=config['Candle']['CandleTimeColName']) # probably not needed, but ensures order
                if len(df) <= lookback:
                    continue
                prices = df[config['SQL']['DataColName']].values.reshape(-1, 1)
                prices_scaled = scaler.fit_transform(prices)
                X, y = [], []
                for k in range(len(prices_scaled) - lookback):
                    X.append(prices_scaled[k:k+lookback])
                    y.append(prices_scaled[k+lookback])
                X, y = np.array(X), np.array(y)
                model.fit(X, y, epochs=1, batch_size=32, verbose=0, shuffle=False)

            # --- VALIDATION ---
            val_losses = []
            for period in tqdm(valPeriods, desc="Validation Periods"):
                df = val_df[val_df['Period'] == period]
                if len(df) <= lookback:
                    continue
                prices = df[config['SQL']['DataColName']].values.reshape(-1, 1)
                prices_scaled = scaler.fit_transform(prices)
                X, y = [], []
                for k in range(len(prices_scaled) - lookback):
                    X.append(prices_scaled[k:k+lookback])
                    y.append(prices_scaled[k+lookback])
                X, y = np.array(X), np.array(y)
                loss = model.evaluate(X, y, verbose=0)
                val_losses.append(loss)
            if val_losses:
                print(f"Validation loss after epoch {epoch+1}: {np.mean(val_losses):.6f}")

        fileName = config['SQL']['TableName']+"_lstm_model_"+str(i)+".h5"
        model.save(fileName)
        print("Model saved as ", fileName)
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        val_losses_all.append((avg_val_loss, fileName))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_file = fileName

    # --- Evaluate and plot testset ---
    if best_model_file is not None:
        print(f"Evaluating best model ({best_model_file}) on test set (per period)...")
        best_model = load_model(best_model_file)
        test_df = data[data['Period'].isin(testSet)]
        test_df = test_df.sort_values(by=config['Candle']['CandleTimeColName'])
        test_periods = sorted(list(testSet))
        project_root = os.path.dirname(os.path.abspath(__file__))
        plots_dir = os.path.join(project_root, 'Plots', config['SQL']['TableName'])
        os.makedirs(plots_dir, exist_ok=True)

        for period in test_periods:
            period_df = test_df[test_df['Period'] == period]
            if len(period_df) > lookback:
                prices = period_df[config['SQL']['DataColName']].values.reshape(-1, 1)
                prices_scaled = scaler.fit_transform(prices)
                X_test, y_test = [], []
                for k in range(len(prices_scaled) - lookback):
                    X_test.append(prices_scaled[k:k+lookback])
                    y_test.append(prices_scaled[k+lookback])
                X_test, y_test = np.array(X_test), np.array(y_test)
                y_pred = best_model.predict(X_test)
                test_loss = best_model.evaluate(X_test, y_test, verbose=0)
                print(f"Test loss for period {period}: {test_loss:.6f}")

                # Plot
                plt.figure(figsize=(10, 5))
                plt.plot(y_test, label='True')
                plt.plot(y_pred, label='Predicted')
                plt.title(f'LSTM Model: Testset Prediction for Period {period}')
                plt.legend()
                plt.savefig(os.path.join(plots_dir, f'lstm_test_set_{str(period)}.png'))
                plt.close()
                print(f"Testset plot for period {period} saved.")
            else:
                print(f"Warning: Testset for period {period} too small for evaluation.")
    else:
        print("No valid model found for testset evaluation.")