"""
LSTM-RL Trading System mit Single-Policy-Architektur
===================================================

Ein Reinforcement-Learning-basiertes Handelssystem mit Long Short-Term Memory (LSTM)
Neuronalen Netzwerken für die Marktprognose und Handelsentscheidungen.

Basierend auf dem Projektplan mit vereinfachter Architektur und Single-Policy-Ansatz.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import random
from collections import deque
import time
import datetime
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Multiprocessing-Module für parallele Verarbeitung
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import functools
import math

# Überprüfe, ob CUDA verfügbar ist
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwende Gerät: {device}")

# Konfigurationsvariablen
CONFIG = {
    # Dateipfade
    'data_path': r"e:\PFAD\CandleData_Indicators.csv",
    'model_save_path': r"e:\PFAD\lstm_rl_model.pth",
    'scaler_save_path': r"e:\PFAD\lstm_rl_scaler.pkl",
    'actions_csv_path': r"e:\Pfad\Validation_Actions_{timestamp}.csv",
    'validation_csv_path': r"e:\PFAD\Test.csv",
    'trades_csv_path': r"e:\Pfad\Validation_Trades_{timestamp}.csv",
    
    # Feature-Auswahl - hier können Sie die zu verwendenden Features auswählen
    'features_to_use': ['Open', 'High', 'Low', 'Close', 'RSI', 'Momentum', 'MACD_Main', 'MACD_Signal', 'MACD_Cross'],
    
    # Datenparameter
    'sequence_length': 124,  # Anzahl der historischen Zeitschritte als Eingabe
    'test_size': 0.2,
    'validation_size': 0.1,
    
    # Modellparameter
    'state_dim': None,       # Wird automatisch basierend auf den ausgewählten Features gesetzt
    'action_dim': 2,        # Dimension des Aktionsraums (0: HOLD, 1: LONG) - nur Long-Positionen
    'hidden_dim': 16,      # Anzahl der versteckten LSTM-Einheiten
    'num_layers': 2,        # Anzahl der LSTM-Schichten
    
    # Training-Parameter
    'batch_size': 64,
    'learning_rate': 0.0005,
    'gamma': 0.99,          # Discount-Faktor für zukünftige Belohnungen
    'epsilon_start': 1.0,   # Startwahrscheinlichkeit für zufällige Aktionen
    'epsilon_end': 0.005,    # Endwahrscheinlichkeit für zufällige Aktionen
    'epsilon_decay': 0.99999, # Abklingrate für Epsilon
    'target_update': 5,    # Anzahl der Episoden zwischen Target-Network-Updates
    'memory_capacity': 500000, # Kapazität des Replay-Speichers
    'num_episodes': 50,    # Anzahl der Trainingsepisoden
    'validation_interval': 10, # Nach wie vielen Episoden soll die externe Validierung erfolgen
    
    # Trading-Parameter
    'initial_balance': 80,  # Startkapital
    'position_size': 1.0,      # Größe der Position als Anteil des Kapitals
    'stop_loss': -0.5,          # Stop-Loss als absoluter Betrag in USD
    'take_profit': 0.12,        # Take-Profit als absoluter Betrag in USD
    'max_holding_period': 60,  # Maximale Haltedauer in Zeitschritten
    'cooldown_period': 6,      # Anzahl der Zeitschritte, die nach Schließung einer Position gewartet werden muss
    
    # Multiprocessing-Parameter
    'use_multiprocessing': True,  # Parallelverarbeitung aktivieren
    'num_workers': max(1, mp.cpu_count() - 1),  # Anzahl der Prozesse (einen Kern für System-Prozesse reservieren)
    'chunk_size': 1000,  # Größe der Daten-Chunks für parallele Verarbeitung
    
    # Parallele Evaluation
    'parallel_evaluation': True,  # Parallele Evaluation aktivieren
    'eval_workers': max(1, min(8, mp.cpu_count() - 1)),  # Anzahl der Prozesse für Evaluation (begrenzt auf 5)
    'eval_episodes': 5,  # Anzahl der Evaluierungs-Episoden
    
    # Paralleles Training
    'parallel_training': True,  # Parallele Training-Episoden aktivieren
    'training_workers': max(1, min(8, mp.cpu_count() - 1)),  # Anzahl der Prozesse für Training (begrenzt auf 4)
    'parallel_episodes': 5,  # Anzahl der parallel zu trainierenden Episoden
}


# ================= 1. Datenverarbeitungsmodul =================

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.scaler = None
        self.feature_columns = None
    
    def load_data(self):
        """Lädt die Daten aus der CSV-Datei und konvertiert das Datum"""
        print("\n📂 Lade Daten...")
        df = pd.read_csv(self.config['data_path'], delimiter=';')
        
        # Konvertiere Datum und Uhrzeit - mit explizitem dayfirst=True für europäisches Datumsformat
        df['DateTime'] = pd.to_datetime(df['Timestamp'], dayfirst=True)
        df.set_index('DateTime', inplace=True)
        df.drop(columns=['Timestamp'], inplace=True, errors='ignore')
        
        # Entferne Zeilen mit NaN-Werten
        initial_len = len(df)
        df.dropna(inplace=True)
        if initial_len > len(df):
            print(f"⚠️  {initial_len - len(df)} Zeilen mit NaN-Werten entfernt")
        
        # Merke die Feature-Spalten
        self.feature_columns = df.columns.tolist()
        print(f"✅ Verfügbare Feature-Spalten: {self.feature_columns}")
        
        # Filtere die ausgewählten Features
        features_to_use = self.config['features_to_use']
        if features_to_use and len(features_to_use) > 0:
            # Überprüfe, ob alle ausgewählten Features existieren
            missing_features = [f for f in features_to_use if f not in self.feature_columns]
            if missing_features:
                print(f"⚠️  WARNUNG: Folgende ausgewählte Features wurden nicht gefunden: {missing_features}")
                # Entferne nicht existierende Features aus der Liste
                features_to_use = [f for f in features_to_use if f in self.feature_columns]
            
            # Filtere Daten nach ausgewählten Features
            df = df[features_to_use]
            print(f"🔍 Verwende nur folgende Features: {features_to_use}")
        
        # Aktualisiere feature_columns nach dem Filtern
        self.feature_columns = df.columns.tolist()
        
        # Setze die state_dim auf die Anzahl der ausgewählten Features
        if self.config['state_dim'] is None:
            self.config['state_dim'] = len(self.feature_columns)
            print(f"✅ state_dim automatisch auf {self.config['state_dim']} gesetzt")
        
        print(f"📊 Datensatz geladen: {len(df)} Zeilen, Zeitraum: {df.index[0]} bis {df.index[-1]}")
        return df
        
    def load_validation_data(self):
        """Lädt die Validierungsdaten aus einer separaten CSV-Datei"""
        print("Lade externe Validierungsdaten...")
        try:
            # Prüfe ob die Validierungsdatei existiert
            if not os.path.exists(self.config['validation_csv_path']):
                print(f"WARNUNG: Validierungsdatei nicht gefunden: {self.config['validation_csv_path']}")
                return None
                
            # Lade die CSV-Datei
            df_val = pd.read_csv(self.config['validation_csv_path'], delimiter=';')
            
            # Konvertiere Datum und Uhrzeit
            df_val['DateTime'] = pd.to_datetime(df_val['Timestamp'])
            df_val.set_index('DateTime', inplace=True)
            df_val.drop(columns=['Timestamp'], inplace=True, errors='ignore')
            
            # Entferne Zeilen mit NaN-Werten
            df_val.dropna(inplace=True)
            
            # Überprüfe, ob die erforderlichen Feature-Spalten vorhanden sind
            features_to_use = self.config['features_to_use']
            missing_columns = [f for f in features_to_use if f not in df_val.columns]
            if missing_columns:
                print(f"WARNUNG: Folgende Features fehlen in der Validierungsdatei: {missing_columns}")
                print("Externe Validierung kann nicht durchgeführt werden.")
                return None
                
            # Filtere die Validierungsdaten nach den gleichen Features wie die Trainingsdaten
            df_val = df_val[features_to_use]
            
            print(f"Externe Validierungsdaten geladen: {len(df_val)} Datenpunkte")
            return df_val
            
        except Exception as e:
            print(f"Fehler beim Laden der Validierungsdaten: {e}")
            return None
    
    def _create_sequence_chunk(self, data_chunk, sequence_length):
        """Erzeugt Sequenzen für einen Datenblock"""
        X, y = [], []
        chunk_len = len(data_chunk)
        
        if chunk_len <= sequence_length:
            return [], []  # Zu wenig Daten für eine Sequenz
            
        for i in range(chunk_len - sequence_length):
            X.append(data_chunk[i:i + sequence_length])
            y.append(data_chunk[i + sequence_length])
        
        return X, y

    def _process_chunk_wrapper(self, args):
        """Eine Hilfsfunktion für die parallele Verarbeitung, die pickle-bar ist"""
        chunk, sequence_length = args
        return self._create_sequence_chunk(chunk, sequence_length)
    
    def preprocess_data_parallel(self, df):
        """Skaliert die Daten und erstellt Sequenzen für das LSTM-Modell mit paralleler Verarbeitung"""
        print("\n🔄 Vorverarbeitung der Daten mit Multiprocessing...")
        
        # Prüfe, ob Multiprocessing aktiviert ist
        if not self.config.get('use_multiprocessing', True):
            # Fallback zur Standard-Methode ohne Multiprocessing
            return self.preprocess_data(df)
        
        # Initialisiere MinMaxScaler und skaliere die Daten
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(df)
        
        # Speichere den Scaler für spätere Verwendung
        joblib.dump(self.scaler, self.config['scaler_save_path'])
        
        # Bestimme die Anzahl der Worker
        num_workers = self.config.get('num_workers', max(1, mp.cpu_count() - 1))
        print(f"🖥️  Verwende {num_workers} CPU-Kerne für die parallele Sequenzgenerierung")
        
        # Berechne die optimale Chunk-Größe
        total_samples = len(scaled_data)
        chunk_size = self.config.get('chunk_size', 1000)
        
        # Teile die Daten in Chunks für parallele Verarbeitung
        chunks = []
        args_list = []
        for i in range(0, total_samples, chunk_size):
            end_idx = min(i + chunk_size + self.config['sequence_length'], total_samples)
            chunk = scaled_data[i:end_idx]
            chunks.append(chunk)
            args_list.append((chunk, self.config['sequence_length']))  # Tuple mit Argumenten
        
        # Erzeuge Sequenzen parallel mit einer pickle-baren Funktion
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Verwende die _process_chunk_wrapper Methode anstelle einer Lambda
            results = list(executor.map(self._process_chunk_wrapper, args_list))
        
        # Kombiniere Ergebnisse
        X_list, y_list = [], []
        for X_chunk, y_chunk in results:
            X_list.extend(X_chunk)
            y_list.extend(y_chunk)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        processing_time = time.time() - start_time
        print(f"⏱️  Parallele Sequenzgenerierung abgeschlossen in {processing_time:.2f} Sekunden")
        print(f"📊 Generierte Sequenzen: {len(X)}")
        
        # Teile in Trainings-, Validierungs- und Testdaten
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], shuffle=False)
        
        val_size = self.config['validation_size'] / (1 - self.config['test_size'])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, shuffle=False)
        
        print(f"📋 Daten vorverarbeitet:")
        print(f"   - X_train: {X_train.shape} ({len(X_train)} Sequenzen)")
        print(f"   - X_val: {X_val.shape} ({len(X_val)} Sequenzen)")
        print(f"   - X_test: {X_test.shape} ({len(X_test)} Sequenzen)")
        
        # Konvertiere direkt zu PyTorch-Tensoren für schnellere Verarbeitung
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.FloatTensor(y_val).to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.FloatTensor(y_test).to(device)
        
        print(f"🚀 Daten auf {device} übertragen")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, X_train_tensor, X_val_tensor, X_test_tensor
    
    def preprocess_validation_data(self, df_val):
        """Verarbeitet externe Validierungsdaten mit dem vorhandenen Scaler"""
        print("Verarbeite externe Validierungsdaten...")
        
        # Überprüfen ob der Scaler vorhanden ist
        if self.scaler is None:
            # Versuche den Scaler zu laden
            try:
                self.scaler = joblib.load(self.config['scaler_save_path'])
                print("Scaler aus Datei geladen")
            except Exception as e:
                print(f"Fehler beim Laden des Scalers: {e}")
                print("Externe Validierung nicht möglich ohne Scaler")
                return None
        
        # Skaliere die Validierungsdaten mit dem trainierten Scaler
        try:
            scaled_data = self.scaler.transform(df_val)
            
            # Erstelle Sequenzen für LSTM
            X, y = [], []
            for i in range(len(scaled_data) - self.config['sequence_length']):
                X.append(scaled_data[i:i + self.config['sequence_length']])
                y.append(scaled_data[i + self.config['sequence_length']])
            
            X_val = np.array(X)
            y_val = np.array(y)
            
            print(f"Externe Validierungsdaten verarbeitet - X_val: {X_val.shape}")
            
            # Erstelle Tensoren für GPU falls verfügbar
            X_val_tensor = torch.FloatTensor(X_val).to(device)
            y_val_tensor = torch.FloatTensor(y_val).to(device)
            
            return X_val, y_val, X_val_tensor, y_val_tensor
            
        except Exception as e:
            print(f"Fehler bei der Verarbeitung der externen Validierungsdaten: {e}")
            return None
    

# ================= 2. Trading-Umgebung =================

class TradingEnvironment:
    def __init__(self, data, config, use_gpu=True):
        """
        Trading-Umgebung für Reinforcement Learning
        
        Args:
            data: Skalierte Daten als NumPy-Array mit Form (n_samples, n_features)
            config: Dictionary mit Konfigurationsparametern
            use_gpu: Ob GPU für Berechnungen verwendet werden soll (wenn verfügbar)
        """
        self.config = config
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Speichere Daten auf der CPU oder GPU
        if self.use_gpu and not isinstance(data, torch.Tensor):
            # Konvertiere zu Tensor und verschiebe auf GPU
            self.data_tensor = torch.FloatTensor(data).to(device)
            self.data = data  # Behalte auch die NumPy-Version für Kompatibilität
            print("Trading-Umgebung verwendet GPU-Tensoren")
        else:
            self.data = data
            self.data_tensor = None
        
        # Cache für häufig verwendete Daten
        self.cached_states = {}
        
        self.reset()
    
    def reset(self):
        """Setzt die Umgebung zurück"""
        self.current_step = self.config['sequence_length']
        self.balance = self.config['initial_balance']
        self.position = 0  # 0: keine Position, 1: Long, -1: Short
        self.entry_price = 0
        self.entry_time = 0
        self.cooldown_counter = 0  # Countdown-Zähler für die Wartezeit nach Positionsschließung
        self.done = False
        self.history = []
        
        # Leere den Cache
        self.cached_states = {}
        
        # Initialer Zustand
        return self._get_state()
    
    def _get_state(self):
        """Erstellt den aktuellen Zustand für den Agenten mit optionaler GPU-Beschleunigung"""
        # Überprüfe zuerst den Cache
        if self.current_step in self.cached_states:
            return self.cached_states[self.current_step]
            
        # Marktdaten: Die letzten sequence_length Zeitschritte
        start_idx = max(0, self.current_step - self.config['sequence_length'])
        end_idx = self.current_step
        
        try:
            if self.use_gpu and self.data_tensor is not None:
                # GPU-Pfad: Verwende PyTorch-Tensoren direkt
                if len(self.data_tensor.shape) == 3:  # 3D-Tensor
                    current_idx = min(self.current_step, len(self.data_tensor) - 1)
                    market_data = self.data_tensor[current_idx].clone()
                else:  # 2D-Tensor
                    market_data = self.data_tensor[start_idx:end_idx].clone()
                    
                # Sicherstellung der korrekten Form
                if len(market_data.shape) != 2:
                    if len(market_data.shape) == 1:
                        market_data = market_data.unsqueeze(0)
                    elif len(market_data.shape) == 3:
                        market_data = market_data[-1]
                
                # Padding falls notwendig
                if market_data.shape[0] < self.config['sequence_length']:
                    padding_size = self.config['sequence_length'] - market_data.shape[0]
                    padding = torch.zeros(padding_size, market_data.shape[1], device=device)
                    market_data = torch.cat([padding, market_data], dim=0)
                    
                # Diese Berechnung erfolgt weiterhin auf der CPU da sie einfach ist
                position = float(self.position)
                holding_time = 0.0
                
                if self.position != 0:
                    holding_time = float(self.current_step - self.entry_time) / max(1, self.config['max_holding_period'])
                
                pnl = 0.0
                if self.position != 0 and self.entry_price > 0.0001:
                    try:
                        if market_data.shape[1] > 3:
                            current_price = float(market_data[-1, 3].item())
                        else:
                            current_price = 1.0
                        
                        pnl = (current_price - self.entry_price) / self.entry_price
                        if torch.isnan(torch.tensor(pnl)) or torch.isinf(torch.tensor(pnl)):
                            pnl = 0.0
                    except Exception as e:
                        pnl = 0.0
                
                position_info = torch.tensor([position, holding_time, pnl], device=device)
                
                # Für die Schnittstelle mit dem restlichen Code, konvertieren wir zurück zu NumPy
                # Dies ist ineffizient, aber notwendig für die Kompatibilität
                state = {
                    'market_data': market_data.cpu().numpy(),
                    'position_info': position_info.cpu().numpy()
                }
                
                # Cache das Ergebnis
                self.cached_states[self.current_step] = state
                return state
                
            else:
                # CPU-Pfad (ursprünglicher Code)
                # Marktdaten: Die letzten sequence_length Zeitschritte
                start_idx = max(0, self.current_step - self.config['sequence_length'])
                end_idx = self.current_step
                
                if len(self.data.shape) == 3:  # 3D-Array (n_samples, sequence_length, n_features)
                    # Wenn self.data bereits ein 3D-Array ist, nehmen wir die letzten Zeitschritte des aktuellen Samples
                    current_idx = min(self.current_step, len(self.data) - 1)
                    market_data = self.data[current_idx].copy()
                else:  # 2D-Array (n_samples, n_features)
                    # Wenn self.data ein 2D-Array ist, nehmen wir die letzten sequence_length Samples
                    market_data = self.data[start_idx:end_idx].copy()
                    
                # Überprüfe die Dimension von market_data
                if len(market_data.shape) != 2:
                    if len(market_data.shape) == 1:
                        # Wenn 1D, erweitere zu 2D
                        market_data = market_data.reshape(1, -1)
                    elif len(market_data.shape) == 3:
                        # Wenn 3D, nehme den letzten Zeitschritt
                        market_data = market_data[-1]

                # Stelle sicher, dass die Länge korrekt ist
                if market_data.shape[0] < self.config['sequence_length']:
                    padding = np.zeros((self.config['sequence_length'] - market_data.shape[0], market_data.shape[1]))
                    market_data = np.vstack((padding, market_data))
                    
                # Positionsdaten - mit Fehlerbehandlung für Nulldivisionen
                position = float(self.position)  # Konvertiere zu Skalar
                holding_time = 0.0
                
                # Überprüfe, ob self.position ein Skalar ist, nicht ein Array
                if isinstance(self.position, (int, float)) and self.position != 0:
                    holding_time = float(self.current_step - self.entry_time) / max(1, self.config['max_holding_period'])
                
                # Aktuelle P&L mit Fehlerbehandlung
                pnl = 0.0
                
                # Überprüfe, ob entry_price ein Skalar ist
                if isinstance(self.position, (int, float)) and isinstance(self.entry_price, (int, float)):
                    if self.position != 0 and self.entry_price > 0.0001:  # Vermeide Division durch sehr kleine Werte
                        try:
                            # Verwende den aktuellen Preis aus den Marktdaten wenn möglich
                            if len(market_data) > 0 and market_data.shape[1] > 3:
                                current_price = float(market_data[-1, 3])  # Nehme den Close-Preis aus dem letzten Zeitschritt
                            else:
                                # Fallback-Wert
                                current_price = 1.0
                            
                            pnl = (current_price - self.entry_price) / self.entry_price
                            # Überprüfe auf NaN und ersetze sie
                            if np.isnan(pnl) or np.isinf(pnl):
                                pnl = 0.0
                        except Exception as e:
                            print(f"Fehler bei P&L-Berechnung: {e}")
                            pnl = 0.0
                
                position_info = np.array([
                    position,           # Aktuelle Position
                    holding_time,       # Normalisierte Haltedauer
                    pnl                 # Aktueller P&L
                ])
                
                # Kombiniere Markt- und Positionsdaten
                state = {
                    'market_data': market_data,
                    'position_info': position_info
                }
                
                # Cache das Ergebnis
                self.cached_states[self.current_step] = state
                return state
                
        except Exception as e:
            print(f"Fehler beim Extrahieren der Marktdaten: {e}")
            print(f"Datenform: {self.data.shape if hasattr(self, 'data') else 'unbekannt'}, current_step: {self.current_step}")
            # Fallback: Erstelle ein Nullarray mit korrekter Form
            market_data = np.zeros((self.config['sequence_length'], self.config['state_dim']))
            position_info = np.array([0.0, 0.0, 0.0])
            
            state = {
                'market_data': market_data,
                'position_info': position_info
            }
            return state
    
    def step(self, action):
        """
        Führt einen Schritt in der Umgebung aus
        
        Args:
            action: 0 (Halten), 1 (Long)
            
        Returns:
            next_state: Der nächste Zustand
            reward: Die Belohnung für die Aktion
            done: Ob die Episode beendet ist
            info: Zusätzliche Informationen
        """
        # Aktionszuordnung: 0 = Halten, 1 = Long
        try:
            # Extrahiere den Close-Preis, High-Preis und Low-Preis mit GPU-Unterstützung, wenn möglich
            current_index = min(self.current_step, len(self.data) - 1)
            
            if self.use_gpu and self.data_tensor is not None:
                if len(self.data_tensor.shape) == 3:
                    current_price = self.data_tensor[current_index, -1, 3].item()  # Close-Preis
                    current_high = self.data_tensor[current_index, -1, 1].item()   # High-Preis
                    current_low = self.data_tensor[current_index, -1, 2].item()    # Low-Preis
                else:
                    current_price = self.data_tensor[current_index, 3].item()  # Close-Preis
                    current_high = self.data_tensor[current_index, 1].item()   # High-Preis
                    current_low = self.data_tensor[current_index, 2].item()    # Low-Preis
            else:
                # Ursprünglicher CPU-Pfad
                if len(self.data.shape) == 3:
                    current_price = self.data[current_index, -1, 3]  # Close-Preis
                    current_high = self.data[current_index, -1, 1]   # High-Preis
                    current_low = self.data[current_index, -1, 2]    # Low-Preis
                else:
                    current_price = self.data[current_index, 3]  # Close-Preis
                    current_high = self.data[current_index, 1]   # High-Preis
                    current_low = self.data[current_index, 2]    # Low-Preis
                
                # Stelle sicher, dass es ein Skalar ist
                if isinstance(current_price, np.ndarray):
                    current_price = current_price.item()
                    current_high = current_high.item()
                    current_low = current_low.item()
                else:
                    current_price = float(current_price)
                    current_high = float(current_high)
                    current_low = float(current_low)
            
            # Überprüfe auf gültige Werte
            if (self.use_gpu and (torch.isnan(torch.tensor(current_price)) or torch.isinf(torch.tensor(current_price)))) or \
               (not self.use_gpu and (np.isnan(current_price) or np.isinf(current_price))):
                current_price = 1.0
                current_high = 1.0
                current_low = 1.0
                print(f"Warnung: Ungültiger Preis an Position {self.current_step}, verwende Fallback-Wert")
        except Exception as e:
            print(f"Fehler beim Extrahieren des Preises: {e}")
            current_price = 1.0  # Fallback-Wert
            current_high = 1.0   # Fallback-Wert
            current_low = 1.0    # Fallback-Wert für Low-Preis
            
        reward = 0
        info = {}
        position_closed_this_step = False  # Neue Variable, um zu verfolgen, ob in diesem Schritt eine Position geschlossen wurde
        
        # Aktualisiere den Cooldown-Zähler, falls aktiv
        if self.cooldown_counter > 0:
            info['cooldown'] = self.cooldown_counter  # Für Debugging
            self.cooldown_counter -= 1
        
        # 1. Berechne Belohnung für aktuelle Position (falls vorhanden)
        if self.position != 0 and self.entry_price > 0.0001:  # Vermeide Division durch sehr kleine Werte
            # Berechne aktuellen Gewinn/Verlust (absoluter Preis)
            price_diff = current_price - self.entry_price
            high_diff = current_high - self.entry_price  # Differenz zwischen High und Einstiegspreis
            low_diff = current_low - self.entry_price    # Differenz zwischen Low und Einstiegspreis
            holding_duration = self.current_step - self.entry_time
            
            try:
                # Nur Long-Positionen betrachten
                reward = price_diff / self.entry_price if self.entry_price != 0 else 0
                
                # --- ÜBERARBEITETES REWARD-SYSTEM --- #
                
                # 1. Progressive Belohnung für steigende Gewinne
                # Je höher der Gewinn, desto höher der Belohnungsfaktor
                # Dies fördert das Ausnutzen von Kurssteigerungen und das Erkennen von Trends
                if reward > 0:
                    # Progressive Belohnung basierend auf dem prozentualen Gewinn
                    take_profit_threshold = self.config['take_profit'] / self.entry_price
                    if reward >= take_profit_threshold * 0.8:  # Näherung an Take-Profit
                        reward *= 1.2  # Erhöhte Belohnung für nahende Take-Profit-Situationen
                    elif reward >= take_profit_threshold * 0.5:  # Auf halbem Weg zum Take-Profit
                        reward *= 1.1  # Leicht erhöhte Belohnung
                
                # 2. Neutrale bis leicht negative Belohnung für Seitwärtsbewegungen
                # Dies reduziert unnötige Positionshaltezeiten in Seitwärtsmärkten
                if abs(reward) < (self.config['take_profit'] / self.entry_price) * 0.2:
                    # Kleiner negativer Anreiz, der mit der Zeit größer wird
                    sideways_penalty = 0.0003 * min(holding_duration / 10, 1.0)
                    reward -= sideways_penalty  # Leichte Bestrafung für Seitwärtsbewegungen
                
                # 3. Dynamisiertes Momentum-basiertes Reward-System
                # Verwende den Trend der letzten N Ticks für zusätzliche Belohnungen/Strafen
                # Dies verbessert das Timing für Ein- und Ausstieg
                if holding_duration > 5:  # Nach einigen Ticks beginnen wir mit der Momentum-Analyse
                    # Approximieren wir das Momentum mit dem aktuellen Gewinn/Verlust im Vergleich zum vorherigen
                    momentum_factor = 0.0005 * (1.0 + min(holding_duration / 20, 1.0))
                    
                    if price_diff > 0:
                        # Bei positivem Trend: zusätzliche kleine Belohnung
                        reward += momentum_factor
                    elif price_diff < 0:
                        # Bei negativem Trend: zusätzliche kleine Bestrafung
                        reward -= momentum_factor * 1.2  # Leicht verstärkte Bestrafung für negative Trends
            except Exception as e:
                print(f"Fehler bei Reward-Berechnung: {e}, price_diff: {price_diff}, entry_price: {self.entry_price}")
                reward = 0
            
            # Regelbasierte Ausstiegslogik mit absoluten USD-Werten
            exit_signal = False
            
            try:
                # Stop-Loss mit Low-Preis und Take-Profit mit High-Preis als absolute USD-Werte
                # KORRIGIERTER CODE: Prüfe ob High-Differenz GRÖSSER ODER GLEICH dem Take-Profit ist
                if low_diff <= -self.config['stop_loss']:  # Stop-Loss (0.2 USD) - Vergleiche mit Low-Preis
                    # --- ÜBERARBEITETER STOP-LOSS REWARD --- #
                    # Differenzierter Stop-Loss-Reward basierend auf der Haltedauer
                    stop_loss_severity = 2.0  # Basis-Multiplikator
                    
                    # Frühzeitiger Stop-Loss ist schlimmer als später Stop-Loss
                    if holding_duration < self.config['max_holding_period'] * 0.2:
                        stop_loss_severity = 2.5  # Erhöhte Bestrafung für sehr frühen Stop-Loss 
                    elif holding_duration > self.config['max_holding_period'] * 0.6:
                        stop_loss_severity = 1.2  # Reduzierte Bestrafung für späten Stop-Loss
                        
                    reward *= stop_loss_severity  # Verstärkte negative Belohnung bei Stop-Loss
                    exit_signal = True
                    info['exit_reason'] = 'stop_loss'
                    info['low_diff'] = low_diff  # Für Debugging
                    info['entry_price'] = self.entry_price
                    info['low_price'] = current_low
                
                # WICHTIG: Take-Profit-Prüfung mit High-Preis statt Close-Preis
                # Take-Profit als absoluter USD-Wert
                # KORRIGIERTER CODE: Prüfe ob High-Differenz GRÖSSER ODER GLEICH dem Take-Profit ist
                elif high_diff >= self.config['take_profit']:  # Take-Profit (0.12 USD) - Vergleiche mit High-Preis
                    # --- ÜBERARBEITETER TAKE-PROFIT REWARD --- #
                    # Differenzierte Belohnung basierend auf:
                    # 1. Geschwindigkeit (wie schnell wurde der Take-Profit erreicht)
                    # 2. Marktbedingungen (war es ein starker Trend oder volatiler Markt)
                    
                    # Bonus für schnelles Erreichen des Take-Profits
                    speed_bonus = max(0, 1.0 - (holding_duration / (self.config['max_holding_period'] * 0.6)))
                    # Je schneller der Take-Profit erreicht wird, desto höher der Bonus (bis zu 1.0)
                    
                    # Basisbelohnung plus Geschwindigkeitsbonus
                    reward_multiplier = 2.0 + (speed_bonus * 1.5)
                    
                    # Zusätzlicher Bonus für signifikante Überschreitung des Take-Profits
                    if high_diff > self.config['take_profit'] * 1.5:
                        reward_multiplier += 0.5  # Bonus für deutliche Überschreitung des Take-Profit
                    
                    reward *= reward_multiplier
                    exit_signal = True
                    info['exit_reason'] = 'take_profit'
                    info['high_diff'] = high_diff  # Für Debugging
                    info['entry_price'] = self.entry_price
                    info['high_price'] = current_high
            except Exception as e:
                print(f"Fehler bei Ausstiegslogik: {e}")
                # Kein Exit-Signal setzen
            
            # --- ÜBERARBEITETE MAXIMALE HALTEDAUER --- #
            # Maximale Haltedauer erreicht - mit angepasster dynamischer Bestrafung
            if self.current_step - self.entry_time >= self.config['max_holding_period']:
                # Differenzierte Bestrafung basierend auf der aktuellen P&L-Situation
                
                # Grundstrafe, die für alle Fälle gilt
                base_penalty = -0.015  # Reduzierte Grundstrafe (war -0.02)
                
                # Dynamische Anpassung basierend auf der aktuellen Position
                if reward < 0:
                    # Bei bereits negativem Reward: verstärkte Bestrafung aber weniger als zuvor
                    # Dies erkennt an, dass der Agent bereits "bestraft" wurde durch die negative Position
                    reward *= 1.75  # Reduziert von 2.0
                else:
                    # Bei positivem Reward: stärkere Reduktion
                    # Der Agent hat versäumt, eine profitable Position rechtzeitig zu schließen
                    reward *= 0.2  # Reduziert von 0.25, stärkere Bestrafung für das Verpassen von Gewinnen
                
                # Geringere Bestrafung, wenn die Position nahe am Break-Even ist
                if abs(reward) < 0.002:
                    base_penalty = -0.01  # Mildere Strafe für neutrale Positionen
                
                # Füge die angepasste Grundstrafe hinzu
                reward += base_penalty
                
                exit_signal = True
                info['exit_reason'] = 'max_holding'
                # Debug-Ausgabe stark reduziert - nur noch bei jeder 50. Position
                if random.random() < 0.0002:  # Ungefähr 2% der Fälle
                    print(f"Max holding period erreicht: current_step={self.current_step}, entry_time={self.entry_time}, diff={self.current_step - self.entry_time}, max={self.config['max_holding_period']}, penalty={base_penalty}, final_reward={reward:.4f}")
            
            # Position schließen wenn Exit-Signal
            if exit_signal:
                # Position schließen
                try:
                    if self.entry_price > 0:
                        # Berechne absoluten Gewinn/Verlust in USD und übertrage auf Balance
                        
                        # KORREKTUR: Gewinn/Verlust je nach Ausstiegsgrund
                        if info.get('exit_reason') == 'take_profit':
                            # Verwende GENAU den take_profit-Wert für die Gewinnberechnung
                            # anstatt die vollständige high_diff
                            pnl = self.config['take_profit'] * self.config['position_size']
                        elif info.get('exit_reason') == 'stop_loss':
                            # Verwende GENAU den stop_loss-Wert für die Verlustberechnung
                            # anstatt die vollständige low_diff
                            pnl = -self.config['stop_loss'] * self.config['position_size']
                        else:
                            # Bei max_holding verwende den Close-Preis
                            pnl = price_diff * self.config['position_size']
                            
                        self.balance += pnl
                except Exception as e:
                    print(f"Fehler beim Schließen der Position: {e}")
                
                self.position = 0
                self.entry_price = 0
                self.entry_time = 0  # Zurücksetzen der entry_time, um Tracker leichter überprüfen zu können
                
                # Aktiviere die Cooldown-Periode nach dem Schließen einer Position
                self.cooldown_counter = self.config['cooldown_period']
                
                info['action_taken'] = 'closed_position'
                info['holding_duration'] = holding_duration
                position_closed_this_step = True  # Markiere, dass in diesem Schritt eine Position geschlossen wurde
        
        # 2. Neue Position eröffnen (nur Long), falls keine aktive Position UND nicht gerade geschlossen
        # WICHTIG: Jetzt auch prüfen, ob wir nicht in einer Cooldown-Periode sind
        if self.position == 0 and action == 1 and current_price > 0 and not position_closed_this_step and self.cooldown_counter == 0:
            # --- ÜBERARBEITETE EINSTIEGS-BELOHNUNG --- #
            # Feiner abgestimmte initiale Bestrafung für das Eröffnen einer Position
            # Diese Bestrafung ist wichtig, um übermäßiges Trading zu vermeiden,
            # sollte aber nicht zu stark sein, um legitime Handelsgelegenheiten nicht zu bestrafen
            
            # Basisstrafe für jede Positionseröffnung
            entry_penalty = -0.00075  # Leicht reduziert von -0.001
            
            # Füge die Einstiegsstrafe zum Reward hinzu
            reward += entry_penalty
            
            self.position = 1  # Nur Long-Position
            self.entry_price = current_price
            self.entry_time = self.current_step
            info['action_taken'] = 'opened_position'
        elif action == 1 and self.cooldown_counter > 0:
            # --- ÜBERARBEITETE COOLDOWN-BESTRAFUNG --- #
            # Logarithmisch abnehmende Bestrafung während der Cooldown-Periode
            # Dies verhindert zu häufiges Handeln, ohne den Agenten zu stark zu bestrafen
            cooldown_penalty = -0.0002 * (1 - (self.cooldown_counter / self.config['cooldown_period']))
            reward += cooldown_penalty
            info['cooldown_blocked'] = True  # Für Debugging
        
        # Nächster Zeitschritt
        self.current_step += 1
        self.cached_states.pop(self.current_step - 1, None)  # Entferne alten gecachten Zustand
        
        # Überprüfe, ob die Episode beendet ist
        data_len = len(self.data_tensor if self.use_gpu and self.data_tensor is not None else self.data)
        if self.current_step >= data_len - 1:
            self.done = True
        
        # Speichere den aktuellen Zustand in der Historie
        self.history.append({
            'step': self.current_step,
            'price': current_price,
            'high': current_high,
            'low': current_low,
            'position': self.position,
            'balance': self.balance,
            'reward': reward,
            'cooldown': self.cooldown_counter
        })
        
        # Rückgabe
        return self._get_state(), reward, self.done, info


# ================= 3. LSTM-RL-Agent =================

class ReplayBuffer:
    def __init__(self, capacity, gpu_accelerated=True):
        self.buffer = deque(maxlen=capacity)
        self.gpu_accelerated = gpu_accelerated and torch.cuda.is_available()
        
        # Zwischenspeicher für GPU-beschleunigte Batches
        if self.gpu_accelerated:
            print("Verwende GPU-beschleunigten Replay-Buffer")
            # Pre-allokierte Tensoren für Batch-Sampling
            self.preallocated_market_data = None
            self.preallocated_position_info = None
            self.preallocated_next_market_data = None
            self.preallocated_next_position_info = None
    
    def push(self, state, action, reward, next_state, done):
        if self.gpu_accelerated:
            # Bei GPU-Beschleunigung speichern wir nur die Referenzen und halten die Tensoren direkt auf der GPU
            if isinstance(state['market_data'], torch.Tensor):
                market_data = state['market_data'].detach().cpu().numpy()
                position_info = state['position_info'].detach().cpu().numpy() if isinstance(state['position_info'], torch.Tensor) else state['position_info']
            else:
                market_data = state['market_data']
                position_info = state['position_info']
                
            if isinstance(next_state['market_data'], torch.Tensor):
                next_market_data = next_state['market_data'].detach().cpu().numpy()
                next_position_info = next_state['position_info'].detach().cpu().numpy() if isinstance(next_state['position_info'], torch.Tensor) else next_state['position_info']
            else:
                next_market_data = next_state['market_data']
                next_position_info = next_state['position_info']
                
            state_store = {
                'market_data': market_data,
                'position_info': position_info
            }
            
            next_state_store = {
                'market_data': next_market_data,
                'position_info': next_position_info
            }
            
            self.buffer.append((state_store, action, reward, next_state_store, done))
        else:
            # Bei CPU-Beschleunigung speichern wir alles wie gewohnt
            self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        indices = random.sample(range(len(self.buffer)), batch_size)
        
        if self.gpu_accelerated:
            batch = [self.buffer[i] for i in indices]
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # Stack market_data und position_info separat
            market_data_batch = np.stack([s['market_data'] for s in states])
            position_info_batch = np.stack([s['position_info'] for s in states])
            next_market_data_batch = np.stack([s['market_data'] for s in next_states])
            next_position_info_batch = np.stack([s['position_info'] for s in next_states])
            
            # Konvertiere zu Tensoren und übertrage auf GPU
            market_data_batch = torch.FloatTensor(market_data_batch).to(device)
            position_info_batch = torch.FloatTensor(position_info_batch).to(device)
            next_market_data_batch = torch.FloatTensor(next_market_data_batch).to(device)
            next_position_info_batch = torch.FloatTensor(next_position_info_batch).to(device)
            
            # Konvertiere Actions, Rewards und Done-Flags zu Tensoren
            action_batch = torch.LongTensor(actions).unsqueeze(1).to(device)
            reward_batch = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            done_batch = torch.FloatTensor(dones).unsqueeze(1).to(device)
            
            # Rekonstruiere die Batch-Dictionaries für den Return-Wert
            states_batch = [{'market_data': market_data_batch[i], 'position_info': position_info_batch[i]} for i in range(batch_size)]
            next_states_batch = [{'market_data': next_market_data_batch[i], 'position_info': next_position_info_batch[i]} for i in range(batch_size)]
            
            # Gebe zusätzlich die vorbereiteten Tensoren zurück für effizienten Modell-Update
            return states_batch, actions, rewards, next_states_batch, dones, market_data_batch, position_info_batch, next_market_data_batch, next_position_info_batch, action_batch, reward_batch, done_batch
            
        else:
            # Bei CPU-Beschleunigung standard sampling
            batch = random.sample(self.buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class LSTMQNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMQNetwork, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc_position = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.ReLU()
        )
        
        self.fc_combined = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, market_data, position_info):
        # LSTM für Marktdaten
        lstm_out, _ = self.lstm(market_data)
        lstm_out = lstm_out[:, -1, :]  # Nehme nur den letzten Zeitschritt
        
        # Verarbeite Positionsdaten
        position_features = self.fc_position(position_info)
        
        # Kombiniere Features
        combined = torch.cat((lstm_out, position_features), dim=1)
        
        # Q-Werte berechnen
        q_values = self.fc_combined(combined)
        
        return q_values


class LSTMRLAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.config = config
        
        self.policy_network = LSTMQNetwork(state_dim, hidden_dim, action_dim, num_layers=config['num_layers']).to(device)
        self.target_network = LSTMQNetwork(state_dim, hidden_dim, action_dim, num_layers=config['num_layers']).to(device)
        
        # Synchronisiere Target-Netzwerk mit Policy-Netzwerk zu Beginn
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()  # Target-Netzwerk im Evaluierungsmodus
        
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=config['learning_rate'])
        self.loss_fn = nn.MSELoss()
        
        self.epsilon = config['epsilon_start']
        self.memory = ReplayBuffer(config['memory_capacity'])
        
        self.steps_done = 0
    
    def select_action(self, state, eval_mode=False):
        """Wählt eine Aktion aus basierend auf dem aktuellen Zustand mit GPU-Beschleunigung"""
        # Epsilon-Greedy-Strategie
        if eval_mode:
            epsilon = 0.01  # Geringe Exploration im Evaluierungsmodus
        else:
            self.epsilon = max(self.config['epsilon_end'], self.epsilon * self.config['epsilon_decay'])
            epsilon = self.epsilon
        
        if random.random() > epsilon:
            # Greedy-Aktion mit GPU-Beschleunigung
            with torch.no_grad():
                try:
                    # Verarbeite die Marktdaten und stelle sicher, dass sie die richtige Form haben
                    market_data = state['market_data']
                    position_info = state['position_info']
                    
                    # Prüfen ob die Eingaben bereits Tensoren sind
                    is_market_data_tensor = isinstance(market_data, torch.Tensor)
                    is_position_info_tensor = isinstance(position_info, torch.Tensor)
                    
                    # Stelle sicher, dass market_data ein 2D Array ist (sequence_length, features)
                    if not is_market_data_tensor:
                        if len(market_data.shape) == 1:
                            # Wenn 1D, erweitere zu 2D
                            market_data = market_data.reshape(1, -1)
                        elif len(market_data.shape) == 3:
                            # Wenn 3D und die erste Dimension ist 1, dann squeeze
                            if market_data.shape[0] == 1:
                                market_data = market_data.squeeze(0)
                            else:
                                # Wenn 3D mit mehreren Samples, nehme das erste Sample
                                market_data = market_data[0]
                    else:
                        # Bei Tensor ähnlich vorgehen
                        if len(market_data.shape) == 1:
                            market_data = market_data.unsqueeze(0)
                        elif len(market_data.shape) == 3:
                            if market_data.shape[0] == 1:
                                market_data = market_data.squeeze(0)
                            else:
                                market_data = market_data[0]
                    
                    # Konvertiere zu Tensor und füge Batch-Dimension hinzu wenn nötig
                    if not is_market_data_tensor:
                        market_data = torch.FloatTensor(market_data).unsqueeze(0).to(device)
                    elif market_data.dim() == 2:  # Wenn bereits ein 2D-Tensor, füge Batch-Dim hinzu
                        market_data = market_data.unsqueeze(0)
                        
                    if not is_position_info_tensor:
                        position_info = torch.FloatTensor(position_info).unsqueeze(0).to(device)
                    elif position_info.dim() == 1:  # Wenn bereits ein 1D-Tensor, füge Batch-Dim hinzu
                        position_info = position_info.unsqueeze(0)
                    
                    # Forward pass durch das Netzwerk mit minimaler CPU-GPU-Kommunikation
                    q_values = self.policy_network(market_data, position_info)
                    
                    # Bleibe auf der GPU für die argmax-Operation
                    max_action = q_values.argmax(dim=1)
                    return max_action.item()  # Minimale CPU-GPU-Kommunikation: nur ein Skalar
                except Exception as e:
                    print(f"Fehler bei der Aktionsauswahl: {e}")
                    # Im Fehlerfall zufällige Aktion zurückgeben
                    return random.randrange(self.action_dim)
        else:
            # Zufällige Aktion
            return random.randrange(self.action_dim)
    
    def update_model(self):
        """Aktualisiert das Modell basierend auf einem Batch aus dem Replay-Speicher mit GPU-Beschleunigung"""
        if len(self.memory) < self.config['batch_size']:
            return
        
        try:
            # Optimiertes Sampling mit GPU-Beschleunigung
            if hasattr(self.memory, 'gpu_accelerated') and self.memory.gpu_accelerated:
                # Direktes GPU-Sampling, gibt bereits die vorbereiteten Tensoren zurück
                _, _, _, _, _, market_data_batch, position_info_batch, next_market_data_batch, \
                next_position_info_batch, action_batch, reward_batch, done_batch = self.memory.sample(self.config['batch_size'])
                
                # Tensoren sind bereits auf dem richtigen Gerät (GPU)
            else:
                # Standard-Sampling für CPU-Modus
                states, actions, rewards, next_states, dones = self.memory.sample(self.config['batch_size'])
                
                # Batch-Arrays für Netzwerkeingaben erstellen
                market_data_batch = torch.FloatTensor(np.stack([s['market_data'] for s in states])).to(device)
                position_info_batch = torch.FloatTensor(np.stack([s['position_info'] for s in states])).to(device)
                
                next_market_data_batch = torch.FloatTensor(np.stack([s['market_data'] for s in next_states])).to(device)
                next_position_info_batch = torch.FloatTensor(np.stack([s['position_info'] for s in next_states])).to(device)
                
                action_batch = torch.LongTensor(actions).unsqueeze(1).to(device)
                reward_batch = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                done_batch = torch.FloatTensor(dones).unsqueeze(1).to(device)
            
            # Aktuelle Q-Werte mit CUDA-Beschleunigung
            current_q = self.policy_network(market_data_batch, position_info_batch).gather(1, action_batch)
            
            # Nächste Q-Werte (von Target-Netzwerk) mit CUDA-Beschleunigung
            with torch.no_grad():
                next_q = self.target_network(next_market_data_batch, next_position_info_batch).max(1)[0].unsqueeze(1)
                expected_q = reward_batch + self.config['gamma'] * next_q * (1 - done_batch)
            
            # Loss berechnen und Optimierungsschritt
            loss = self.loss_fn(current_q, expected_q)
            self.optimizer.zero_grad()
            
            # Optimierter Backward-Pass
            loss.backward()
            
            # Gradient Clipping für stabiles Training
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
            
            # Optimierter Update-Schritt
            self.optimizer.step()
            
            # Gibt den Loss als Python-Skalar zurück (mit CUDA-Synchronisation)
            return loss.item()
        except Exception as e:
            print(f"Fehler beim Update des Modells: {e}")
            return None
    
    def update_target_network(self):
        """Aktualisiert das Target-Netzwerk mit den Gewichten des Policy-Netzwerks"""
        self.target_network.load_state_dict(self.policy_network.state_dict())
    
    def save_model(self, path):
        """Speichert das Modell"""
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"Modell gespeichert unter {path}")
    
    def load_model(self, path):
        """Lädt das Modell und behandelt unterschiedliche Hidden-Dimensionen"""
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path)
                
                # Versuche das Modell direkt zu laden
                try:
                    self.policy_network.load_state_dict(checkpoint['policy_network'])
                    self.target_network.load_state_dict(checkpoint['target_network'])
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    self.epsilon = checkpoint['epsilon']
                    print(f"Modell erfolgreich geladen von {path}")
                    return True
                except RuntimeError as e:
                    print(f"Warnung: Modell-Dimensionen stimmen nicht überein: {e}")
                    print("Versuche neues Modell mit aktuellen Dimensionen zu erstellen...")
                    
                    # Speichere den aktuellen Epsilon-Wert
                    if 'epsilon' in checkpoint:
                        self.epsilon = checkpoint['epsilon']
                    
                    # Wir starten mit einem frischen Modell mit den aktuellen Dimensionen
                    # Bei Bedarf könnte hier eine komplexere Umwandlung zwischen Dimensionen implementiert werden
                    print(f"Neues Modell mit hidden_dim={self.hidden_dim} wurde initialisiert.")
                    print("HINWEIS: Das Training beginnt mit frischen Gewichten, aber mit dem gespeicherten Epsilon-Wert.")
                    return False
            except Exception as e:
                print(f"Fehler beim Laden des Modells: {e}")
                return False
        return False
        
    def evaluate_with_external_data(self):
        """Evaluiert den Agenten auf externen Validierungsdaten"""
        print("\n🔄 Starte Validierung mit externen Daten...")
        
        # Lade die Daten für die externe Validierung
        data_processor = DataProcessor(self.config)
        df_val = data_processor.load_validation_data()
        
        if df_val is None:
            print("❌ Keine externen Validierungsdaten verfügbar.")
            return None
        
        # Verarbeite die externen Validierungsdaten
        processed_data = data_processor.preprocess_validation_data(df_val)
        if processed_data is None:
            print("❌ Fehler bei der Verarbeitung der externen Validierungsdaten.")
            return None
        
        X_val, y_val, X_val_tensor, y_val_tensor = processed_data
        
        # Erstelle eine Umgebung für die externe Validierung
        val_env = TradingEnvironment(X_val, self.config, use_gpu=True)
        
        # Führe die Evaluation durch
        total_reward = 0
        total_balance = 0
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        
        # Ein einzelner Durchlauf durch die Validierungsdaten
        state = val_env.reset()
        done = False
        trades = []
        
        # Für die CSV-Ausgabe: Alle Aktionen speichern mit Zeitstempel
        all_actions = []
        
        while not done:
            # Aktion mit minimaler Exploration wählen
            action = self.select_action(state, eval_mode=True)
            next_state, reward, done, info = val_env.step(action)
            
            # Alle Aktionen für die CSV-Ausgabe speichern
            all_actions.append({
                'step': val_env.current_step - 1,
                'action': action,  # 0: Hold, 1: Long
                'position': val_env.position,
                'reward': reward
            })
            
            # Trade-Informationen sammeln
            if 'action_taken' in info:
                if info['action_taken'] == 'opened_position':
                    trades.append({
                        'type': 'LONG' if val_env.position == 1 else 'SHORT',
                        'entry_step': val_env.current_step - 1,
                        'entry_price': val_env.entry_price,
                        'exit_step': None,
                        'exit_price': None,
                        'reward': 0
                    })
                elif info['action_taken'] == 'closed_position' and trades:
                    # Korrigierte Zeile: Überprüfe ob der letzte Trade existiert und keine exit_step hat
                    if len(trades) > 0 and 'exit_step' not in trades[-1] or trades[-1].get('exit_step') is None:
                        trades[-1]['exit_step'] = val_env.current_step - 1
                        
                        # Extrahiere den aktuellen Preis
                        if val_env.use_gpu and val_env.data_tensor is not None:
                            if len(val_env.data_tensor.shape) == 3:
                                exit_price = val_env.data_tensor[val_env.current_step-1, -1, 3].item()
                            else:
                                exit_price = val_env.data_tensor[val_env.current_step-1, 3].item()
                        else:
                            exit_price = val_env.data[val_env.current_step-1, 3] if len(val_env.data.shape) == 2 else val_env.data[val_env.current_step-1, -1, 3]
                            
                        trades[-1]['exit_price'] = exit_price
                        trades[-1]['reward'] = reward
                        trades[-1]['exit_reason'] = info.get('exit_reason', 'action')
                        
                        # Berechne, ob es ein gewinnbringender Trade war
                        if trades[-1]['type'] == 'LONG':
                            is_winning = trades[-1]['exit_price'] > trades[-1]['entry_price']
                        else:  # SHORT
                            is_winning = trades[-1]['exit_price'] < trades[-1]['entry_price']
                        
                        if is_winning:
                            winning_trades += 1
                        else:
                            losing_trades += 1
                        
                        total_trades += 1
            
            # Zustand aktualisieren
            state = next_state
            total_reward += reward
        
        # Performance-Metriken
        final_balance = val_env.balance
        total_balance = final_balance
        
        win_rate = winning_trades / max(1, total_trades) * 100
        
        avg_profit = 0
        avg_loss = 0
        max_profit = 0
        max_loss = 0
        
        # Berechne durchschnittlichen Gewinn/Verlust und maximalen Gewinn/Verlust
        if trades:
            completed_trades = [t for t in trades if 'exit_step' in t and t['exit_step'] is not None]
            
            if completed_trades:
                profits = []
                losses = []
                
                for trade in completed_trades:
                    profit_pct = ((trade['exit_price'] - trade['entry_price']) / trade['entry_price']) * 100
                    if trade['type'] == 'SHORT':
                        profit_pct = -profit_pct
                        
                    if profit_pct > 0:
                        profits.append(profit_pct)
                    else:
                        losses.append(profit_pct)
                
                if profits:
                    avg_profit = sum(profits) / len(profits)
                    max_profit = max(profits)
                
                if losses:
                    avg_loss = sum(losses) / len(losses)
                    max_loss = min(losses)
        
        # Neue Funktion: Exportiere die Aktionen und Trades als CSV
        self.export_validation_results_to_csv(df_val, all_actions, trades)
        
        print(f"\n✅ Externe Validierung abgeschlossen:")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📊 Performance-Metriken:")
        print(f"   • Gesamtbelohnung: {total_reward:.4f}")
        print(f"   • Endkapital: ${final_balance:.2f}")
        print(f"   • Kapitalveränderung: {((final_balance/self.config['initial_balance'])-1)*100:.2f}%")
        print(f"   • Anzahl Trades: {total_trades}")
        
        if total_trades > 0:
            profit_factor = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
            
            print(f"\n📈 Trade-Statistiken:")
            print(f"   • Gewinner: {winning_trades} ({win_rate:.2f}%)")
            print(f"   • Verlierer: {losing_trades} ({100-win_rate:.2f}%)")
            print(f"   • Durchschn. Gewinn: {avg_profit:.2f}%")
            print(f"   • Durchschn. Verlust: {avg_loss:.2f}%")
            print(f"   • Max. Gewinn: {max_profit:.2f}%")
            print(f"   • Max. Verlust: {max_loss:.2f}%")
            print(f"   • Profit-Faktor: {profit_factor:.2f}")
            
            # Ausstiegsgründe zählen
            exit_reasons = {}
            for trade in trades:
                if 'exit_reason' in trade and trade['exit_reason']:
                    exit_reasons[trade['exit_reason']] = exit_reasons.get(trade['exit_reason'], 0) + 1
            
            if exit_reasons:
                print(f"\n🚪 Ausstiegsgründe:")
                for reason, count in exit_reasons.items():
                    percent = (count / total_trades) * 100
                    print(f"   • {reason}: {count} ({percent:.1f}%)")
        
        # Detaillierte Trade-Analyse (gekürzt)
        if trades and total_trades > 0:
            print("\n🔍 Trade-Analyse (Top 5 Trades):")
            print(f"{'Typ':<6} | {'Einstieg':<8} | {'Ausstieg':<8} | {'P&L':<8} | {'Grund':<12}")
            print("-" * 50)
            
            # Sortiere Trades nach Profitabilität
            completed_trades = [t for t in trades if 'exit_step' in t and t['exit_step'] is not None]
            sorted_trades = sorted(completed_trades, 
                key=lambda t: ((t['exit_price'] - t['entry_price']) / t['entry_price']) if t['type'] == 'LONG' 
                else ((t['entry_price'] - t['exit_price']) / t['entry_price']), 
                reverse=True)
            
            # Zeige nur die Top 5 Trades
            for trade in sorted_trades[:5]:
                profit_loss = ((trade['exit_price'] - trade['entry_price']) / trade['entry_price']) * 100
                if trade['type'] == 'SHORT':
                    profit_loss = -profit_loss
                
                print(f"{trade['type']:<6} | {trade['entry_step']:<8} | {trade['exit_step']:<8} | {profit_loss:>6.2f}% | {trade['exit_reason']:<12}")
        
        return {
            'reward': total_reward,
            'balance': total_balance,
            'trades': total_trades,
            'win_rate': win_rate,
            'trade_details': trades,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
        }
        
    def export_validation_results_to_csv(self, original_df, all_actions, trades):
        """
        Exportiert die Validierungsergebnisse in zwei CSV-Dateien:
        1. Eine erweiterte Version der Test.csv mit Aktionen
        2. Eine detaillierte Trades-Liste
        
        Args:
            original_df: Das originale DataFrame der Validierungsdaten
            all_actions: Liste aller Aktionen mit zugehörigen Schritten
            trades: Liste der abgeschlossenen Trades
        """
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. Erstelle eine erweiterte Version der Test.csv mit Aktionen
            # Erstelle eine Kopie des originalen DataFrames
            df_with_actions = original_df.copy()
            
            # Initialisiere die neuen Spalten mit Standardwerten
            df_with_actions['Action'] = 0        # 0: Hold, 1: Long
            df_with_actions['Position'] = 0      # 0: Keine Position, 1: Long
            df_with_actions['Signal'] = "Hold"   # Textversion für bessere Lesbarkeit
            
            # Fülle die neuen Spalten basierend auf den Aktionen
            # Variablen zur Nachverfolgung der Position
            current_position = 0
            
            for action_info in all_actions:
                if action_info['step'] < len(df_with_actions):  # Sicherheitsüberprüfung
                    # Aktion und Position aus action_info extrahieren
                    action = action_info['action']
                    position = action_info['position']
                    
                    # Aktualisiere die Position in den Daten
                    df_with_actions.iloc[action_info['step'], df_with_actions.columns.get_loc('Action')] = action
                    df_with_actions.iloc[action_info['step'], df_with_actions.columns.get_loc('Position')] = position
                    
                    # Textversion der Signale für bessere Lesbarkeit - verbesserte Logik
                    if action == 1 and current_position == 0 and position == 1:
                        # Echtes Kaufsignal (von keiner Position zu Long)
                        signal = "Buy"
                    elif action == 1 and current_position == 1:
                        # Versuch, Long zu kaufen während bereits Long-Position besteht (inkonsistent)
                        signal = "Hold Long"  # Korrigiert zu "Hold Long" statt "Buy"
                    elif action == 0 and position == 1:
                        # Halte Long-Position
                        signal = "Hold Long"
                    elif action == 0 and position == 0:
                        # Keine Position
                        signal = "Hold"
                    elif current_position == 1 and position == 0:
                        # Position wurde geschlossen
                        signal = "Sell"  # Neues Signal für geschlossene Positionen
                    else:
                        signal = "Unknown"
                    
                    df_with_actions.iloc[action_info['step'], df_with_actions.columns.get_loc('Signal')] = signal
                    
                    # Aktualisiere die aktuelle Position für den nächsten Schritt
                    current_position = position
            
            # Speichere das erweiterte DataFrame als CSV
	    actions_csv_path = 'actions_csv_path'
            df_with_actions.to_csv(actions_csv_path, sep=';')
            print(f"\n✅ Aktionen als CSV gespeichert: {actions_csv_path}")
            
            # 2. Erstelle eine CSV mit detaillierten Trade-Informationen
            trades_data = []
            for trade in trades:
                if 'exit_step' in trade and trade['exit_step'] is not None:
                    
                    # Extrahiere die originalen Preisdaten aus dem DataFrame anstatt skalierte Werte zu verwenden
                    try:
                        # Verwende die Original-Preise aus dem Dataframe für Entry und Exit
                        entry_index = min(trade['entry_step'], len(original_df) - 1)
                        exit_index = min(trade['exit_step'], len(original_df) - 1)
                        
                        # Hole den Close-Preis für Entry und Exit
                        entry_price = float(original_df['Close'].iloc[entry_index])
                        
                        # Korrektur: Exit-Preis basierend auf dem Exit-Grund bestimmen
                        if 'exit_reason' in trade:
                            if trade['exit_reason'] == 'stop_loss':
                                # Bei Stop-Loss: Berechne den exakten Stop-Loss-Preis 
                                # anstatt den Close-Preis zu verwenden
                                stop_loss_value = abs(self.config['stop_loss'])  # Korrigiert: Verwende den konfigurierten Stop-Loss-Wert
                                exit_price = entry_price - stop_loss_value
                            elif trade['exit_reason'] == 'take_profit':
                                # Bei Take-Profit: Berechne den exakten Take-Profit-Preis
                                # anstatt den Close-Preis zu verwenden
                                take_profit_value = self.config['take_profit']  # Wert aus der Konfiguration
                                exit_price = entry_price + take_profit_value
                            else:
                                # Bei anderen Exit-Gründen (max_holding, etc.) verwende den Close-Preis
                                exit_price = float(original_df['Close'].iloc[exit_index])
                        else:
                            # Falls kein Exit-Grund angegeben, verwende den Close-Preis
                            exit_price = float(original_df['Close'].iloc[exit_index])
                        
                        # Berechne Profit/Loss mit den korrigierten Preisen
                        profit_loss = ((exit_price - entry_price) / entry_price) * 100
                        if trade['type'] == 'SHORT':
                            profit_loss = -profit_loss
                        
                    except Exception as e:
                        print(f"⚠️ Fehler bei der Preiskonvertierung: {e}")
                        # Fallback auf die skalierten Preise, falls die originalen nicht verfügbar sind
                        entry_price = trade['entry_price']
                        exit_price = trade['exit_price']
                        profit_loss = ((exit_price - entry_price) / entry_price) * 100
                        if trade['type'] == 'SHORT':
                            profit_loss = -profit_loss
                    
                    # Erstelle einen Eintrag für die CSV
                    trade_entry = {
                        'Type': trade['type'],
                        'Entry_Step': trade['entry_step'],
                        'Entry_Price': entry_price,  # Originaler Preis
                        'Exit_Step': trade['exit_step'],
                        'Exit_Price': exit_price,    # Korrigierter Preis basierend auf dem Exit-Grund
                        'Duration': trade['exit_step'] - trade['entry_step'],
                        'Profit_Loss_Pct': profit_loss,
                        'Exit_Reason': trade.get('exit_reason', 'Unknown')
                    }
                    
                    # Füge Zeitstempel hinzu, wenn im originalen DataFrame enthalten
                    if not original_df.index.equals(pd.RangeIndex(len(original_df))):
                        try:
                            entry_time = original_df.index[entry_index]
                            exit_time = original_df.index[exit_index]
                            
                            if isinstance(entry_time, pd.Timestamp):
                                trade_entry['Entry_Time'] = entry_time
                            if isinstance(exit_time, pd.Timestamp):
                                trade_entry['Exit_Time'] = exit_time
                        except Exception as e:
                            print(f"⚠️ Warnung: Konnte Zeitstempel nicht extrahieren: {e}")
                    elif 'DateTime' in original_df.columns:
                        try:
                            entry_time = original_df['DateTime'].iloc[entry_index]
                            exit_time = original_df['DateTime'].iloc[exit_index]
                            
                            if entry_time is not None:
                                trade_entry['Entry_Time'] = entry_time
                            if exit_time is not None:
                                trade_entry['Exit_Time'] = exit_time
                        except Exception as e:
                            print(f"⚠️ Warnung: Konnte Zeitstempel nicht extrahieren: {e}")
                    
                    trades_data.append(trade_entry)
            
            # Erstelle ein DataFrame aus den Trade-Daten
            if trades_data:
                trades_df = pd.DataFrame(trades_data)
                
                # Speichere das Trades-DataFrame als CSV
                trades_csv_path = 'trades_csv_path'
                trades_df.to_csv(trades_csv_path, sep=';', index=False)
                print(f"✅ Trades als CSV gespeichert: {trades_csv_path}")
                
                # Debug-Ausgabe der ersten Trade-Daten
                print("\n📊 Beispiel der gespeicherten Trade-Daten (erster Trade):")
                if len(trades_df) > 0:
                    first_trade = trades_df.iloc[0]
                    print(f"   • Typ: {first_trade['Type']}")
                    print(f"   • Entry: Schritt {first_trade['Entry_Step']}, Preis {first_trade['Entry_Price']:.2f}")
                    print(f"   • Exit: Schritt {first_trade['Exit_Step']}, Preis {first_trade['Exit_Price']:.2f}")
                    print(f"   • P&L: {first_trade['Profit_Loss_Pct']:.2f}%")
                    print(f"   • Exit-Grund: {first_trade['Exit_Reason']}")
        
        except Exception as e:
            print(f"❌ Fehler beim Exportieren der Validierungsergebnisse: {e}")
            import traceback
            traceback.print_exc()


# ================= 4. Training und Validierung =================

class TrainingManager:
    def __init__(self, agent, env, eval_env, config):
        self.agent = agent
        self.env = env
        self.eval_env = eval_env
        self.config = config
        
        self.episode_rewards = []
        self.eval_rewards = []
        self.losses = []
        self.best_eval_reward = float('-inf')
        # Zusätzliche Metriken für detaillierte Analyse
        self.win_rates = []          # Gewinnrate pro Episode
        self.avg_trade_durations = [] # Durchschnittliche Handelsdauer
        self.positions_distribution = [] # Verteilung der Positionstypen (long/short/neutral)
        self.avg_rewards_per_trade = [] # Durchschnittliche Belohnung pro Trade
        self.trade_counts = []       # Anzahl der Trades pro Episode
    
    def train(self, num_episodes):
        """Führt das Training über eine bestimmte Anzahl von Episoden durch"""
        # Prüfe, ob paralleles Training aktiviert ist
        if self.config.get('parallel_training', True) and num_episodes > 1:
            return self.train_parallel(num_episodes)
            
        print("\n🚀 Starte sequentielles Training...")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📋 Training für {num_episodes} Epochen")
        print("🔄 Fortschritt: [0%] ", end="", flush=True)
        
        # Halte externe Validierungsergebnisse fest
        self.external_val_results = []
        
        for episode in range(num_episodes):
            start_time = time.time()
            state = self.env.reset()
            episode_reward = 0
            episode_loss = 0
            step_count = 0
            trades_info = []
            
            # Fortschrittsanzeige für die Episode
            total_steps = len(self.env.data) - self.env.current_step
            progress_interval = max(1, total_steps // 10)
            current_progress = 0
            
            while True:
                # Fortschritt innerhalb der Episode anzeigen
                if step_count % progress_interval == 0:
                    progress_percent = min(100, int((step_count / total_steps) * 100))
                    if progress_percent > current_progress:
                        current_progress = progress_percent
                        print(f"\r⏳ Epoche {episode+1}/{num_episodes} - Fortschritt: [{progress_percent}%]", end="", flush=True)
                
                # Aktion auswählen und ausführen
                action = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                # Handelsinfos sammeln
                if 'action_taken' in info:
                    if info['action_taken'] == 'opened_position':
                        trades_info.append({
                            'type': 'LONG' if self.env.position == 1 else 'SHORT',
                            'entry_step': self.env.current_step - 1,
                            'entry_price': self.env.entry_price,
                            'exit_step': None,
                            'exit_price': None,
                            'reward': 0
                        })
                    elif info['action_taken'] == 'closed_position' and trades_info:
                        # Korrigierte Zeile: Überprüfe ob der letzte Trade existiert und keine exit_step hat
                        if len(trades_info) > 0 and 'exit_step' not in trades_info[-1] or trades_info[-1].get('exit_step') is None:
                            trades_info[-1]['exit_step'] = self.env.current_step - 1
                            trades_info[-1]['exit_price'] = self.env.data[self.env.current_step-1, -1, 3] if len(self.env.data.shape) == 3 else self.env.data[self.env.current_step-1, 3]
                            trades_info[-1]['reward'] = reward
                
                # In Replay-Speicher speichern
                self.agent.memory.push(state, action, reward, next_state, done)
                
                # Modell aktualisieren
                if len(self.agent.memory) > self.config['batch_size']:
                    loss = self.agent.update_model()
                    episode_loss += loss if loss is not None else 0
                    step_count += 1
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # Target-Netzwerk aktualisieren
            if episode % self.config['target_update'] == 0:
                self.agent.update_target_network()
                print(f"\n🔄 Target-Netzwerk aktualisiert in Episode {episode+1}")
            
            # Reguläre Evaluierung durchführen
            if episode % self.config['validation_interval'] == 0:
                eval_reward = self.evaluate()
                self.eval_rewards.append(eval_reward)
                
                # Bestes Modell speichern
                if eval_reward > self.best_eval_reward:
                    self.best_eval_reward = eval_reward
                    self.agent.save_model(self.config['model_save_path'])
                    print(f"🏆 Neues bestes Modell! Reward: {eval_reward:.4f}")
                    
                # Externe Validierung mit Test.csv durchführen
                print("\n━━━━━━━━━ EXTERNE VALIDIERUNG ━━━━━━━━━")
                ext_val_results = self.agent.evaluate_with_external_data()
                if ext_val_results is not None:
                    self.external_val_results.append({
                        'episode': episode,
                        'results': ext_val_results
                    })
                print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            
            # Statistiken aktualisieren
            self.episode_rewards.append(episode_reward)
            if step_count > 0:
                self.losses.append(episode_loss / step_count)
            else:
                self.losses.append(0)
                
            # Zusätzliche Metriken berechnen
            completed_trades = [t for t in trades_info if 'exit_step' in t and t['exit_step'] is not None]
            num_trades = len(completed_trades)
            
            # Gewinnrate
            win_rate = 0
            if num_trades > 0:
                num_wins = len([t for t in completed_trades if t['reward'] > 0])
                win_rate = num_wins / num_trades
            self.win_rates.append(win_rate)
            
            # Durchschnittliche Handelsdauer
            avg_duration = 0
            if num_trades > 0:
                durations = [t['exit_step'] - t['entry_step'] for t in completed_trades]
                avg_duration = sum(durations) / len(durations)
            self.avg_trade_durations.append(avg_duration)
            
            # Positionsverteilung (Long vs. Short)
            long_count = len([t for t in trades_info if t['type'] == 'LONG'])
            short_count = len([t for t in trades_info if t['type'] == 'SHORT'])
            self.positions_distribution.append((long_count, short_count))
            
            # Durchschnittliche Belohnung pro Trade
            avg_reward_per_trade = 0
            if num_trades > 0:
                trade_rewards = [t['reward'] for t in completed_trades]
                avg_reward_per_trade = sum(trade_rewards) / len(trade_rewards)
            self.avg_rewards_per_trade.append(avg_reward_per_trade)
            
            # Anzahl der Trades
            self.trade_counts.append(num_trades)
    
            # Gesamtfortschritt anzeigen
            elapsed_time = time.time() - start_time
            overall_progress = int(((episode + 1) / num_episodes) * 100)
            
            # Ausführliche Ausgabe am Ende jeder Episode
            print(f"\r━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"📊 Epoche {episode+1}/{num_episodes} abgeschlossen [{overall_progress}%]")
            print(f"⏱️  Zeit: {elapsed_time:.2f}s | 🧠 Memory: {len(self.agent.memory)}/{self.config['memory_capacity']}")
            print(f"🎯 Epsilon: {self.agent.epsilon:.4f} | 📉 Loss: {self.losses[-1]:.6f}")
            print(f"🏆 Episode-Reward: {episode_reward:.4f} | Eval-Reward: {self.eval_rewards[-1] if len(self.eval_rewards) > 0 else 'N/A':.4f}")
            
            # Handel-Statistiken
            print(f"📈 Trading-Statistiken:")
            print(f"   • Trades: {num_trades} | Win-Rate: {win_rate*100:.2f}% | Durchschn. Reward/Trade: {avg_reward_per_trade:.4f}")
            print(f"   • Durchschn. Haltedauer: {avg_duration:.2f} Steps | Long: {long_count} | Short: {short_count}")
            
            # Berechne und zeige Delta (Änderung) zum vorherigen Durchlauf
            if episode > 0:
                try:
                    prev_reward = self.episode_rewards[-2]
                    reward_delta = episode_reward - prev_reward
                    reward_delta_pct = (reward_delta /abs(prev_reward)) * 100 if abs(prev_reward) > 0.0001 else 0
                    
                    prev_win_rate = self.win_rates[-2] if len(self.win_rates) > 1 else 0
                    win_rate_delta = win_rate - prev_win_rate
                    win_rate_delta_pct = win_rate_delta * 100
                    
                    # Symbolbasierte Darstellung der Änderung für schnelles Erfassen
                    reward_arrow = "🔼" if reward_delta > 0 else "🔽" if reward_delta < 0 else "➡️"
                    win_rate_arrow = "🔼" if win_rate_delta > 0 else "🔽" if win_rate_delta < 0 else "➡️"
                    
                    print(f"\n📊 Änderungen zur vorherigen Episode:")
                    print(f"   • Reward: {reward_arrow} {reward_delta:.4f} ({reward_delta_pct:.2f}%)")
                    print(f"   • Win-Rate: {win_rate_arrow} {win_rate_delta_pct:.2f}%")
                except Exception as e:
                    print(f"Fehler bei Berechnung der Deltas: {e}")
            
            # Öffne Positionen ausgeben
            open_positions = [t for t in trades_info if 'exit_step' not in t or t['exit_step'] is None]
            if open_positions:
                print(f"\n📌 Offene Positionen am Ende der Episode: {len(open_positions)}")
                for i, pos in enumerate(open_positions):
                    current_step = self.env.current_step - 1
                    current_price = self.env.data[current_step, -1, 3] if len(self.env.data.shape) == 3 else self.env.data[current_step, 3]
                    duration = current_step - pos['entry_step']
                    pnl = (current_price - pos['entry_price']) / pos['entry_price']
                    if pos['type'] == 'SHORT':
                        pnl = -pnl
                    print(f"  {i+1}. {pos['type']} - Dauer: {duration} Steps, P&L: {pnl*100:.2f}%")
            
            # Lernkurven-Analyse - Neu hinzugefügt
            if episode >= 5:  # Ab 5 Episoden können wir einen Trend berechnen
                recent_rewards = self.episode_rewards[-5:]
                reward_trend = sum(recent_rewards) / len(recent_rewards)
                reward_slope = (recent_rewards[-1] - recent_rewards[0]) / len(recent_rewards)
                
                recent_win_rates = self.win_rates[-5:]
                win_rate_trend = sum(recent_win_rates) / len(recent_win_rates)
                win_rate_slope = (recent_win_rates[-1] - recent_win_rates[0]) / len(recent_win_rates)
                
                print(f"\n📊 Lernkurven-Analyse (letzte 5 Episoden):")
                print(f"   • Reward-Trend: {reward_trend:.4f} | Steigung: {'positiv 📈' if reward_slope > 0 else 'negativ 📉'} ({reward_slope:.4f})")
                print(f"   • Win-Rate-Trend: {win_rate_trend*100:.2f}% | Steigung: {'positiv 📈' if win_rate_slope > 0 else 'negativ 📉'} ({win_rate_slope*100:.2f}%)")
            
            # Abbruchkriterien prüfen (optional)
            if episode > 20 and episode_reward < -1.0:  # Beispiel: Abbrechen bei schlechter Leistung
                print("\n⚠️  Warnung: Schlechte Performance erkannt. Training wird fortgesetzt.")
            
            # Ausgabe von Sicherheitsinformationen
            if np.isnan(episode_reward) or np.isnan(self.losses[-1]):
                print("\n⚠️  Warnung: NaN-Werte erkannt! Überprüfen Sie die Daten und Parameter.")
        
        print(f"\r━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("✅ Training abgeschlossen")
        print(f"🏆 Bester Evaluierungs-Reward: {self.best_eval_reward:.4f}")
        
        # Lade das beste Modell
        self.agent.load_model(self.config['model_save_path'])
        
        return self.episode_rewards, self.eval_rewards, self.losses
        
    def train_parallel(self, num_episodes):
        """
        Führt das Training mit parallelen Episoden durch. Dies kann die Trainingszeit erheblich verkürzen, da mehrere Episoden gleichzeitig
        ausgeführt werden können.
        
        Args:
            num_episodes: Anzahl der zu trainierenden Episoden
            
        Returns:
            Tuple aus (episode_rewards, eval_rewards, losses)
        """
        print("Starte paralleles Training...")
        print("-" * 80)
        print(f"Training für {num_episodes} Episoden mit parallelen Episoden")
        
        # Halte externe Validierungsergebnisse fest
        self.external_val_results = []
        
        # Bestimme die Anzahl der Worker und parallelen Episoden
        workers = self.config.get('training_workers', max(1, min(4, mp.cpu_count() - 1)))
        parallel_episodes = min(self.config.get('parallel_episodes', 4), workers)
        
        print(f"Verwende {workers} Worker für {parallel_episodes} parallele Episoden")
        
        # Zähle Episode von 0 an
        episode = 0
        
        # Zähler für Target-Updates einführen
        target_update_counter = 0
        
        # Schleife bis alle Episoden abgeschlossen sind
        while episode < num_episodes:
            start_time = time.time()
            
            # Bestimme die Anzahl der noch zu trainierenden Episoden
            remaining_episodes = num_episodes - episode
            batch_size = min(parallel_episodes, remaining_episodes)
            
            # Bereite die Trainingsdaten vor (als CPU-Array, nicht als GPU-Tensor)
            if isinstance(self.env.data, torch.Tensor):
                train_data = self.env.data.cpu().numpy()
            else:
                train_data = self.env.data
            
            # Modellparameter für die Worker extrahieren (auf CPU)
            model_state_dict = {k: v.cpu() for k, v in self.agent.policy_network.state_dict().items()}
            
            # Definiere Argumentenliste für die parallelen Episoden
            shared_memory_size = self.config['memory_capacity'] // parallel_episodes
            args_list = [(train_data, self.config, model_state_dict, shared_memory_size) for _ in range(batch_size)]
            
            episode_results = []
            
            # Führe die Episoden parallel aus
            print(f"\nStarte Batch mit {batch_size} parallelen Episoden...")
            try:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    episode_results = list(executor.map(run_training_episode, args_list))
                    
                print(f"Parallele Episoden abgeschlossen in {time.time() - start_time:.2f} Sekunden")
                
            except Exception as e:
                print(f"Fehler bei parallelen Episoden: {e}")
                print("Verwende sequentielles Training als Fallback")
                # Erstelle einen sequentiellen Fallback, um die fehlenden Daten für Episoden bereitzustellen
                episode_results = []
                for _ in range(batch_size):
                    state = self.env.reset()
                    episode_reward = 0
                    step_count = 0
                    experiences = []
                    trades_info = []
                    
                    while not self.env.done:
                        action = self.agent.select_action(state)
                        next_state, reward, done, info = self.env.step(action)
                        
                        experiences.append((state, action, reward, next_state, done))
                        
                        if 'action_taken' in info:
                            if info['action_taken'] == 'opened_position':
                                trades_info.append({
                                    'type': 'LONG' if self.env.position == 1 else 'SHORT',
                                    'entry_step': self.env.current_step - 1,
                                    'entry_price': self.env.entry_price,
                                    'exit_step': None,
                                    'exit_price': None,
                                    'reward': 0
                                })
                            elif info['action_taken'] == 'closed_position' and len(trades_info) > 0 and trades_info[-1]['exit_step'] is None:
                                trades_info[-1]['exit_step'] = self.env.current_step - 1
                                trades_info[-1]['exit_price'] = self.env.data[self.env.current_step-1, -1, 3] if len(self.env.data.shape) == 3 else self.env.data[self.env.current_step-1, 3]
                                trades_info[-1]['reward'] = reward
                        
                        state = next_state
                        episode_reward += reward
                        step_count += 1
                    
                    completed_trades = [t for t in trades_info if 'exit_step' in t and t['exit_step'] is not None]
                    num_trades = len(completed_trades)
                    
                    # Gewinnrate
                    win_rate = 0
                    if num_trades > 0:
                        num_wins = len([t for t in completed_trades if t['reward'] > 0])
                        win_rate = num_wins / num_trades
                    self.win_rates.append(win_rate)
                    
                    # Durchschnittliche Handelsdauer
                    avg_duration = 0
                    if num_trades > 0:
                        durations = [t['exit_step'] - t['entry_step'] for t in completed_trades]
                        avg_duration = sum(durations) / len(durations)
                    self.avg_trade_durations.append(avg_duration)
                    
                    # Positionsverteilung (Long vs. Short)
                    long_count = len([t for t in trades_info if t['type'] == 'LONG'])
                    short_count = len([t for t in trades_info if t['type'] == 'SHORT'])
                    self.positions_distribution.append((long_count, short_count))
                    
                    # Durchschnittliche Belohnung pro Trade
                    avg_reward_per_trade = 0
                    if num_trades > 0:
                        trade_rewards = [t['reward'] for t in completed_trades]
                        avg_reward_per_trade = sum(trade_rewards) / len(trade_rewards)
                    self.avg_rewards_per_trade.append(avg_reward_per_trade)
                    
                    episode_results.append({
                        'experiences': experiences,
                        'episode_reward': episode_reward,
                        'step_count': step_count,
                        'trades_info': trades_info,
                        'num_trades': num_trades,
                        'win_rate': win_rate,
                        'avg_duration': avg_duration,
                        'positions': (long_count, short_count),
                        'avg_reward_per_trade': avg_reward_per_trade,
                    })
            
            # Verarbeite die Ergebnisse der parallelen Episoden
            total_episode_loss = 0
            
            # Übertrage Erfahrungen in den gemeinsamen Replay-Speicher
            for result in episode_results:
                # Füge Erfahrungen zum zentralen Replay-Speicher hinzu
                for exp in result['experiences']:
                    state, action, reward, next_state, done = exp
                    self.agent.memory.push(state, action, reward, next_state, done)
                
                # Aktualisiere Statistiken
                self.episode_rewards.append(result['episode_reward'])
                self.trade_counts.append(result['num_trades'])
                self.win_rates.append(result['win_rate'])
                self.avg_trade_durations.append(result['avg_duration'])
                self.positions_distribution.append(result['positions'])
                self.avg_rewards_per_trade.append(result['avg_reward_per_trade'])
            
            # Trainiere das Modell mit dem gemeinsamen Replay-Speicher
            if len(self.agent.memory) > self.config['batch_size']:
                num_updates = min(100, len(self.agent.memory) // self.config['batch_size'])
                batch_losses = []
                
                for _ in range(num_updates):
                    loss = self.agent.update_model()
                    if loss is not None:
                        batch_losses.append(loss)
                
                avg_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0
                for _ in range(batch_size):
                    self.losses.append(avg_loss)
                
                total_episode_loss = avg_loss * batch_size
            else:
                # Füge Nullverluste hinzu, wenn nicht genügend Daten für ein Training vorhanden sind
                for _ in range(batch_size):
                    self.losses.append(0)
            
            # Aktualisiere den Target-Netzwerk-Zähler für jede Episode im Batch
            target_update_counter += batch_size
            
            # Aktualisiere das Target-Netzwerk basierend auf dem Zähler, nicht auf der Episodennummer
            # Dies stellt sicher, dass Updates genau nach der definierten Anzahl von Episoden erfolgen
            if target_update_counter >= self.config['target_update']:
                self.agent.update_target_network()
                print(f"\n🔄 Target-Netzwerk aktualisiert nach {target_update_counter} Episoden")
                # Reduziere den Zähler um die target_update-Frequenz, behält aber den Rest für den nächsten Zyklus
                target_update_counter %= self.config['target_update']
            
            # Führe Evaluation immer nach jedem parallelen Batch durch
            # Validierung bei jedem Batch durchführen, unabhängig vom validation_interval
            eval_reward = self.evaluate()
            for _ in range(batch_size):
                self.eval_rewards.append(eval_reward)
            
            # Bestes Modell speichern
            if eval_reward > self.best_eval_reward:
                self.best_eval_reward = eval_reward
                self.agent.save_model(self.config['model_save_path'])
                print(f"🏆 Neues bestes Modell! Reward: {eval_reward:.4f}")
            
            # Externe Validierung mit Test.csv
            print("\n━━━━━━━━━ EXTERNE VALIDIERUNG ━━━━━━━━━")
            ext_val_results = self.agent.evaluate_with_external_data()
            if ext_val_results is not None:
                self.external_val_results.append({
                    'episode': episode + batch_size - 1,
                    'results': ext_val_results
                })
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            
            # Ausgabe der Episoden-Zusammenfassung
            elapsed_time = time.time() - start_time
            overall_progress = int(((episode + batch_size) / num_episodes) * 100)
            
            # Zusammenfassung der Batch-Ergebnisse
            batch_rewards = [result['episode_reward'] for result in episode_results]
            avg_batch_reward = sum(batch_rewards) / len(batch_rewards)
            
            batch_trades = [result['num_trades'] for result in episode_results]
            avg_batch_trades = sum(batch_trades) / len(batch_trades)
            
            batch_win_rates = [result['win_rate'] for result in episode_results]
            avg_batch_win_rate = sum(batch_win_rates) / len(batch_win_rates)
            
            # Ausführliche Ausgabe am Ende jedes Batches
            print(f"\r━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"Parallel-Training: Epochen {episode+1} bis {episode+batch_size}/{num_episodes} abgeschlossen [{overall_progress}%]")
            print(f"⏱️  Zeit: {elapsed_time:.2f}s | 🧠 Memory: {len(self.agent.memory)}/{self.config['memory_capacity']}")
            print(f"🎯 Epsilon: {self.agent.epsilon:.4f} | 📉 Durchschn. Loss: {total_episode_loss / batch_size:.6f}")
            
            # Korrigierte Zeile mit Fehlerbehandlung für den Eval-Reward
            eval_reward_str = "N/A"
            if self.eval_rewards:
                eval_reward_str = f"{self.eval_rewards[-1]:.4f}"
            print(f"🏆 Durchschn. Reward: {avg_batch_reward:.4f} | Eval-Reward: {eval_reward_str}")
            
            # Aktualisiere den Episodenzähler
            episode += batch_size
            
            # Epsilon-Decay beschleunigen
            # Bei parallelem Training müssen wir Epsilon stärker reduzieren,
            # da weniger Updates des Hauptmodells durchgeführt werden
            epsilon_reduce_factor = 0.9
            self.agent.epsilon *= epsilon_reduce_factor
            self.agent.epsilon = max(self.config['epsilon_end'], self.agent.epsilon)
        
        print(f"\r━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("✅ Paralleles Training abgeschlossen")
        print(f"🏆 Bester Evaluierungs-Reward: {self.best_eval_reward:.4f}")
        
        # Lade das beste Modell
        self.agent.load_model(self.config['model_save_path'])
        
        return self.episode_rewards, self.eval_rewards, self.losses

    def evaluate(self):
        """
        Führt eine Evaluierung des Agenten auf den Validierungsdaten durch.
        
        Returns:
            float: Die durchschnittliche Belohnung pro Episode.
        """
        print("\n🔍 Führe Evaluierung auf Validierungsdaten durch...")
        
        # Setze die Umgebung zurück
        state = self.eval_env.reset()
        total_reward= 0
        num_steps = 0
        trades = []
        
        # Führe die Evaluierung ohne Exploration durch
        while not self.eval_env.done:
            action = self.agent.select_action(state, eval_mode=True)
            next_state, reward, done, info = self.eval_env.step(action)
            
            # Sammle Handelsinformationen
            if 'action_taken' in info:
                if info['action_taken'] == 'opened_position':
                    trades.append({
                        'type': 'LONG' if self.eval_env.position == 1 else 'SHORT',
                        'entry_step': self.eval_env.current_step - 1,
                        'entry_price': self.eval_env.entry_price,
                        'exit_step': None,
                        'exit_price': None,
                        'reward': 0
                    })
                elif info['action_taken'] == 'closed_position':
                    # Überprüfe ob der letzte Trade existiert und ob der Schlüssel 'exit_step' existiert und None ist
                    if len(trades) > 0 and ('exit_step' not in trades[-1] or trades[-1]['exit_step'] is None):
                        trades[-1]['exit_step'] = self.eval_env.current_step - 1
                        trades[-1]['exit_price'] = self.eval_env.data[self.eval_env.current_step-1, -1, 3] if len(self.eval_env.data.shape) == 3 else self.eval_env.data[self.eval_env.current_step-1, 3]
            
            # Zustand aktualisieren
            state = next_state
            total_reward += reward
        
        # Berechne Statistiken
        avg_reward = total_reward
        # Sicherere Methode: Filtere Trades, bei denen 'exit_step' vorhanden und nicht None ist
        completed_trades = [t for t in trades if 'exit_step' in t and t['exit_step'] is not None]
        num_trades = len(completed_trades)
        
        win_rate = 0
        avg_profit = 0
        avg_loss = 0
        
        if num_trades > 0:
            winning_trades = [t for t in completed_trades if t.get('reward', 0) > 0]
            losing_trades = [t for t in completed_trades if t.get('reward', 0) <= 0]
            win_rate = len(winning_trades) / num_trades
            
            if winning_trades:
                avg_profit = sum(t.get('reward', 0) for t in winning_trades) / len(winning_trades)
            if losing_trades:
                avg_loss = sum(t.get('reward', 0) for t in losing_trades) / len(losing_trades)
        
        final_balance = self.eval_env.balance
        
        print(f"✅ Evaluierung abgeschlossen:")
        print(f"   • Reward: {avg_reward:.4f}")
        print(f"   • Endkapital: ${final_balance:.2f} (Änderung: {((final_balance/self.config['initial_balance'])-1)*100:.2f}%)")
        print(f"   • Anzahl Trades: {num_trades}")
        
        if num_trades > 0:
            print(f"   • Gewinner: {len(winning_trades)} ({len(winning_trades) / num_trades * 100:.2f}%)")
            print(f"   • Verlierer: {len(losing_trades)} ({len(losing_trades) / num_trades * 100:.2f}%)")
            print(f"   • Durchschn. Gewinn: {avg_profit:.4f}")
            print(f"   • Durchschn. Verlust: {avg_loss:.4f}")
        
        return avg_reward


# ================= 5. Backtesting und Signalgenerierung =================

class BacktestManager:
    def __init__(self, agent, data, scaler, config):
        self.agent = agent
        self.data = data
        self.scaler = scaler
        self.config = config
    
    def backtest(self):
        """Führt einen Backtest des trainierten Agenten mit GPU-Beschleunigung durch"""
        print("Starte Backtest mit GPU-Beschleunigung...")
        
        # Initialisiere die Umgebung für den Backtest mit GPU-Unterstützung
        env = TradingEnvironment(self.data, self.config, use_gpu=True)
        state = env.reset()
        
        trades = []
        actions = []
        equity_curve = [env.balance]
        positions = [0]
        
        # Batch-Verarbeitung für forward pass (wenn möglich)
        batch_actions = []
        batch_states = []
        batch_size = 32  # Eine kleine Batch-Größe für effizientere Vorhersagen
        
        while not env.done:
            # Sammle Zustände für Batch-Verarbeitung
            batch_states.append(state)
            
            if len(batch_states) >= batch_size or env.current_step >= len(env.data) - 2:
                # Batch-Verarbeitung durchführen
                if len(batch_states) > 0:
                    with torch.no_grad():
                        for batch_state in batch_states:
                            # Aktion mit GPU-Beschleunigung auswählen
                            action = self.agent.select_action(batch_state, eval_mode=True)
                            batch_actions.append(action)
                    
                    # Batch zurücksetzen
                    batch_states = []
            
            # Wenn wir Aktionen haben, führe die nächste aus
            if batch_actions:
                action = batch_actions.pop(0)
                actions.append(action)
                
                # Aktion ausführen
                next_state, reward, done, info = env.step(action)
                
                # Handelsinfo speichern, wenn Position eröffnet oder geschlossen wurde
                if 'action_taken' in info:
                    # Verwende GPU-Tensor, wenn verfügbar
                    if env.use_gpu and env.data_tensor is not None:
                        price_idx = min(env.current_step - 1, len(env.data_tensor) - 1)
                        if len(env.data_tensor.shape) == 3:
                            price = env.data_tensor[price_idx, -1, 3].item()
                        else:
                            price = env.data_tensor[price_idx, 3].item()
                    else:
                        price_idx = min(env.current_step - 1, len(env.data) - 1) 
                        price = self.data[price_idx, 3] if len(self.data.shape) == 2 else self.data[price_idx, -1, 3]
                        
                    if info['action_taken'] == 'opened_position':
                        trades.append({
                            'type': 'LONG' if env.position == 1 else 'SHORT',
                            'entry_step': env.current_step - 1,
                            'entry_price': price,
                            'exit_step': None,
                            'exit_price': None,
                            'reward': 0
                        })
                    
                    elif info['action_taken'] == 'closed_position':
                        if len(trades) > 0 and trades[-1]['exit_step'] is None:
                            trades[-1]['exit_step'] = env.current_step - 1
                            trades[-1]['exit_price'] = price
            
            # Zustand aktualisieren
            state = next_state
            equity_curve.append(env.balance)
            positions.append(env.position)
        
        # Performance-Metriken mit Torch berechnen für Geschwindigkeit
        returns = torch.tensor(equity_curve) / self.config['initial_balance'] - 1
        mean_return = torch.mean(returns).item()
        std_return = torch.std(returns).item() + 1e-9  # Verhindere Division durch Null
        
        # Sharpe-Ratio berechnen
        sharpe = mean_return / std_return * np.sqrt(252)  # Annahme: Tägliche Daten
        
        max_equity = torch.tensor(equity_curve).max().item()
        drawdowns = torch.maximum(torch.tensor(0.0), torch.tensor([max(equity_curve[:i+1]) for i in range(len(equity_curve))]) - torch.tensor(equity_curve))
        max_drawdown = torch.max(drawdowns).item() / max_equity
        
        # Zusammenfassung
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['profit'] is not None and t['profit'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        print("Backtest abgeschlossen")
        print(f"Endkapital: ${equity_curve[-1]:.2f} (Rendite: {(equity_curve[-1]/self.config['initial_balance']-1)*100:.2f}%)")
        print(f"Sharpe-Ratio: {sharpe:.2f}")
        print(f"Maximaler Drawdown: {max_drawdown*100:.2f}%")
        print(f"Anzahl Trades: {total_trades}, Gewinner: {winning_trades}, Win-Rate: {win_rate*100:.2f}%")
        
        return {
            'equity_curve': equity_curve,
            'positions': positions,
            'actions': actions,
            'trades': trades,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }
    
    def generate_signals(self):
        """Generiert aktuelle Handelssignale basierend auf dem trainierten Modell"""
        # Implementiere die Signal-Generierung für Live-Trading
        pass
    
    def plot_backtest_results(self, results):
        """Visualisiert die Backtesting-Ergebnisse"""
        plt.figure(figsize=(15, 12))
        
        # Plot Performance
        plt.subplot(2, 1, 1)
        plt.plot(results['equity_curve'])
        plt.title('Equity Curve')
        plt.xlabel('Zeitschritt')
        plt.ylabel('Kapital')
        
        # Plot Positionen
        plt.subplot(2, 1, 2)
        plt.plot(results['positions'])
        plt.title('Positionen (1: Long, -1: Short, 0: Neutral)')
        plt.xlabel('Zeitschritt')
        plt.ylabel('Position')
        plt.yticks([-1, 0, 1])
        
        plt.tight_layout()
        plt.show()
        
        # Plot der Trades
        plt.figure(figsize=(15, 8))
        
        # Lade die originalen (nicht-skalierten) Preisdaten
        prices = self.scaler.inverse_transform(self.data)[:, 3]  # Close-Preis
        
        plt.plot(prices)
        plt.title('Trades auf Preischart')
        plt.xlabel('Zeitschritt')
        plt.ylabel('Preis')
        
        # Markiere Long-Einstiege
        long_entries = [t['entry_time'] for t in results['trades'] if t['type'] == 'LONG']
        long_entry_prices = [prices[t] for t in long_entries]
        plt.scatter(long_entries, long_entry_prices, marker='^', color='green', s=100, label='Long Entry')
        
        # Markiere Long-Ausstiege
        long_exits = [t['exit_time'] for t in results['trades'] if t['type'] == 'LONG' and t['exit_time'] is not None]
        long_exit_prices = [prices[t] for t in long_exits]
        plt.scatter(long_exits, long_exit_prices, marker='v', color='blue', s=100, label='Long Exit')
        
        # Markiere Short-Einstiege
        short_entries = [t['entry_time'] for t in results['trades'] if t['type'] == 'SHORT']
        short_entry_prices = [prices[t] for t in short_entries]
        plt.scatter(short_entries, short_entry_prices, marker='v', color='red', s=100, label='Short Entry')
        
        # Markiere Short-Ausstiege
        short_exits = [t['exit_time'] for t in results['trades'] if t['type'] == 'SHORT' and t['exit_time'] is not None]
        short_exit_prices =[prices[t] for t in short_exits]
        plt.scatter(short_exits, short_exit_prices, marker='^', color='orange', s=100, label='Short Exit')
        
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Plot zusätzliche Metriken
        plt.figure(figsize=(15, 15))
        
        # Win-Rate
        plt.subplot(3, 2, 1)
        plt.plot(self.win_rates)
        plt.title('Win-Rate pro Episode')
        plt.xlabel('Episode')
        plt.ylabel('Win-Rate')
        
        # Durchschnittliche Handelsdauer
        plt.subplot(3, 2, 2)
        plt.plot(self.avg_trade_durations)
        plt.title('Durchschnittliche Handelsdauer')
        plt.xlabel('Episode')
        plt.ylabel('Dauer (Zeitschritte)')
        
        # Anzahl der Trades
        plt.subplot(3, 2, 3)
        plt.plot(self.trade_counts)
        plt.title('Anzahl der Trades pro Episode')
        plt.xlabel('Episode')
        plt.ylabel('Anzahl Trades')
        
        # Durchschnittliche Belohnung pro Trade
        plt.subplot(3, 2, 4)
        plt.plot(self.avg_rewards_per_trade)
        plt.title('Durchschnittliche Belohnung pro Trade')
        plt.xlabel('Episode')
        plt.ylabel('Belohnung')
        
        # Positionsverteilung (Long vs. Short)
        plt.subplot(3, 2, 5)
        if self.positions_distribution:  # Überprüfe ob Daten vorhanden sind
            long_counts = [pos[0] for pos in self.positions_distribution]
            short_counts = [pos[1] for pos in self.positions_distribution]
            
            x = range(len(self.positions_distribution))
            plt.bar(x, long_counts, width=0.4, label='Long', align='edge', color='green')
            plt.bar(x, short_counts, width=-0.4, label='Short', align='edge', color='red')
            plt.title('Position Distribution (Long vs. Short)')
            plt.xlabel('Episode')
            plt.ylabel('Count')
            plt.legend()
        
        # Trade Counts
        plt.subplot(3, 2, 6)
        plt.plot(self.trade_counts)
        plt.title('Anzahl der Trades')
        plt.xlabel('Episode')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.show()
        
        # Weitere Metriken in separatem Plot
        if self.avg_rewards_per_trade or self.avg_trade_durations:
            plt.figure(figsize=(12, 5))
            
            # Durchschnittliche Belohnung pro Trade
            plt.subplot(1, 2, 1)
            plt.plot(self.avg_rewards_per_trade)
            plt.title('Durchschnittliche Belohnung pro Trade')
            plt.xlabel('Episode')
            plt.ylabel('Belohnung')
            
            # Durchschnittliche Handelsdauer
            plt.subplot(1, 2, 2)
            plt.plot(self.avg_trade_durations)
            plt.title('Durchschnittliche Handelsdauer')
            plt.xlabel('Episode')
            plt.ylabel('Dauer (Zeitschritte)')
            
            plt.tight_layout()
            plt.show()
        
        # Externe Validierungsergebnisse, falls vorhanden
        if hasattr(self, 'external_val_results') and self.external_val_results:
            plt.figure(figsize=(15, 10))
            
            # Extrahiere Daten
            episodes = [result['episode'] for result in self.external_val_results]
            rewards = [result['results']['reward'] for result in self.external_val_results]
            balances = [result['results']['balance'] for result in self.external_val_results]
            win_rates = [result['results']['win_rate'] for result in self.external_val_results]
            
            # Plot Reward
            plt.subplot(2, 2, 1)
            plt.plot(episodes, rewards)
            plt.title('Externe Validierung - Reward')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            
            # Plot Balance
            plt.subplot(2, 2, 2)
            plt.plot(episodes, balances)
            plt.title('Externe Validierung - Endkapital')
            plt.xlabel('Episode')
            plt.ylabel('Kapital')
            
            # Plot Win Rate
            plt.subplot(2, 2, 3)
            plt.plot(episodes, win_rates)
            plt.title('Externe Validierung - Win Rate')
            plt.xlabel('Episode')
            plt.ylabel('Win Rate (%)')
            
            plt.tight_layout()
            plt.show()
        


# ================= Hilfsfunktion für das parallele Training =================

def run_training_episode(args):
    """
    Führt eine einzelne Trainingsepisode aus. Diese Funktion wird in parallelen Prozessen aufgerufen.
    
    Args:
        args: Tupel aus (data, config, model_state_dict, shared_memory_size)
            - data: Die Trainingsdaten als NumPy-Array
            - config: Die Konfigurationsparameter
            - model_state_dict: Dictionary mit den Gewichten des Modells
            - shared_memory_size: Maximale Größe des gemeinsamen Replay-Speichers
    
    Returns:
        Ein Dictionary mit den Ergebnissen der Episode (Belohnungen, Verluste, Trades etc.)
    """
    # Entpacke Argumente
    data, config, model_state_dict, shared_memory_size = args
    
    # Erstelle lokale Umgebung für diesen Prozess (ohne GPU für Parallelisierung)
    env = TradingEnvironment(data, config, use_gpu=False)
    
    # Erstelle lokalen Agenten für diesen Prozess
    agent = LSTMRLAgent(
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        hidden_dim=config['hidden_dim'],
        config=config
    )
    
    # Lade Modellgewichte
    agent.policy_network.load_state_dict(model_state_dict)
    agent.target_network.load_state_dict(model_state_dict)
    
    # Lokaler Replay-Speicher für diese Episode
    agent.memory = ReplayBuffer(shared_memory_size, gpu_accelerated=False)
    
    # Initialisiere Zähler und Sammlungen
    state = env.reset()
    episode_reward = 0
    episode_loss = 0
    step_count = 0
    experiences = []  # Liste von (state, action, reward, next_state, done) Tupeln
    trades_info = []
    
    # Durchlaufe die Episode
    while not env.done:
        # Aktion auswählen mit Exploration
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        
        # Speichere Erfahrungen für späteres Update
        experiences.append((state, action, reward, next_state, done))
        
        # Handelsinfos sammeln für Statistiken
        if 'action_taken' in info:
            if info['action_taken'] == 'opened_position':
                trades_info.append({
                    'type': 'LONG' if env.position == 1 else 'SHORT',
                    'entry_step': env.current_step - 1,
                    'entry_price': env.entry_price,
                    'exit_step': None,
                    'exit_price': None,
                    'reward': 0
                })
            elif info['action_taken'] == 'closed_position' and len(trades_info) > 0 and trades_info[-1]['exit_step'] is None:
                trades_info[-1]['exit_step'] = env.current_step - 1
                trades_info[-1]['exit_price'] = env.data[env.current_step-1, -1, 3] if len(env.data.shape) == 3 else env.data[env.current_step-1, 3]
                trades_info[-1]['reward'] = reward
        
        # Zustand aktualisieren
        state = next_state
        episode_reward += reward
        step_count += 1
    
    # Berechne zusätzliche Statistiken
    completed_trades = [t for t in trades_info if 'exit_step' in t and t['exit_step'] is not None]
    num_trades = len(completed_trades)
    
    # Gewinnrate
    win_rate = 0
    if num_trades > 0:
        num_wins = len([t for t in completed_trades if t['reward'] > 0])
        win_rate = num_wins / num_trades
    
    # Durchschnittliche Handelsdauer
    avg_duration = 0
    if num_trades > 0:
        durations = [t['exit_step'] - t['entry_step'] for t in completed_trades]
        avg_duration = sum(durations) / len(durations)
    
    # Positionsverteilung (Long vs. Short)
    long_count = len([t for t in trades_info if t['type'] == 'LONG'])
    short_count = len([t for t in trades_info if t['type'] == 'SHORT'])
    
    # Durchschnittliche Belohnung pro Trade
    avg_reward_per_trade = 0
    if num_trades > 0:
        trade_rewards = [t['reward'] for t in completed_trades]
        avg_reward_per_trade = sum(trade_rewards) / len(trade_rewards)
    
    # Rückgabe der Ergebnisse als Dictionary
    results = {
        'experiences': experiences,  # Für den Replay-Speicher
        'episode_reward': episode_reward,
        'step_count': step_count,
        'trades_info': trades_info,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_duration': avg_duration,
        'positions': (long_count, short_count),
        'avg_reward_per_trade': avg_reward_per_trade,
    }
    
    return results


# ================= Main-Funktion =================

def main():
    print("LSTM-RL Trading System mit Single-Policy Architektur")
    print("-" * 50)
    print(f"Gerät: {device}")
    
    # Daten laden und vorverarbeiten
    data_processor = DataProcessor(CONFIG)
    df = data_processor.load_data()
    
    # Verwende die parallele Datenvorverarbeitung, wenn aktiviert
    if CONFIG.get('use_multiprocessing', True):
        print(f"Verwende parallele Datenvorverarbeitung mit {CONFIG['num_workers']} Prozessen...")
        X_train, X_val, X_test, y_train, y_val, y_test, X_train_tensor, X_val_tensor, X_test_tensor = data_processor.preprocess_data_parallel(df)
    else:
        print("Verwende Standard-Datenvorverarbeitung (ohne Multiprocessing)...")
        X_train, X_val, X_test, y_train, y_val, y_test, X_train_tensor, X_val_tensor, X_test_tensor = data_processor.preprocess_data(df)
    
    # Umgebungen erstellen - Wir verwenden für TradingEnvironment immer noch NumPy-Arrays
    # weil die Schnittstelle der Umgebung auf NumPy basiert
    train_env = TradingEnvironment(X_train, CONFIG)
    val_env = TradingEnvironment(X_val, CONFIG)
    test_env = TradingEnvironment(X_test, CONFIG)
    
    # GPU-beschleunigten Replay-Puffer aktivieren
    memory = ReplayBuffer(CONFIG['memory_capacity'], gpu_accelerated=True)
    
    # Agenten erstellen mit GPU-beschleunigtem Replay-Puffer
    agent = LSTMRLAgent(
        state_dim=CONFIG['state_dim'],
        action_dim=CONFIG['action_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        config=CONFIG
    )
    
    # Setze den optimierten Replay-Puffer
    agent.memory = memory
    
    # Modell laden, falls vorhanden
    model_loaded = agent.load_model(CONFIG['model_save_path'])
    
    if not model_loaded or input("Modell gefunden. Neu trainieren? (j/n): ").lower() == 'j':
        print("Beginne Training mit GPU-Beschleunigung...")
        
        # Training durchführen
        trainer = TrainingManager(agent, train_env, val_env, CONFIG)
        episode_rewards, eval_rewards, losses = trainer.train(CONFIG['num_episodes'])
        
        # Trainingsergebnisse visualisieren
        trainer.plot_results()
    
    # Backtesting
    backtester = BacktestManager(agent, X_test, data_processor.scaler, CONFIG)
    backtest_results = backtester.backtest()
    
    # Ergebnisse visualisieren
    backtester.plot_backtest_results(backtest_results)


if __name__ == "__main__":
    main()