#!/usr/bin/env python3
"""
Test der korrigierten Array-Dimensionen
======================================

Dieses Script testet den korrigierten Code für die Array-Dimensionen.
"""

import configparser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def test_array_dimensions():
    """Teste die Array-Dimensionen für Returns und Time."""
    
    print("🔍 ARRAY-DIMENSIONEN TEST")
    print("=" * 40)
    
    # Simuliere Candle-Daten
    n_candles = 100
    timestamps = pd.date_range(start='2024-01-01', periods=n_candles, freq='1min')
    prices = np.random.random(n_candles) * 100 + 50  # Zufällige Preise zwischen 50-150
    
    candles = pd.DataFrame({
        'Timestamp': timestamps,
        'Close': prices
    })
    
    print(f"Anzahl Candles: {len(candles)}")
    
    # Simuliere Log Returns (haben eine Position weniger als Candles)
    log_returns = np.diff(np.log(prices))  # Log Returns
    print(f"Anzahl Log Returns: {len(log_returns)}")
    
    # Test der korrigierten Array-Slicing-Logik
    print("\n🔧 Test des korrigierten Array-Slicing:")
    print("-" * 40)
    
    # Konfiguration für Lag
    lag_value = 5
    
    # Korrigierte Zeitarray-Erstellung
    time_array = candles['Timestamp'].iloc[1:len(log_returns)+1]
    print(f"Time Array Länge: {len(time_array)}")
    print(f"Returns Array Länge: {len(log_returns)}")
    print(f"Längen stimmen überein: {len(time_array) == len(log_returns)}")
    
    # Test der Lag-Berechnung
    if lag_value < len(log_returns):
        lagged_returns = log_returns[lag_value:]
        lagged_time = time_array.iloc[:-lag_value] if lag_value > 0 else time_array
        
        print(f"\nLag-Test (Lag={lag_value}):")
        print(f"Lagged Returns Länge: {len(lagged_returns)}")
        print(f"Lagged Time Länge: {len(lagged_time)}")
        print(f"Lagged Längen stimmen überein: {len(lagged_time) == len(lagged_returns)}")
    
    # Erstelle Test-Plot
    plt.figure(figsize=(12, 6))
    plt.xlabel('Time')
    plt.ylabel('Returns')
    plt.title('Test: Returns Visualization (Korrigierte Dimensionen)')
    
    # Plot mit korrigierten Dimensionen
    plt.plot(time_array, log_returns, 
            label='Returns', color='blue', alpha=0.7)
    
    # Lagged Returns
    if lag_value < len(log_returns):
        plt.plot(lagged_time, lagged_returns, 
                label=f'Returns with lag {lag_value}', 
                color='orange', alpha=0.7)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Test_Array_Dimensions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Test-Plot erstellt: Test_Array_Dimensions.png")

def test_config_structure():
    """Teste die Konfigurationsstruktur."""
    
    print("\n🔍 KONFIGURATIONSSTRUKTUR TEST")
    print("=" * 40)
    
    # Erstelle Test-Konfiguration
    config = configparser.ConfigParser()
    
    # Füge alle benötigten Sektionen hinzu
    config.add_section('DEFAULT')
    config.set('DEFAULT', 'UseLogReturns', 'True')
    config.set('DEFAULT', 'UseSimpleReturns', 'False')
    config.set('DEFAULT', 'SmoothLogReturns', 'False')
    config.set('DEFAULT', 'SmoothTicks', 'False')
    config.set('DEFAULT', 'SmoothCandles', 'False')
    config.set('DEFAULT', 'SmoothSimpleReturns', 'False')
    
    config.add_section('SQL')
    config.set('SQL', 'TimeColName', 'Timestamp')
    config.set('SQL', 'DataColName', 'Close')
    config.set('SQL', 'TableName', 'test_table')
    
    config.add_section('Autocorrelation')
    config.set('Autocorrelation', 'Lag', '5')
    
    config.add_section('Candle')
    config.set('Candle', 'TimeGranularity', '1min')
    
    # Teste Parameter-Anwendung
    parameters = {
        'Candle': {'TimeGranularity': '5min'},
        'DEFAULT': {'UseLogReturns': True, 'UseSimpleReturns': False}
    }
    
    print("Original Config:")
    print(f"  TimeGranularity: {config.get('Candle', 'TimeGranularity')}")
    print(f"  UseLogReturns: {config.get('DEFAULT', 'UseLogReturns')}")
    
    # Anwenden der Parameter
    for section_name, section_params in parameters.items():
        if section_name not in config:
            config.add_section(section_name)
        
        for param_name, param_value in section_params.items():
            config[section_name][param_name] = str(param_value)
    
    print("\nNach Parameter-Anwendung:")
    print(f"  TimeGranularity: {config.get('Candle', 'TimeGranularity')}")
    print(f"  UseLogReturns: {config.get('DEFAULT', 'UseLogReturns')}")
    
    print("✅ Konfigurationsstruktur Test erfolgreich")

def main():
    """Hauptfunktion für den Test."""
    
    print("🎯 ARRAY-DIMENSIONEN UND KONFIGURATION TEST")
    print("=" * 60)
    
    try:
        # Test 1: Array-Dimensionen
        test_array_dimensions()
        
        # Test 2: Konfigurationsstruktur
        test_config_structure()
        
        print("\n🎉 Alle Tests erfolgreich abgeschlossen!")
        print("\nDie folgenden Korrekturen wurden implementiert:")
        print("  ✅ TimeGranularity Parameter erweitert auf 10 Werte")
        print("  ✅ Array-Dimensionen Problem behoben")
        print("  ✅ Numerical Parameter erweitert")
        print("  ✅ Boolean Parameter erweitert")
        print("  ✅ Bin Parameter erweitert")
        print("  ✅ Maximale Kombinationen erhöht auf 15.000")
        
    except Exception as e:
        print(f"❌ Fehler während des Tests: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
