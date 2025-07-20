#!/usr/bin/env python3
"""
Test Script für die korrigierten Parameter-Kombinationen
======================================================

Dieses Script testet die Anzahl der generierten Parameter-Kombinationen
und zeigt die ersten paar Kombinationen an.
"""

import configparser
import sys
import os
from Training_Auto_ParameterSweep_Enhanced import (
    ParameterManager, 
    TIME_GRANULARITY_PARAMETERS,
    BOOLEAN_PARAMETERS,
    NUMERICAL_PARAMETERS,
    NUM_BINS_VALUES
)

def test_parameter_counts():
    """Teste die Anzahl der generierten Parameter-Kombinationen."""
    
    print("🔍 PARAMETER-KOMBINATIONEN TEST")
    print("=" * 50)
    
    # Basis-Konfiguration laden
    config = configparser.ConfigParser()
    config.read('MyConfig.ini')
    
    # Parameter Manager initialisieren
    param_manager = ParameterManager(config)
    
    # Test verschiedene Modi
    modes = [
        "bins_only",
        "boolean_only", 
        "numerical_only",
        "time_granularity_only",
        "full_parameter_sweep"
    ]
    
    for mode in modes:
        print(f"\n📊 Modus: {mode}")
        print("-" * 30)
        
        try:
            combinations = param_manager.generate_parameter_combinations(mode)
            print(f"Anzahl Kombinationen: {len(combinations)}")
            
            # Zeige erste 2 Kombinationen
            for i, combo in enumerate(combinations[:2]):
                print(f"  {i+1}. {combo}")
            
            if len(combinations) > 2:
                print(f"  ... und {len(combinations) - 2} weitere")
                
        except Exception as e:
            print(f"❌ Fehler in Modus {mode}: {e}")
    
    # Zeige Parameter-Bereiche
    print(f"\n📋 PARAMETER-BEREICHE:")
    print("-" * 30)
    print(f"TimeGranularity Werte: {TIME_GRANULARITY_PARAMETERS['Candle']['TimeGranularity']}")
    print(f"Boolean Parameter Sektionen: {list(BOOLEAN_PARAMETERS.keys())}")
    print(f"Numerical Parameter Sektionen: {list(NUMERICAL_PARAMETERS.keys())}")
    print(f"Bin Werte: {NUM_BINS_VALUES}")
    
    # Berechne theoretische Kombinationen
    print(f"\n🧮 THEORETISCHE BERECHNUNGEN:")
    print("-" * 30)
    
    # TimeGranularity
    time_count = len(TIME_GRANULARITY_PARAMETERS['Candle']['TimeGranularity'])
    print(f"TimeGranularity: {time_count} Werte")
    
    # Boolean (mit XOR-Validierung)
    bool_count = 0
    for section, params in BOOLEAN_PARAMETERS.items():
        section_combinations = 1
        for param, values in params.items():
            section_combinations *= len(values)
        bool_count += section_combinations
    print(f"Boolean (vor Validierung): ~{bool_count} Kombinationen")
    
    # Numerical
    num_count = 0
    for section, params in NUMERICAL_PARAMETERS.items():
        for param, values in params.items():
            num_count += len(values)
    print(f"Numerical: {num_count} Einzelparameter")
    
    # Bins
    bin_count = len(NUM_BINS_VALUES) * len(NUM_BINS_VALUES)
    print(f"Bins: {bin_count} Kombinationen")

def test_xor_validation():
    """Teste die XOR-Validierung für UseLogReturns/UseSimpleReturns."""
    
    print(f"\n🔍 XOR-VALIDIERUNG TEST")
    print("=" * 30)
    
    # Basis-Konfiguration laden
    config = configparser.ConfigParser()
    config.read('MyConfig.ini')
    
    param_manager = ParameterManager(config)
    
    # Test verschiedene Kombinationen
    test_cases = [
        {'DEFAULT': {'UseLogReturns': True, 'UseSimpleReturns': False}},  # Gültig
        {'DEFAULT': {'UseLogReturns': False, 'UseSimpleReturns': True}},  # Gültig
        {'DEFAULT': {'UseLogReturns': True, 'UseSimpleReturns': True}},   # Ungültig
        {'DEFAULT': {'UseLogReturns': False, 'UseSimpleReturns': False}}, # Ungültig
        {'DEFAULT': {'UseLogReturns': True, 'SmoothLogReturns': True}},   # Gültig
        {'DEFAULT': {'UseLogReturns': False, 'SmoothLogReturns': True}},  # Ungültig
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        result = param_manager.validate_parameter_combination(test_case)
        status = "✅ Gültig" if result else "❌ Ungültig"
        print(f"  Test {i}: {test_case} -> {status}")

def main():
    """Hauptfunktion für den Test."""
    
    print("🎯 PARAMETER SWEEP SYSTEM TEST")
    print("=" * 50)
    
    # Prüfe ob Config-Datei existiert
    if not os.path.exists('MyConfig.ini'):
        print("❌ Fehler: MyConfig.ini nicht gefunden!")
        print("Erstelle eine Test-Konfiguration...")
        
        # Erstelle eine minimale Test-Konfiguration
        config = configparser.ConfigParser()
        config.add_section('DEFAULT')
        config.set('DEFAULT', 'UseLogReturns', 'True')
        config.set('DEFAULT', 'UseSimpleReturns', 'False')
        config.set('DEFAULT', 'SmoothLogReturns', 'False')
        config.set('DEFAULT', 'SmoothSimpleReturns', 'False')
        
        with open('MyConfig.ini', 'w') as f:
            config.write(f)
        
        print("✅ Test-Konfiguration erstellt.")
    
    try:
        # Test 1: Parameter-Kombinationen
        test_parameter_counts()
        
        # Test 2: XOR-Validierung
        test_xor_validation()
        
        print("\n🎉 Test abgeschlossen!")
        
    except Exception as e:
        print(f"❌ Fehler während des Tests: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
