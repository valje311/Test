#!/usr/bin/env python3
"""
Enhanced Parameter Sweep System for Trading Analysis - NO CONFIG VERSION
=======================================================================

Eigenständiges System für systematische Parameter-Tests OHNE MyConfig.ini.
Alle Konfigurationen sind direkt im Code enthalten.

Features:
- Parameter-Kombinationen (Boolean, Numerical, Bins, TimeGranularity)
- Intelligente Validierung
- Robuste Array-Behandlung
- Komplette Eigenständigkeit - KEINE externe Config-Datei erforderlich
- Alle ursprünglichen MyConfig.ini Werte direkt im Code

Author: Trading Analysis System
Date: Juli 2025
"""

import os
import json
import copy
import itertools
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Lokale Module (werden auch in NoConfig_System kopiert)
import FeatureSpace_NoConfig as FeatureSpace
import TimeSeriesAnalysis_NoConfig as TimeSeriesAnalysis
import TimeSeriesManipulation_NoConfig as TimeSeriesManipulation

# ==================================================================================
# KOMPLETT INTEGRIERTE KONFIGURATION (ERSETZT MyConfig.ini)
# ==================================================================================

# VOLLSTÄNDIGE Konfiguration für SYSTEM_MODE = "config"
# Alle ursprünglichen MyConfig.ini Werte sind hier fest codiert:
INTEGRATED_CONFIG = {
    'DEFAULT': {
        'SmoothTicks': True,
        'SmoothTicksMethod': 'Perona-Malik',
        'SmoothCandles': False,
        'SmoothCandlesMethod': 'Perona-Malik',
        'UseLogReturns': True,
        'UseSimpleReturns': False,
        'SmoothLogReturns': False,
        'SmoothLogReturnsMethod': 'Perona-Malik',
        'SmoothSimpleReturns': False
    },
    'CSV': {
        'FilePath': r'D:\export_WTI_Tick_01012020_13052025.csv',
        'ChunkSize': 5000000
    },
    'SQL': {
        'UserName': 'Filler',
        'Password': '5gs.1-',
        'Port': 3306,
        'Database': 'trading',
        'TableName': 'oil',
        'TimeColName': 'Timestamp',
        'DataColName': 'Close',
        'StartDate': '01.09.2021 01:00:00',
        'EndDate': '30.09.2021 23:59:58'
    },
    'Candle': {
        'TimeGranularity': '30s'
    },
    'Perona-Malik': {
        'Iterations': 200,
        'TimeStep': 0.1,
        'Kappa': 2
    },
    'Autocorrelation': {
        'MaxLag': 100,
        'Lag': 4
    },
    'Mutual Information': {
        'UseFriedmanDiaconis': False,
        'NumBinsX': 275,
        'NumBinsY': 350,
        'MaxLag': 100
    },
    'False Nearest Neighbour': {
        'MaxDim': 15,
        'RTol': 15,
        'ATol': 3
    },
    'Takens Embedding': {
        'UseUserDefinedParameters': True,
        'TimeDelay': 3,
        'EmbeddingDim': 3
    }
}

# MINIMALE Konfiguration für SYSTEM_MODE = "sweep" 
# Nur die essentiellen Werte für Datenbank/CSV-Zugriff:
MINIMAL_CONFIG = {
    'CSV': {
        'FilePath': r'D:\export_WTI_Tick_01012020_13052025.csv',
        'ChunkSize': 5000000
    },
    'SQL': {
        'UserName': 'Filler',
        'Password': '5gs.1-',
        'Port': 3306,
        'Database': 'trading',
        'TableName': 'oil',
        'TimeColName': 'Timestamp',
        'DataColName': 'Close',
        'StartDate': '01.09.2021 01:00:00',
        'EndDate': '30.09.2021 23:59:58'
    }
}

# ==================================================================================
# SYSTEM MODUS KONFIGURATION
# ==================================================================================

# HAUPT-SCHALTER - Wähle zwischen zwei Betriebsmodi:
# "config": Verwendet alle INTEGRATED_CONFIG Werte für EINEN Durchlauf (wie MyConfig.ini)
# "sweep": Macht Parameter-Sweeps nur mit minimalen Config-Werten (CSV/SQL)
SYSTEM_MODE = "config"  # <-- HIER ÄNDERN! ("config" oder "sweep")

# ==================================================================================
# PARAMETER SWEEP KONFIGURATION (nur für SYSTEM_MODE = "sweep")
# ==================================================================================

# SWEEP-MODUS EINSTELLUNG - Hier kannst du den gewünschten Sweep-Modus einstellen:
# Verfügbare Modi: "boolean", "numerical", "bins", "time_granularity", "full"
SWEEP_MODE = "full"  # <-- HIER ÄNDERN!

# ZUFALLS-EINSTELLUNG - Kombinationen in zufälliger Reihenfolge durchlaufen
SHUFFLE_COMBINATIONS = True  # <-- HIER ÄNDERN! (True = zufällig, False = original Reihenfolge)

NUM_BINS_VALUES = [300]
MAX_COMBINATIONS = 50000  # Erhöht für mehr Kombinationen - kannst du beliebig anpassen!

BOOLEAN_PARAMS = {
    'DEFAULT': {
        'SmoothTicks': [False],
        'SmoothCandles': [True],
        'UseLogReturns': [True],
        'UseSimpleReturns': [False],
        'SmoothLogReturns': [False],
        'SmoothSimpleReturns': [False]
    },
    'Mutual Information': {
        'UseFriedmanDiaconis': [False]
    },
    'Takens Embedding': {
        'UseUserDefinedParameters': [True]
    }
}

NUMERICAL_PARAMS = {
    'Perona-Malik': {
        'Iterations': [75, 100, 125, 150, 175, 200],
        'TimeStep': [0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25],
        'Kappa': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    },
    'Autocorrelation': {
        'MaxLag': [100, 150, 200, 250, 300],
        'Lag': [4, 6, 8, 10, 12, 14, 16, 18]
    },
    'False Nearest Neighbour': {
        'MaxDim': [15],
        'RTol': [10, 15, 20, 25, 30],
        'ATol': [4, 6, 8, 10, 12]
    },
    'Takens Embedding': {
        'TimeDelay': [4, 6, 8, 10],
        'EmbeddingDim': [4, 6, 8, 10]
    }
}

TIME_GRANULARITY_PARAMS = {
    'Candle': {
        'TimeGranularity': ['3min']
    }
}

# ==================================================================================
# CONFIG DICT KLASSE (ERSETZT configparser.ConfigParser)
# ==================================================================================

class ConfigDict:
    """Ersetzt configparser.ConfigParser mit einem einfachen Dictionary-basierten System."""
    
    def __init__(self, data: Optional[Dict[str, Dict[str, Any]]] = None, use_minimal: bool = False):
        # Wähle die richtige Basis-Konfiguration je nach Modus
        if data is not None:
            self.data = copy.deepcopy(data)
        elif use_minimal:
            # Für Parameter-Sweeps: Nur minimale Config verwenden
            self.data = copy.deepcopy(MINIMAL_CONFIG)
        else:
            # Für Config-Modus: Vollständige integrierte Config verwenden
            self.data = copy.deepcopy(INTEGRATED_CONFIG)
    
    def get(self, section: str, key: str, fallback: Any = None) -> Any:
        """Wert aus Konfiguration abrufen."""
        try:
            return str(self.data[section][key])
        except KeyError:
            return fallback
    
    def getboolean(self, section: str, key: str, fallback: bool = False) -> bool:
        """Boolean-Wert aus Konfiguration abrufen."""
        try:
            value = self.data[section][key]
            if isinstance(value, bool):
                return value
            elif isinstance(value, str):
                return value.lower() in ('true', '1', 'yes', 'on')
            else:
                return bool(value)
        except KeyError:
            return fallback
    
    def getint(self, section: str, key: str, fallback: int = 0) -> int:
        """Integer-Wert aus Konfiguration abrufen."""
        try:
            return int(self.data[section][key])
        except (KeyError, ValueError):
            return fallback
    
    def getfloat(self, section: str, key: str, fallback: float = 0.0) -> float:
        """Float-Wert aus Konfiguration abrufen."""
        try:
            return float(self.data[section][key])
        except (KeyError, ValueError):
            return fallback
    
    def has_section(self, section: str) -> bool:
        """Prüfen ob Sektion existiert."""
        return section in self.data
    
    def has_option(self, section: str, key: str) -> bool:
        """Prüfen ob Option in Sektion existiert."""
        return section in self.data and key in self.data[section]
    
    def sections(self) -> List[str]:
        """Alle Sektionen auflisten."""
        return [s for s in self.data.keys() if s != 'DEFAULT']
    
    def set(self, section: str, key: str, value: Any) -> None:
        """Wert setzen."""
        if section not in self.data:
            self.data[section] = {}
        self.data[section][key] = value
    
    def add_section(self, section: str) -> None:
        """Sektion hinzufügen."""
        if section not in self.data:
            self.data[section] = {}
    
    def __getitem__(self, section: str) -> Dict[str, Any]:
        """Dictionary-Style Access."""
        return self.data.get(section, {})
    
    def __contains__(self, section: str) -> bool:
        """'in' operator support."""
        return section in self.data
    
    def write(self, file_handle):
        """Schreibe Konfiguration in Datei (INI-Format)."""
        for section_name, section_data in self.data.items():
            file_handle.write(f"[{section_name}]\n")
            for key, value in section_data.items():
                file_handle.write(f"{key} = {value}\n")
            file_handle.write("\n")

# ==================================================================================
# PARAMETER MANAGER
# ==================================================================================

class ParameterManager:
    """Parameter-Kombinationen und Validierung."""
    
    def __init__(self, use_minimal_config: bool = False):
        # Erstelle Config basierend auf Modus
        self.config = ConfigDict(use_minimal=use_minimal_config)
        self.use_minimal = use_minimal_config
    
    def validate_params(self, params: Dict[str, Dict[str, Any]]) -> bool:
        """Validiert Parameter-Kombinationen."""
        default_params = params.get('DEFAULT', {})
        
        # XOR für UseLogReturns und UseSimpleReturns
        use_log = default_params.get('UseLogReturns', 
                                   self.config.getboolean('DEFAULT', 'UseLogReturns', fallback=True))
        use_simple = default_params.get('UseSimpleReturns', 
                                      self.config.getboolean('DEFAULT', 'UseSimpleReturns', fallback=False))
        
        if use_log == use_simple:
            return False
        
        # Smoothing-Validierung
        smooth_log = default_params.get('SmoothLogReturns', False)
        smooth_simple = default_params.get('SmoothSimpleReturns', False)
        
        if smooth_log and not use_log:
            return False
        if smooth_simple and not use_simple:
            return False
        
        return True
    
    def generate_combinations(self, mode: str = "full") -> List[Dict[str, Dict[str, Any]]]:
        """Generiert Parameter-Kombinationen."""
        modes = {
            "boolean": self._gen_boolean_combinations,
            "numerical": self._gen_numerical_combinations,
            "bins": self._gen_bin_combinations,
            "time_granularity": self._gen_time_combinations,
            "full": self._gen_full_combinations
        }
        
        if mode not in modes:
            raise ValueError(f"Unknown mode: {mode}")
        
        return modes[mode]()
    
    def _gen_boolean_combinations(self) -> List[Dict[str, Dict[str, Any]]]:
        """Boolean Parameter-Kombinationen."""
        combinations = []
        
        for section, params in BOOLEAN_PARAMS.items():
            param_names = list(params.keys())
            param_values = list(params.values())
            
            for combo in itertools.product(*param_values):
                param_dict = dict(zip(param_names, combo))
                test_params = {section: param_dict}
                
                if self.validate_params(test_params):
                    combinations.append(test_params)
        
        return combinations
    
    def _gen_numerical_combinations(self) -> List[Dict[str, Dict[str, Any]]]:
        """Numerical Parameter-Kombinationen."""
        combinations = []
        
        for section, params in NUMERICAL_PARAMS.items():
            for param_name, values in params.items():
                for value in values:
                    combo = {section: {param_name: value}}
                    combinations.append(combo)
        
        return combinations
    
    def _gen_bin_combinations(self) -> List[Dict[str, Dict[str, Any]]]:
        """Bin Parameter-Kombinationen (quadratisch: NumBinsX = NumBinsY)."""
        combinations = []
        
        # Verwende nur quadratische Bins (X = Y) für korrekte Mutual Information Analyse
        for bins in NUM_BINS_VALUES:
            combo = {
                'Mutual Information': {
                    'NumBinsX': bins,
                    'NumBinsY': bins  # Immer gleich wie NumBinsX
                }
            }
            combinations.append(combo)
        
        return combinations
    
    def _gen_time_combinations(self) -> List[Dict[str, Dict[str, Any]]]:
        """TimeGranularity Parameter-Kombinationen."""
        combinations = []
        
        for section, params in TIME_GRANULARITY_PARAMS.items():
            for param_name, values in params.items():
                for value in values:
                    combo = {section: {param_name: value}}
                    combinations.append(combo)
        
        return combinations
    
    def _gen_full_combinations(self) -> List[Dict[str, Dict[str, Any]]]:
        """Vollständige Parameter-Kombinationen - ALLE Kombinationen ohne Begrenzung."""
        combinations = []
        
        # ALLE Kombinationen ohne Begrenzung
        bool_combos = self._gen_boolean_combinations()
        num_combos = self._gen_numerical_combinations()
        bin_combos = self._gen_bin_combinations()
        time_combos = self._gen_time_combinations()
        
        print(f"🔢 Generating full combinations:")
        print(f"   Boolean: {len(bool_combos)} combinations")
        print(f"   Numerical: {len(num_combos)} combinations")
        print(f"   Bins: {len(bin_combos)} combinations")
        print(f"   Time: {len(time_combos)} combinations")
        print(f"   Total: {len(bool_combos)} × {len(num_combos)} × {len(bin_combos)} × {len(time_combos)} = {len(bool_combos) * len(num_combos) * len(bin_combos) * len(time_combos)} combinations")
        
        for bool_combo in bool_combos:
            for num_combo in num_combos:
                for bin_combo in bin_combos:
                    for time_combo in time_combos:
                        # Kombinationen mergen
                        merged = copy.deepcopy(bool_combo)
                        
                        for combo_dict in [num_combo, bin_combo, time_combo]:
                            for sect_name, sect_params in combo_dict.items():
                                if sect_name in merged:
                                    merged[sect_name].update(sect_params)
                                else:
                                    merged[sect_name] = sect_params
                        
                        if self.validate_params(merged):
                            combinations.append(merged)
                            
                            # Nur MAX_COMBINATIONS prüfen, nicht begrenzen
                            if len(combinations) >= MAX_COMBINATIONS:
                                print(f"⚠️  Warning: Reached MAX_COMBINATIONS limit of {MAX_COMBINATIONS}")
                                print(f"   Increase MAX_COMBINATIONS in code if you want more!")
                                return combinations
        
        return combinations

# ==================================================================================
# ANALYSIS RUNNER
# ==================================================================================

class AnalysisRunner:
    """Analysen mit Parameter-Kombinationen."""
    
    def __init__(self, use_minimal_config: bool = False):
        # Erstelle Config basierend auf Modus
        self.config = ConfigDict(use_minimal=use_minimal_config)
        self.param_manager = ParameterManager(use_minimal_config=use_minimal_config)
        self.previous_config = None  # Speichert die vorherige Konfiguration
        self.use_minimal = use_minimal_config
    
    def print_parameter_overview(self, current_config: ConfigDict, run_id: str) -> None:
        """Gibt eine saubere Übersicht aller Parameter ohne Wiederholungen aus."""
        print(f"\n🔧 Parameter Overview - Run: {run_id}")
        print("=" * 80)
        
        # DEFAULT-Sektion zuerst anzeigen
        if current_config.has_section('DEFAULT') or 'DEFAULT' in current_config:
            print(f"\n[DEFAULT]")
            if 'DEFAULT' in current_config:
                for key in current_config['DEFAULT']:
                    current_value = current_config.get('DEFAULT', key)
                    self._print_parameter_with_changes('DEFAULT', key, current_value)
        
        # Alle anderen Sektionen durchgehen (ohne DEFAULT)
        for section_name in current_config.sections():
            if section_name == 'DEFAULT':
                continue  # DEFAULT bereits oben behandelt
                
            print(f"\n[{section_name}]")
            
            # Nur die Parameter anzeigen, die spezifisch für diese Sektion sind
            # (nicht die von DEFAULT geerbten)
            section = current_config[section_name]
            
            # Alle Parameter der Sektion durchgehen
            for key in section:
                # Prüfen ob dieser Parameter in der ursprünglichen INI-Datei für diese Sektion steht
                # oder ob er nur von DEFAULT geerbt wurde
                if self._is_section_specific_parameter(section_name, key):
                    current_value = current_config.get(section_name, key)
                    self._print_parameter_with_changes(section_name, key, current_value)
        
        print("=" * 80)
    
    def _is_section_specific_parameter(self, section_name: str, param_name: str) -> bool:
        """Prüft ob ein Parameter spezifisch für diese Sektion ist (nicht von DEFAULT geerbt)."""
        # Diese Parameter gehören spezifisch zu ihren Sektionen:
        section_specific_params = {
            'CSV': ['filepath', 'chunksize'],
            'SQL': ['username', 'password', 'port', 'database', 'tablename', 'timecolname', 'datacolname', 'startdate', 'enddate'],
            'Candle': ['timegranularity'],
            'Perona-Malik': ['iterations', 'timestep', 'kappa'],
            'Autocorrelation': ['maxlag', 'lag'],
            'Mutual Information': ['usefriedmandiaconis', 'numbinsx', 'numbinsy', 'maxlag'],
            'False Nearest Neighbour': ['maxdim', 'rtol', 'atol'],
            'Takens Embedding': ['useuserdefinedparameters', 'timedelay', 'embeddingdim']
        }
        
        # Parameter-Name normalisieren (case-insensitive)
        param_lower = param_name.lower()
        section_params = section_specific_params.get(section_name, [])
        
        return param_lower in [p.lower() for p in section_params]
    
    def _print_parameter_with_changes(self, section_name: str, key: str, current_value: str) -> None:
        """Hilfsfunktion zum Anzeigen von Parametern mit Änderungsmarkierung."""
        # Vorherigen Wert prüfen
        if self.previous_config and self.previous_config.has_section(section_name):
            if self.previous_config.has_option(section_name, key):
                previous_value = self.previous_config.get(section_name, key)
                
                # Änderung anzeigen
                if current_value != previous_value:
                    print(f"  {key} = {current_value} 🔄[{previous_value}]")
                else:
                    print(f"  {key} = {current_value}")
            else:
                # Neuer Parameter
                print(f"  {key} = {current_value} ✨[NEW]")
        else:
            # Erste Ausführung oder neue Sektion
            print(f"  {key} = {current_value}")
    
    def apply_params(self, params: Dict[str, Dict[str, Any]]) -> ConfigDict:
        """Parameter auf Konfiguration anwenden."""
        config_copy = ConfigDict(self.config.data)
        
        for section_name, section_params in params.items():
            # DEFAULT ist eine spezielle Sektion, die immer existiert
            if section_name != 'DEFAULT' and not config_copy.has_section(section_name):
                config_copy.add_section(section_name)
            
            for param_name, param_value in section_params.items():
                config_copy.set(section_name, param_name, param_value)
        
        return config_copy
    
    def generate_run_id(self, params: Dict[str, Dict[str, Any]]) -> str:
        """Eindeutige Run-ID generieren."""
        if not params:
            return "single_run"
        
        parts = []
        for section, sect_params in params.items():
            section_short = section.replace(' ', '')[:3]
            for param, value in sect_params.items():
                param_short = param.replace(' ', '')[:3]
                value_short = str(value)[:4]
                parts.append(f"{section_short}_{param_short}_{value_short}")
        
        return "_".join(parts)
    
    def run_single_analysis(self, params: Dict[str, Dict[str, Any]], df: pd.DataFrame) -> Dict[str, Any]:
        """Einzelne Analyse durchführen."""
        config = self.apply_params(params)
        run_id = self.generate_run_id(params)
        
        # Parameter-Übersicht anzeigen
        self.print_parameter_overview(config, run_id)
        
        # Output-Verzeichnis
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        table_name = config.get('SQL', 'TableName', fallback='default')
        plots_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            'Plots', 
            table_name,
            f'{timestamp}_{run_id}'
        )
        os.makedirs(plots_dir, exist_ok=True)
        
        print(f"\n🔬 Running: {run_id}")
        
        try:
            # Analyse ausführen
            self._execute_analysis(df, config, plots_dir, run_id)
            
            # Konfiguration speichern
            self._save_config(config, params, plots_dir, run_id)
            
            # Aktuelle Konfiguration als vorherige speichern
            self.previous_config = ConfigDict(config.data)
            
            return {
                'run_id': run_id,
                'parameters': params,
                'plots_dir': plots_dir,
                'status': 'success',
                'timestamp': timestamp
            }
            
        except Exception as e:
            print(f"❌ Failed: {run_id} - {str(e)}")
            
            # Auch bei Fehler die Konfiguration speichern
            self.previous_config = ConfigDict(config.data)
            
            return {
                'run_id': run_id,
                'parameters': params,
                'plots_dir': plots_dir,
                'status': 'failed',
                'error': str(e),
                'timestamp': timestamp
            }
    
    def _execute_analysis(self, df: pd.DataFrame, config: ConfigDict, 
                         plots_dir: str, run_id: str) -> None:
        """Eigentliche Analyse durchführen."""
        df_work = df.copy()
        data_col = config.get('SQL', 'DataColName', fallback='Close')
        time_col = config.get('SQL', 'TimeColName', fallback='Timestamp')
        
        # Tick-Smoothing
        if config.getboolean('DEFAULT', 'SmoothTicks', fallback=False):
            if config.get('DEFAULT', 'SmoothTicksMethod', fallback='') == 'Perona-Malik':
                df_work[data_col] = TimeSeriesManipulation.perona_malik_smoothing(
                    df_work[time_col], 
                    df_work[data_col].tolist(), 
                    config, plots_dir
                )
        
        # Candles erstellen
        candles = FeatureSpace.createCandleDataFrame(df_work, config, plots_dir)
        
        # Candle-Smoothing
        if config.getboolean('DEFAULT', 'SmoothCandles', fallback=False):
            if config.get('DEFAULT', 'SmoothCandlesMethod', fallback='') == 'Perona-Malik':
                candles[data_col] = TimeSeriesManipulation.perona_malik_smoothing(
                    candles[time_col], 
                    candles[data_col].tolist(), 
                    config, plots_dir
                )
        
        # Returns berechnen
        returns = self._calculate_returns(candles, config, plots_dir)
        
        if returns is not None:
            # Time Series Analysis - VOLLSTÄNDIGE Analyse!
            print("🔬 Running Autocorrelation Analysis...")
            TimeSeriesAnalysis.calculate_autocorrelation(returns, config, plots_dir)
            
            print("🔬 Running Takens Embedding Analysis...")
            TimeSeriesAnalysis.TakenEmbedding(returns, plots_dir, config)
            
            print("🔬 Running Mutual Information Analysis...")
            # Mutual Information mit den konfigurierten Parametern
            max_lag = config.getint('Mutual Information', 'MaxLag', fallback=100)
            for lag in range(1, min(max_lag + 1, len(returns) // 10)):  # Bis max_lag oder vernünftige Grenze
                try:
                    TimeSeriesAnalysis.calculate_mutual_information(returns, lag, config, plots_dir)
                except Exception as e:
                    print(f"⚠️  Warning: Mutual Information failed for lag {lag}: {str(e)}")
                    break  # Bei Fehler aufhören
            
            print("🔬 Running False Nearest Neighbors Analysis...")
            # False Nearest Neighbors
            try:
                # Time delay aus Autocorrelation oder Konfiguration
                tau = config.getint('Autocorrelation', 'Lag', fallback=4)
                TimeSeriesAnalysis.calculate_false_nearest_neighbors(returns, tau, config, plots_dir)
            except Exception as e:
                print(f"⚠️  Warning: False Nearest Neighbors failed: {str(e)}")
            
            print("🔬 Running Visualization...")
            # Visualisierung
            self._create_visualization(candles, returns, config, plots_dir, run_id)
    
    def _calculate_returns(self, candles: pd.DataFrame, config: ConfigDict, 
                          plots_dir: str) -> Optional[Any]:
        """Returns berechnen."""
        data_col = config.get('SQL', 'DataColName', fallback='Close')
        time_col = config.get('SQL', 'TimeColName', fallback='Timestamp')
        
        returns = None
        
        if config.getboolean('DEFAULT', 'UseLogReturns', fallback=True):
            returns = TimeSeriesManipulation.getLogReturns(candles[data_col].tolist())
            
            if config.getboolean('DEFAULT', 'SmoothLogReturns', fallback=False):
                if config.get('DEFAULT', 'SmoothLogReturnsMethod', fallback='') == 'Perona-Malik':
                    returns = TimeSeriesManipulation.perona_malik_smoothing(
                        candles[time_col], returns, config, plots_dir
                    )
        
        elif config.getboolean('DEFAULT', 'UseSimpleReturns', fallback=False):
            returns = TimeSeriesManipulation.getSimpleReturns(candles[data_col].tolist())
            
            if config.getboolean('DEFAULT', 'SmoothSimpleReturns', fallback=False):
                if config.get('DEFAULT', 'SmoothSimpleReturnsMethod', fallback='') == 'Perona-Malik':
                    returns = TimeSeriesManipulation.perona_malik_smoothing(
                        candles[time_col], returns, config, plots_dir
                    )
        
        return returns
    
    def _create_visualization(self, candles: pd.DataFrame, returns: Any, 
                            config: ConfigDict, plots_dir: str, run_id: str) -> None:
        """Visualisierung erstellen - verwendet die TATSÄCHLICH berechneten Returns."""
        time_col = config.get('SQL', 'TimeColName', fallback='Timestamp')
        lag = config.getint('Autocorrelation', 'Lag', fallback=2)
        
        if len(returns) <= 0:
            print("⚠️  Warning: No returns data available for visualization")
            return
        
        print(f"📊 Visualizing {len(returns)} returns values (calculated from {len(candles)} candles)")
        
        # Returns-Array: Die TATSÄCHLICHEN berechneten Werte verwenden
        returns_array = returns  # Vollständige berechnete Returns
        
        # Time-Array: Returns haben einen Wert weniger als Candles (da sie Differenzen sind)
        if len(candles) > len(returns):
            time_array = candles[time_col].iloc[1:len(returns)+1]
        else:
            time_array = candles[time_col].iloc[:len(returns)]
        
        # Validierung der Array-Größen
        if len(time_array) != len(returns_array):
            print(f"⚠️  Warning: Array size mismatch - Time: {len(time_array)}, Returns: {len(returns_array)}")
            # Sichere Anpassung
            min_length = min(len(time_array), len(returns_array))
            time_array = time_array[:min_length]
            returns_array = returns_array[:min_length]
        
        print(f"📈 Final arrays - Time: {len(time_array)}, Returns: {len(returns_array)}")
        
        # 1. Time Series Plot der Returns
        plt.figure(figsize=(12, 6))
        plt.plot(time_array, returns_array, label='Returns', color='blue', alpha=0.7)
        
        # Lagged Returns
        if lag > 0 and lag < len(returns_array):
            lagged_returns = returns_array[lag:]
            lagged_time = time_array.iloc[:-lag] if len(time_array) > lag else time_array
            
            # Sicherstellen, dass beide Arrays gleich lang sind
            min_lag_length = min(len(lagged_time), len(lagged_returns))
            lagged_time = lagged_time[:min_lag_length]
            lagged_returns = lagged_returns[:min_lag_length]
            
            plt.plot(lagged_time, lagged_returns, 
                    label=f'Returns (Lag {lag})', color='orange', alpha=0.7)
        
        plt.title(f'Returns Time Series - {run_id}')
        plt.xlabel('Time')
        plt.ylabel('Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'returns_timeseries.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Histogram der Returns
        plt.figure(figsize=(10, 6))
        
        # Histogram mit automatischer Bin-Anzahl
        n_bins = min(50, max(10, len(returns_array) // 50))  # Adaptive Bin-Anzahl
        
        plt.hist(returns_array, bins=n_bins, alpha=0.7, color='skyblue', 
                edgecolor='black', linewidth=0.5, density=True)
        
        # Statistiken berechnen
        import numpy as np
        mean_val = np.mean(returns_array)
        std_val = np.std(returns_array)
        median_val = np.median(returns_array)
        
        # Vertikale Linien für Statistiken
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.6f}')
        plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.6f}')
        plt.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7, label=f'+1σ: {mean_val + std_val:.6f}')
        plt.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7, label=f'-1σ: {mean_val - std_val:.6f}')
        
        # Normalverteilung als Referenz
        x_range = np.linspace(min(returns_array), max(returns_array), 100)
        normal_dist = (1 / (std_val * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean_val) / std_val)**2)
        plt.plot(x_range, normal_dist, 'r-', linewidth=2, alpha=0.8, label='Normal Distribution')
        
        plt.title(f'Returns Distribution Histogram - {run_id}')
        plt.xlabel('Returns')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Statistik-Text hinzufügen
        returns_type = "Log Returns" if config.getboolean('DEFAULT', 'UseLogReturns', fallback=True) else "Simple Returns"
        smoothed = config.getboolean('DEFAULT', 'SmoothLogReturns', fallback=False) or config.getboolean('DEFAULT', 'SmoothSimpleReturns', fallback=False)
        smooth_info = " (Smoothed)" if smoothed else ""
        
        stats_text = f'{returns_type}{smooth_info}\nStatistics:\nCount: {len(returns_array)}\nMean: {mean_val:.6f}\nStd: {std_val:.6f}\nMin: {min(returns_array):.6f}\nMax: {max(returns_array):.6f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'returns_histogram.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Created visualizations using ACTUAL calculated returns:")
        print(f"   - {returns_type}{smooth_info}")
        print(f"   - {len(returns_array)} data points")
        print(f"   - Mean: {mean_val:.6f}, Std: {std_val:.6f}")
        print(f"   - Files: returns_timeseries.png & returns_histogram.png")
    
    def _save_config(self, config: ConfigDict, params: Dict[str, Dict[str, Any]], 
                    plots_dir: str, run_id: str) -> None:
        """Konfiguration als INI-Datei speichern."""
        # Vollständige Konfiguration als INI speichern
        config_file_path = os.path.join(plots_dir, 'config.ini')
        
        # Kommentar-Header hinzufügen
        with open(config_file_path, 'w') as f:
            f.write(f"# Trading Analysis Configuration - NoConfig System\n")
            f.write(f"# Run ID: {run_id}\n")
            f.write(f"# Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"# Changed Parameters: {params}\n")
            f.write(f"# This config was generated automatically - no external MyConfig.ini required\n")
            f.write(f"\n")
        
        # Konfiguration im INI-Format speichern
        with open(config_file_path, 'a') as f:
            config.write(f)
    
    def run_parameter_sweep(self, df: pd.DataFrame, mode: str = "full") -> List[Dict[str, Any]]:
        """Parameter-Sweep durchführen."""
        print(f"🚀 Starting Parameter Sweep - Mode: {mode}")
        
        combinations = self.param_manager.generate_combinations(mode)
        print(f"📊 Generated {len(combinations)} combinations")
        
        # Zufällige Reihenfolge wenn aktiviert
        if SHUFFLE_COMBINATIONS:
            random.shuffle(combinations)
            print(f"🔀 Shuffled combinations into random order")
        else:
            print(f"📋 Using original order")
        
        print(f"🎯 Testing {len(combinations)} combinations")
        
        results = []
        for i, combo in enumerate(combinations, 1):
            print(f"📈 Progress: {i}/{len(combinations)}")
            
            result = self.run_single_analysis(combo, df)
            results.append(result)
            
            status = "✅" if result['status'] == 'success' else "❌"
            print(f"{status} {i}: {result['status']}")
        
        self._save_sweep_results(results, mode)
        return results
    
    def _save_sweep_results(self, results: List[Dict[str, Any]], mode: str) -> None:
        """Sweep-Ergebnisse speichern."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            'ParameterSweepResults', 
            f'{mode}_{timestamp}'
        )
        os.makedirs(results_dir, exist_ok=True)
        
        # Ergebnisse speichern
        with open(os.path.join(results_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Summary
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']
        
        summary = f"""Parameter Sweep Results - NoConfig System
Mode: {mode}
Total: {len(results)}
Successful: {len(successful)}
Failed: {len(failed)}
Success Rate: {len(successful)/len(results)*100:.1f}%
Shuffled: {SHUFFLE_COMBINATIONS}

System Info:
- No external MyConfig.ini required
- All configuration integrated in code
- Standalone parameter sweep system
"""
        
        with open(os.path.join(results_dir, 'summary.txt'), 'w') as f:
            f.write(summary)
        
        print(f"📊 Results saved to: {results_dir}")

# ==================================================================================
# UTILITY FUNCTIONS
# ==================================================================================

def load_data(config: ConfigDict) -> pd.DataFrame:
    """Daten aus Datenbank laden."""
    from sqlalchemy import create_engine, text
    
    # Connection
    conn_str = (f"mysql+mysqlconnector://"
               f"{config.get('SQL', 'UserName')}:{config.get('SQL', 'Password')}@"
               f"localhost:{config.get('SQL', 'Port')}/{config.get('SQL', 'Database')}")
    engine = create_engine(conn_str)
    
    # Zeitraum
    start_date = datetime.strptime(config.get('SQL', 'StartDate'), '%d.%m.%Y %H:%M:%S')
    end_date = datetime.strptime(config.get('SQL', 'EndDate'), '%d.%m.%Y %H:%M:%S')
    
    # Query
    time_col = config.get('SQL', 'TimeColName')
    table_name = config.get('SQL', 'TableName')
    
    query = text(f"""
        SELECT {time_col}, Close, Volume
        FROM {table_name}
        WHERE Timestamp BETWEEN :start AND :end
        ORDER BY Timestamp
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(
            query, conn,
            params={
                'start': int(start_date.timestamp() * 1000),
                'end': int(end_date.timestamp() * 1000)
            }
        )
    
    return df

def validate_config(config: ConfigDict) -> None:
    """Basis-Konfiguration validieren."""
    # Nur validieren wenn DEFAULT-Sektion existiert
    if not config.has_section('DEFAULT') and 'DEFAULT' not in config:
        print("ℹ️  No DEFAULT section found - skipping validation (minimal config mode)")
        return
    
    use_log = config.getboolean('DEFAULT', 'UseLogReturns')
    use_simple = config.getboolean('DEFAULT', 'UseSimpleReturns')
    
    if use_log == use_simple:
        raise ValueError("Exactly one of UseLogReturns or UseSimpleReturns must be True")

# ==================================================================================
# MAIN EXECUTION
# ==================================================================================

def main():
    """Hauptfunktion."""
    import sys
    
    print("🚀 NoConfig Parameter Sweep System Started!")
    print("=" * 60)
    print("✨ This system runs completely standalone - NO MyConfig.ini required!")
    print("📋 All configuration is integrated directly in the code")
    print("=" * 60)
    
    # System-Modus anzeigen
    print(f"🎛️  System Mode: {SYSTEM_MODE}")
    if SYSTEM_MODE == "config":
        print("📄 Config Mode: Using ALL integrated config values (like MyConfig.ini)")
        print("🔄 Will run ONE analysis with complete configuration")
    elif SYSTEM_MODE == "sweep":
        print("🔄 Sweep Mode: Using MINIMAL config + parameter combinations")
        print(f"📊 Sweep Mode: {SWEEP_MODE}")
        print(f"🔀 Shuffle: {SHUFFLE_COMBINATIONS}")
    else:
        print(f"❌ Invalid SYSTEM_MODE: {SYSTEM_MODE}")
        print("📋 Valid modes: 'config' or 'sweep'")
        print("💡 Change SYSTEM_MODE variable at the top of the file")
        return
    
    print("=" * 60)
    
    # Konfiguration erstellen basierend auf Modus
    use_minimal = (SYSTEM_MODE == "sweep")
    config = ConfigDict(use_minimal=use_minimal)
    
    # Validierung (nur im Config-Modus nötig)
    if SYSTEM_MODE == "config":
        validate_config(config)
    
    # Daten laden
    print("📊 Loading data from integrated configuration...")
    df = load_data(config)
    print(f"✅ Loaded {len(df)} rows")
    
    # Runner initialisieren
    runner = AnalysisRunner(use_minimal_config=use_minimal)
    
    if SYSTEM_MODE == "config":
        # CONFIG-MODUS: Einzelne Analyse mit vollständiger integrierter Konfiguration
        print("🎯 Running single analysis with integrated config...")
        result = runner.run_single_analysis({}, df)
        status = "✅" if result['status'] == 'success' else "❌"
        print(f"{status} Analysis completed with integrated configuration")
        print("📋 All MyConfig.ini values were used from integrated config")
        
    elif SYSTEM_MODE == "sweep":
        # SWEEP-MODUS: Parameter-Sweeps mit minimaler Konfiguration
        print(f"🎯 Running parameter sweep - Mode: {SWEEP_MODE}")
        
        # Verfügbare Modi anzeigen
        available_modes = ["boolean", "numerical", "bins", "time_granularity", "full"]
        
        # Modus aus Kommandozeile oder SWEEP_MODE verwenden
        if len(sys.argv) > 1:
            sweep_mode = sys.argv[1]
            if sweep_mode not in available_modes:
                print(f"❌ Invalid sweep mode: {sweep_mode}")
                print(f"📋 Available modes: {', '.join(available_modes)}")
                print(f"💡 Usage: python {sys.argv[0]} [sweep_mode]")
                return
        else:
            sweep_mode = SWEEP_MODE  # Aus dem Programmkopf
        
        print(f"📋 Available sweep modes: {', '.join(available_modes)}")
        print(f"💡 Change sweep mode in code: Edit SWEEP_MODE variable")
        
        # Parameter Sweep durchführen
        results = runner.run_parameter_sweep(df, sweep_mode)
        
        # Zusammenfassung
        successful = len([r for r in results if r['status'] == 'success'])
        total = len(results)
        success_rate = successful / total * 100 if total > 0 else 0
        
        print(f"\n🎉 Parameter Sweep completed!")
        print(f"📈 {successful}/{total} runs successful")
        print(f"📊 Success rate: {success_rate:.1f}%")
        print(f"✨ Parameter sweeps completed with minimal configuration!")
        print(f"🔧 Only CSV/SQL config values were used, parameters were swept")
    
    print("✨ All done without any external configuration files!")

if __name__ == "__main__":
    main()
