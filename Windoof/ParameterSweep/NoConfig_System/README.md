# NoConfig Parameter Sweep System

Ein eigenständiges Trading-Analyse System mit **zwei Betriebsmodi** - komplett ohne externe MyConfig.ini!

## 🎛️ **Neue Schalter-Funktionalität**

Das System hat jetzt einen **intelligenten Modus-Schalter** mit zwei völlig getrennten Betriebsarten:

### 📄 **Config-Modus** (`SYSTEM_MODE = "config"`)
- Verwendet **ALLE** ursprünglichen MyConfig.ini Werte
- Führt **EINEN** kompletten Durchlauf durch  
- Identisch zur ursprünglichen MyConfig.ini Funktionalität

### 🔄 **Sweep-Modus** (`SYSTEM_MODE = "sweep"`)
- Verwendet **NUR** minimale CSV/SQL-Konfiguration
- Parameter-Sweeps sind **komplett getrennt** von anderen Config-Werten
- Systematische Parameter-Tests ohne Beeinflussung

## 🚀 Features

- **Vollständig eigenständig**: Keine externe Konfigurationsdatei erforderlich
- **Parameter Sweep**: Systematische Tests verschiedener Parameter-Kombinationen
- **Alle ursprünglichen Funktionen**: Identische Analyse-Funktionalität wie das Original-System
- **Integrierte Konfiguration**: Alle MyConfig.ini Werte direkt im Code
- **Modi verfügbar**: `boolean`, `numerical`, `bins`, `time_granularity`, `full`, `single`

## 📁 Dateien im System

```
NoConfig_System/
├── Training_Auto_ParameterSweep_NoConfig.py    # Hauptskript (STARTEN!)
├── FeatureSpace_NoConfig.py                    # Candlestick-Analyse
├── TimeSeriesAnalysis_NoConfig.py              # Zeitreihen-Analyse
├── TimeSeriesManipulation_NoConfig.py          # Daten-Manipulation
└── README.md                                   # Diese Datei
```

## 🎯 Verwendung

### 🎛️ **Modus-Schalter setzen (im Code):**
```python
# In Training_Auto_ParameterSweep_NoConfig.py (Zeile ~25):
SYSTEM_MODE = "config"    # Für MyConfig.ini Simulation (ALLE Werte)
SYSTEM_MODE = "sweep"     # Für Parameter-Sweeps (nur CSV/SQL)
```

### 📄 **Config-Modus - Wie MyConfig.ini:**
```bash
python Training_Auto_ParameterSweep_NoConfig.py
# Führt EINEN Durchlauf mit ALLEN integrierten MyConfig.ini Werten durch
```

### 🔄 **Sweep-Modus - Parameter-Tests:**
```bash
python Training_Auto_ParameterSweep_NoConfig.py           # Verwendet SWEEP_MODE aus Code
python Training_Auto_ParameterSweep_NoConfig.py full      # Vollständiger Parameter-Sweep
python Training_Auto_ParameterSweep_NoConfig.py boolean   # Nur Boolean-Parameter
python Training_Auto_ParameterSweep_NoConfig.py numerical # Nur numerische Parameter
```

## ⚙️ Konfiguration anpassen

Alle Einstellungen direkt im Code `Training_Auto_ParameterSweep_NoConfig.py`:

### 🎛️ **System-Schalter (Zeilen 25-35):**
```python
# HAUPT-SCHALTER - Wähle zwischen zwei Betriebsmodi:
SYSTEM_MODE = "config"      # "config" für MyConfig.ini Simulation
                           # "sweep" für Parameter-Sweeps

# Für SWEEP-Modus:
SWEEP_MODE = "full"             # Gewünschter Sweep-Modus  
SHUFFLE_COMBINATIONS = True     # Zufällige Reihenfolge
```

### 📄 **Vollständige Konfiguration (INTEGRATED_CONFIG)**
Für `SYSTEM_MODE = "config"` - **ALLE** MyConfig.ini Werte:
```python
INTEGRATED_CONFIG = {
    'DEFAULT': {
        'SmoothTicks': True,
        'UseLogReturns': True,
        # ... alle ursprünglichen Parameter
    },
    'SQL': {
        'UserName': 'Filler',
        'Password': '5gs.1-',
        'Database': 'trading',
        'TableName': 'oil',
        # ... komplette Datenbankverbindung
    },
    # ... alle Sektionen aus MyConfig.ini
}
```

### 🔄 **Minimale Konfiguration (MINIMAL_CONFIG)**  
Für `SYSTEM_MODE = "sweep"` - **NUR** essentiell für Parameter-Sweeps:
```python
MINIMAL_CONFIG = {
    'CSV': {
        'FilePath': r'D:\export_WTI_Tick_01012020_13052025.csv',
        'ChunkSize': 5000000
    },
    'SQL': {
        'UserName': 'Filler',
        'Password': '5gs.1-',
        # ... nur SQL-Zugang, KEINE anderen Parameter
    }
}
```

### Parameter-Sweep Einstellungen:
```python
BOOLEAN_PARAMS = {
    'DEFAULT': {
        'SmoothCandles': [False, True],  # Test beide Werte
        'UseLogReturns': [True],         # Nur ein Wert
        # ... weitere Parameter
    }
}
```

## 🔧 Anpassungen vornehmen

### 🎛️ **Modus wechseln:**
```python
# Zeile ~25 in Training_Auto_ParameterSweep_NoConfig.py ändern:
SYSTEM_MODE = "config"    # Für komplette MyConfig.ini Simulation
SYSTEM_MODE = "sweep"     # Für isolierte Parameter-Sweeps
```

### 📄 **Config-Modus anpassen:**
1. **Alle Parameter ändern**: `INTEGRATED_CONFIG` bearbeiten
2. **Identisch zu MyConfig.ini**: Alle Werte sind 1:1 übernommen

### 🔄 **Sweep-Modus anpassen:**
1. **Datenbankverbindung**: Nur `MINIMAL_CONFIG['SQL']` bearbeiten  
2. **Parameter-Tests erweitern**: `BOOLEAN_PARAMS`, `NUMERICAL_PARAMS` etc.
3. **Sweep-Modus ändern**: `SWEEP_MODE` Variable anpassen

### ⚠️ **Wichtige Trennung:**
- **Config-Modus**: Parameter aus `INTEGRATED_CONFIG` 
- **Sweep-Modus**: Parameter aus `*_PARAMS` Definitionen
- **KEINE Vermischung** zwischen beiden Modi!

## 📊 Ausgabe

Das System erstellt:
- **Plots-Ordner**: Alle Visualisierungen pro Run
- **ParameterSweepResults**: JSON-Ergebnisse und Zusammenfassungen
- **Config-Dateien**: Automatisch generierte INI-Dateien pro Run

## 🎉 Vorteile gegenüber Original

- ✅ **Keine externe Config-Datei** erforderlich
- ✅ **Einfachere Verteilung** - alle Einstellungen im Code
- ✅ **Bessere Versionskontrolle** - Config-Änderungen sind im Git sichtbar
- ✅ **Keine verlorenen Config-Dateien** mehr
- ✅ **Identische Funktionalität** wie das Original

## 🛠️ Abhängigkeiten

Gleiche Requirements wie das Original-System:
- pandas
- numpy  
- matplotlib
- sqlalchemy
- mysqlconnector
- mplfinance
- tqdm
- statsmodels

## 💡 Tipps

1. **Erste Tests**: Starte mit `single` Modus zum Testen
2. **Performance**: Bei vielen Kombinationen `MAX_COMBINATIONS` anpassen  
3. **Zufälligkeit**: `SHUFFLE_COMBINATIONS = False` für reproduzierbare Reihenfolge
4. **Parameter anpassen**: Direkt in den `*_PARAMS` Dictionaries

---

**Das System läuft komplett eigenständig - keine MyConfig.ini erforderlich!** 🎯
