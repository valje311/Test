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

# Parameter sweep configuration
PARAMETER_SWEEP = True  # Set to False for single run
NUM_BINS_VALUES = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]  # Different bin values to test
SWEEP_MODE = "all_combinations"  # Options: "diagonal", "all_combinations"

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
    CONNECTION_STRING = f"mysql+mysqlconnector://{config['SQL']['Username']}:{config['SQL']['Password']}@localhost:{config['SQL']['Port']}/{config['SQL']['Database']}"
    engine = create_engine(CONNECTION_STRING)
    
    # Prepare query
    features = [config['SQL']['timeColName'], 'Close', 'Volume']
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

def save_config_to_file(config, plots_dir, num_bins_x, num_bins_y):
    """
    Save all configuration parameters to a text file for reproducibility.
    
    Args:
        config: ConfigParser object with all settings
        plots_dir: Directory where the config file should be saved
        num_bins_x: NumBinsX value used for this run
        num_bins_y: NumBinsY value used for this run
    """
    config_file_path = os.path.join(plots_dir, 'config_parameters.txt')
    
    with open(config_file_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("CONFIGURATION PARAMETERS FOR THIS ANALYSIS RUN\n")
        f.write("="*60 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n")
        f.write(f"Analysis Bins: NumBinsX={num_bins_x}, NumBinsY={num_bins_y}\n")
        f.write("="*60 + "\n\n")
        
        # Iterate through all sections and parameters
        for section_name in config.sections():
            f.write(f"[{section_name}]\n")
            for key, value in config[section_name].items():
                # Highlight if this parameter was modified for the sweep
                if section_name == 'Mutual Information' and key in ['NumBinsX', 'NumBinsY']:
                    if key == 'NumBinsX':
                        f.write(f"{key} = {num_bins_x}  # Modified for parameter sweep\n")
                    elif key == 'NumBinsY':
                        f.write(f"{key} = {num_bins_y}  # Modified for parameter sweep\n")
                else:
                    f.write(f"{key} = {value}\n")
            f.write("\n")
        
        # Add some analysis metadata
        f.write("="*60 + "\n")
        f.write("ANALYSIS METADATA\n")
        f.write("="*60 + "\n")
        f.write(f"Plots Directory: {plots_dir}\n")
        f.write(f"Script Used: Training_Auto_ParameterSweep.py\n")
        f.write(f"Parameter Sweep Mode: {'Yes' if PARAMETER_SWEEP else 'No'}\n")
        if PARAMETER_SWEEP:
            f.write(f"Sweep Type: {SWEEP_MODE}\n")
            f.write(f"All Tested Bin Values: {NUM_BINS_VALUES}\n")
            if SWEEP_MODE == "all_combinations":
                f.write(f"Total Combinations: {len(NUM_BINS_VALUES)} × {len(NUM_BINS_VALUES)} = {len(NUM_BINS_VALUES)**2}\n")
        f.write("="*60 + "\n")
    
    print(f"📄 Configuration saved to: {config_file_path}")

def run_analysis_with_bins(num_bins_x, num_bins_y, config, df_original):
    """Run the complete analysis with specific bin values"""
    # Make a copy of config to avoid modifying the original
    import copy
    config_copy = copy.deepcopy(config)
    df = df_original.copy()
    
    # Modify bin values
    config_copy['Mutual Information']['NumBinsX'] = str(num_bins_x)
    config_copy['Mutual Information']['NumBinsY'] = str(num_bins_y)
    
    print(f"\n🔬 Running analysis with NumBinsX={num_bins_x}, NumBinsY={num_bins_y}")
    
    # Create timestamped folder with bin info
    project_root = os.path.dirname(os.path.abspath(__file__))
    training_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plots_dir = os.path.join(project_root, 'Plots', config_copy['SQL']['TableName'], 
                            f'{training_timestamp}_bins_{num_bins_x}x{num_bins_y}')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save configuration parameters to file for reproducibility
    save_config_to_file(config_copy, plots_dir, num_bins_x, num_bins_y)
    
    # Run the analysis
    if config_copy['DEFAULT']['SmoothTicks'] == 'True':
        if config_copy['DEFAULT']['SmoothTicksMethod'] == 'Perona-Malik':
            print("Smoothing data using Perona-Malik method...")
            df[config_copy['SQL']['DataColName']] = TimeSeriesManipulation.perona_malik_smoothing(
                df[config_copy['SQL']['TimeColName']], 
                df[config_copy['SQL']['DataColName']].tolist(), 
                config_copy, plots_dir)

    candles = FeatureSpace.createCandleDataFrame(df, config_copy, plots_dir)
    if config_copy['DEFAULT']['SmoothCandles'] == 'True':
        if config_copy['DEFAULT']['SmoothCandlesMethod'] == 'Perona-Malik':
            print("Smoothing candles using Perona-Malik method...")
            candles[config_copy['SQL']['DataColName']] = TimeSeriesManipulation.perona_malik_smoothing(
                candles[config_copy['SQL']['TimeColName']], 
                candles[config_copy['SQL']['DataColName']].tolist(), 
                config_copy, plots_dir)

    myReturns = None
    if config_copy['DEFAULT']['UseLogReturns'] == 'True':
        myReturns = TimeSeriesManipulation.getLogReturns(candles[config_copy['SQL']['DataColName']].tolist())
        if config_copy['DEFAULT']['SmoothLogReturns'] == 'True':
            if config_copy['DEFAULT']['SmoothLogReturnsMethod'] == 'Perona-Malik':
                print("Smoothing log returns using Perona-Malik method...")
                myReturns = TimeSeriesManipulation.perona_malik_smoothing(
                    candles[config_copy['SQL']['TimeColName']], myReturns, config_copy, plots_dir)
    elif config_copy['DEFAULT']['UseSimpleReturns'] == 'True':
        myReturns = TimeSeriesManipulation.getSimpleReturns(candles[config_copy['SQL']['DataColName']].tolist())
        if config_copy['DEFAULT']['SmoothSimpleReturns'] == 'True':
            if config_copy['DEFAULT']['SmoothSimpleReturnsMethod'] == 'Perona-Malik':
                print("Smoothing simple returns using Perona-Malik method...")
                myReturns = TimeSeriesManipulation.perona_malik_smoothing(
                    candles[config_copy['SQL']['TimeColName']], myReturns, config_copy, plots_dir)

    # Ensure myReturns is not None before proceeding
    if myReturns is not None:
        calculate_autocorr = TimeSeriesAnalysis.calculate_autocorrelation(myReturns, config_copy, plots_dir)
        TimeSeriesAnalysis.TakenEmbedding(myReturns, plots_dir, config_copy)

        plt.figure()
        plt.xlabel('Time')
        plt.ylabel('log Returns')
        plt.title(f'Lag visualization - Bins {num_bins_x}x{num_bins_y}')
        plt.plot(candles[config_copy['SQL']['TimeColName']][1:], myReturns, label='Log Returns', color='blue')
        plt.plot(candles[config_copy['SQL']['TimeColName']][1:-int(config_copy['Autocorrelation']['Lag'])], 
                 myReturns[int(config_copy['Autocorrelation']['Lag']):], label='Log Returns with lag', color='orange')
        plt.savefig(os.path.join(plots_dir, 'LogReturns.png'))
        plt.close()
    else:
        print("Warning: myReturns is None - skipping time series analysis.")
    
    return plots_dir

def main():
    config = configparser.ConfigParser()
    config.read('MyConfig.ini')

    assert xor(config['DEFAULT']['UseLogReturns'] =='True', config['DEFAULT']['UseSimpleReturns'] == 'True'), "Exactly one of UseLogReturns or UseSimpleReturns must be True."
    
    startDateTime = datetime.strptime(config['SQL']['StartDate'], '%d.%m.%Y %H:%M:%S')
    endDateTime = datetime.strptime(config['SQL']['EndDate'], '%d.%m.%Y %H:%M:%S')
    
    print("Loading data from database...")
    df = load_tick_data(startDateTime, endDateTime, config)
    print("Data loaded successfully.")

    # Parameter sweep or single run
    if PARAMETER_SWEEP:
        if SWEEP_MODE == "diagonal":
            # Original behavior: NumBinsX = NumBinsY
            print(f"🔄 Starting diagonal parameter sweep with {len(NUM_BINS_VALUES)} bin values...")
            print("Mode: NumBinsX = NumBinsY")
            results = []
            
            for i, num_bins in enumerate(NUM_BINS_VALUES, 1):
                try:
                    print(f"\n📊 Progress: {i}/{len(NUM_BINS_VALUES)}")
                    plots_dir = run_analysis_with_bins(num_bins, num_bins, config, df)
                    results.append({
                        'bins_x': num_bins,
                        'bins_y': num_bins,
                        'plots_dir': plots_dir,
                        'status': 'success'
                    })
                    print(f"✅ Completed analysis for bins {num_bins}x{num_bins}")
                    print(f"📁 Results saved to: {plots_dir}")
                except Exception as e:
                    print(f"❌ Error with bins {num_bins}x{num_bins}: {e}")
                    results.append({
                        'bins_x': num_bins,
                        'bins_y': num_bins,
                        'plots_dir': None,
                        'status': 'failed',
                        'error': str(e)
                    })
                    continue
        
        elif SWEEP_MODE == "all_combinations":
            # All combinations: NumBinsX × NumBinsY
            total_combinations = len(NUM_BINS_VALUES) ** 2
            print(f"🔄 Starting full parameter sweep with {total_combinations} combinations...")
            print(f"Mode: All combinations of {len(NUM_BINS_VALUES)} × {len(NUM_BINS_VALUES)} values")
            print(f"NumBinsX values: {NUM_BINS_VALUES}")
            print(f"NumBinsY values: {NUM_BINS_VALUES}")
            results = []
            
            combination_count = 0
            for bins_x in NUM_BINS_VALUES:
                for bins_y in NUM_BINS_VALUES:
                    combination_count += 1
                    try:
                        print(f"\n📊 Progress: {combination_count}/{total_combinations}")
                        print(f"🔬 Testing combination: NumBinsX={bins_x}, NumBinsY={bins_y}")
                        plots_dir = run_analysis_with_bins(bins_x, bins_y, config, df)
                        results.append({
                            'bins_x': bins_x,
                            'bins_y': bins_y,
                            'plots_dir': plots_dir,
                            'status': 'success'
                        })
                        print(f"✅ Completed analysis for bins {bins_x}x{bins_y}")
                        print(f"📁 Results saved to: {plots_dir}")
                    except Exception as e:
                        print(f"❌ Error with bins {bins_x}x{bins_y}: {e}")
                        results.append({
                            'bins_x': bins_x,
                            'bins_y': bins_y,
                            'plots_dir': None,
                            'status': 'failed',
                            'error': str(e)
                        })
                        continue
        
        else:
            raise ValueError(f"Unknown SWEEP_MODE: {SWEEP_MODE}. Use 'diagonal' or 'all_combinations'")
        
        # Summary
        print("\n" + "="*60)
        print("🎉 Parameter sweep completed!")
        print("="*60)
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']
        
        total_runs = len(NUM_BINS_VALUES) if SWEEP_MODE == "diagonal" else len(NUM_BINS_VALUES) ** 2
        print(f"✅ Successful runs: {len(successful)}/{total_runs}")
        if successful:
            print("📁 Result directories:")
            for result in successful:
                print(f"   - Bins {result['bins_x']}x{result['bins_y']}: {result['plots_dir']}")
        
        if failed:
            print(f"\n❌ Failed runs: {len(failed)}")
            for result in failed:
                print(f"   - Bins {result['bins_x']}x{result['bins_y']}: {result['error']}")
    
    else:
        # Single run (original behavior)
        project_root = os.path.dirname(os.path.abspath(__file__))
        training_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plots_dir = os.path.join(project_root, 'Plots', config['SQL']['TableName'], training_timestamp)
        os.makedirs(plots_dir, exist_ok=True)
        
        # Save configuration for single run
        save_config_to_file(config, plots_dir, 
                           int(config['Mutual Information']['NumBinsX']), 
                           int(config['Mutual Information']['NumBinsY']))
        
        # Run single analysis...
        print("🔬 Running single analysis with config values...")
        run_analysis_with_bins(
            int(config['Mutual Information']['NumBinsX']), 
            int(config['Mutual Information']['NumBinsY']), 
            config, df
        )

if __name__ == "__main__":
    main()
