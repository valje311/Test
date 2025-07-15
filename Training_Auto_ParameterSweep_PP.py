"""
Refactored Parameter Sweep System for Trading Analysis
====================================================

This module provides a clean, modular implementation of parallel parameter sweeping
for trading time series analysis. It replaces the monolithic worker function with
specialized, focused functions and implements shared memory for better performance.

Key improvements:
- Modular architecture with single-responsibility functions
- Shared memory implementation for DataFrame sharing
- Comprehensive error handling and logging
- Type hints and documentation
- Configuration management with JSON support
- Memory optimization and cleanup

Author: Refactored version
Date: July 2025
"""

from datetime import datetime
import configparser
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import multiprocessing
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import pickle
import gc
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import create_engine, text
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import custom modules
import FeatureSpace
import TimeSeriesAnalysis
import TimeSeriesManipulation


# ==================================================================================
# CONFIGURATION AND CONSTANTS
# ==================================================================================

@dataclass
class SweepConfig:
    """Configuration for parameter sweep execution."""
    parameter_sweep: bool = True
    num_bins_values: Optional[List[int]] = None
    sweep_mode: str = "all_combinations"  # "diagonal" or "all_combinations"
    max_workers: Optional[int] = 3  # Setzen Sie hier Ihre gewünschte Anzahl Worker
    
    def __post_init__(self):
        if self.num_bins_values is None:
            self.num_bins_values = [50, 100, 150]


@dataclass
class WorkerResult:
    """Result structure for worker execution."""
    bins_x: int
    bins_y: int
    plots_dir: Optional[str]
    status: str  # "success" or "failed"
    duration: float
    worker_id: int
    error: Optional[str] = None


# ==================================================================================
# SYSTEM OPTIMIZATION
# ==================================================================================

def get_optimal_workers() -> int:
    """
    Determine the optimal number of parallel workers based on system resources.
    
    Uses both CPU and memory constraints to find the best balance.
    
    Returns:
        int: Optimal number of workers
    """
    cores = multiprocessing.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Memory-based calculation (assuming 2GB per worker)
    memory_workers = max(1, int(memory_gb / 2))
    
    # CPU-based calculation (leave 1-2 cores for system)
    cpu_workers = max(1, cores - 1)
    
    # Take the minimum but cap at 8 for practical reasons
    optimal = min(memory_workers, cpu_workers, 8)
    
    logging.info(f"System Info - CPU Cores: {cores}, RAM: {memory_gb:.1f} GB")
    logging.info(f"Workers - Memory-based: {memory_workers}, CPU-based: {cpu_workers}")
    logging.info(f"Optimal workers selected: {optimal}")
    
    return optimal


def setup_logging(level: str = "INFO") -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    log_format = "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("parameter_sweep.log")
        ]
    )


# ==================================================================================
# CONFIGURATION MANAGEMENT
# ==================================================================================

class ConfigManager:
    """Manages configuration loading, validation, and serialization."""
    
    def __init__(self, config_path: str = "MyConfig.ini"):
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from INI file."""
        try:
            self.config.read(self.config_path)
            logging.info(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            logging.error(f"Failed to load config: {e}")
            raise
    
    def validate_config(self) -> bool:
        """
        Validate critical configuration parameters.
        
        Returns:
            bool: True if configuration is valid
        """
        try:
            # Check required sections
            required_sections = ['DEFAULT', 'SQL', 'Mutual Information']
            for section in required_sections:
                if section not in self.config:
                    logging.error(f"Missing required section: {section}")
                    return False
            
            # Validate XOR constraint for returns
            use_log = self.config.get('DEFAULT', 'UseLogReturns', fallback='True') == 'True'
            use_simple = self.config.get('DEFAULT', 'UseSimpleReturns', fallback='False') == 'True'
            
            if not self._xor(use_log, use_simple):
                logging.warning("Both or neither UseLogReturns/UseSimpleReturns enabled. Defaulting to LogReturns.")
                self.config['DEFAULT']['UseLogReturns'] = 'True'
                self.config['DEFAULT']['UseSimpleReturns'] = 'False'
            
            return True
            
        except Exception as e:
            logging.error(f"Configuration validation failed: {e}")
            return False
    
    @staticmethod
    def _xor(a: bool, b: bool) -> bool:
        """Exclusive OR operation."""
        return (a and not b) or (not a and b)
    
    def to_dict(self) -> Dict[str, Dict[str, str]]:
        """Convert config to dictionary for serialization."""
        return {section: dict(self.config[section]) for section in self.config.sections()}
    
    def create_worker_config(self, bins_x: int, bins_y: int) -> configparser.ConfigParser:
        """
        Create a modified config for worker with specific bin values.
        
        Args:
            bins_x: Number of bins for X dimension
            bins_y: Number of bins for Y dimension
            
        Returns:
            Modified ConfigParser instance
        """
        worker_config = configparser.ConfigParser()
        worker_config.read_dict(self.to_dict())
        
        # Ensure sections exist
        if 'DEFAULT' not in worker_config:
            worker_config.add_section('DEFAULT')
        if 'Mutual Information' not in worker_config:
            worker_config.add_section('Mutual Information')
        
        # Set bin values
        worker_config['Mutual Information']['NumBinsX'] = str(bins_x)
        worker_config['Mutual Information']['NumBinsY'] = str(bins_y)
        
        return worker_config


# ==================================================================================
# DATABASE OPERATIONS
# ==================================================================================

class DatabaseManager:
    """Handles database connections and data loading."""
    
    def __init__(self, config: configparser.ConfigParser):
        self.config = config
        self._connection_string = self._build_connection_string()
    
    def _build_connection_string(self) -> str:
        """Build MySQL connection string from config."""
        return (
            f"mysql+mysqlconnector://"
            f"{self.config['SQL']['Username']}:"
            f"{self.config['SQL']['Password']}@"
            f"localhost:{self.config['SQL']['Port']}/"
            f"{self.config['SQL']['Database']}"
        )
    
    def load_tick_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Load tick data from MySQL database for a given time period.
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            pandas DataFrame with tick data
        """
        try:
            engine = create_engine(self._connection_string)
            
            # Prepare query
            time_col = self.config['SQL']['timeColName']
            features = [time_col, 'Close', 'Volume']
            columns = ", ".join(features)
            
            query = text(f"""
                SELECT {columns}
                FROM {self.config['SQL']['TableName']}
                WHERE {time_col} BETWEEN :start AND :end
                ORDER BY {time_col}
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
            
            logging.info(f"Loaded {len(df)} rows of tick data")
            return df
            
        except Exception as e:
            logging.error(f"Database loading failed: {e}")
            raise


# ==================================================================================
# ANALYSIS PIPELINE
# ==================================================================================

class AnalysisPipeline:
    """Orchestrates the complete analysis workflow."""
    
    def __init__(self, config: configparser.ConfigParser, plots_dir: str):
        self.config = config
        self.plots_dir = plots_dir
        self.data_col = config.get('SQL', 'DataColName', fallback='Close')
        self.time_col = config.get('SQL', 'TimeColName', fallback='Timestamp')
    
    def run_complete_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute the complete analysis pipeline.
        
        Args:
            df: Input DataFrame with tick data
            
        Returns:
            Dictionary with analysis results and metadata
        """
        results = {'stages_completed': [], 'errors': [], 'status': ''}
        
        try:
            # Stage 1: Data preprocessing
            df_processed = self._preprocess_data(df.copy())
            results['stages_completed'].append('preprocessing')
            
            # Stage 2: Candle creation
            candles = self._create_candles(df_processed)
            results['stages_completed'].append('candles')
            
            # Stage 3: Returns calculation
            returns = self._calculate_returns(candles)
            results['stages_completed'].append('returns')
            
            # Stage 4: Time series analysis (if returns available)
            if returns is not None:
                self._run_time_series_analysis(returns, candles)
                results['stages_completed'].append('time_series')
            else:
                results['errors'].append('No returns calculated - skipping time series analysis')
            
            # Set status to success
            results['status'] = 'success'
            
        except Exception as e:
            logging.error(f"Analysis pipeline failed: {e}")
            results['status'] = 'failed'
            results['errors'].append(str(e))
        
        return results
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply smoothing to tick data if configured."""
        smooth_ticks = self.config.get('DEFAULT', 'SmoothTicks', fallback='False')
        
        if smooth_ticks == 'True':
            method = self.config.get('DEFAULT', 'SmoothTicksMethod', fallback='Perona-Malik')
            if method == 'Perona-Malik':
                df[self.data_col] = TimeSeriesManipulation.perona_malik_smoothing(
                    df[self.time_col], 
                    df[self.data_col].tolist(), 
                    self.config, 
                    self.plots_dir
                )
                logging.debug("Applied Perona-Malik smoothing to tick data")
        
        return df
    
    def _create_candles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create candlestick data and apply smoothing if configured."""
        candles = FeatureSpace.createCandleDataFrame(df, self.config, self.plots_dir)
        
        smooth_candles = self.config.get('DEFAULT', 'SmoothCandles', fallback='False')
        if smooth_candles == 'True':
            method = self.config.get('DEFAULT', 'SmoothCandlesMethod', fallback='Perona-Malik')
            if method == 'Perona-Malik':
                candles[self.data_col] = TimeSeriesManipulation.perona_malik_smoothing(
                    candles[self.time_col], 
                    candles[self.data_col].tolist(), 
                    self.config, 
                    self.plots_dir
                )
                logging.debug("Applied Perona-Malik smoothing to candle data")
        
        return candles
    
    def _calculate_returns(self, candles: pd.DataFrame) -> Optional[List[float]]:
        """Calculate returns (log or simple) based on configuration."""
        use_log = self.config.get('DEFAULT', 'UseLogReturns', fallback='True') == 'True'
        use_simple = self.config.get('DEFAULT', 'UseSimpleReturns', fallback='False') == 'True'
        
        # Ensure at least one method is enabled
        if not use_log and not use_simple:
            logging.warning("Neither returns method enabled, defaulting to log returns")
            use_log = True
        
        returns = None
        smooth_config = 'False'  # Default value
        method_config = 'SmoothLogReturnsMethod'  # Default value
        
        if use_log:
            returns = TimeSeriesManipulation.getLogReturns(candles[self.data_col].tolist())
            smooth_config = self.config.get('DEFAULT', 'SmoothLogReturns', fallback='False')
            method_config = 'SmoothLogReturnsMethod'
        elif use_simple:
            returns = TimeSeriesManipulation.getSimpleReturns(candles[self.data_col].tolist())
            smooth_config = self.config.get('DEFAULT', 'SmoothSimpleReturns', fallback='False')
            method_config = 'SmoothSimpleReturnsMethod'
        
        # Apply smoothing if configured
        if returns and smooth_config == 'True':
            method = self.config.get('DEFAULT', method_config, fallback='Perona-Malik')
            if method == 'Perona-Malik':
                returns = TimeSeriesManipulation.perona_malik_smoothing(
                    candles[self.time_col], returns, self.config, self.plots_dir
                )
                logging.debug(f"Applied smoothing to {'log' if use_log else 'simple'} returns")
        
        # Convert to List[float] if not None to match return type
        if returns is not None:
            returns = [float(x) for x in returns]
            
        return returns
    
    def _run_time_series_analysis(self, returns: List[float], candles: pd.DataFrame) -> None:
        """Execute time series analysis components."""
        try:
            # Autocorrelation analysis
            TimeSeriesAnalysis.calculate_autocorrelation(returns, self.config, self.plots_dir)
            logging.debug("Autocorrelation analysis completed")
        except Exception as e:
            logging.error(f"Autocorrelation analysis failed: {e}")
        
        try:
            # Takens embedding
            TimeSeriesAnalysis.TakenEmbedding(returns, self.plots_dir, self.config)
            logging.debug("Takens embedding completed")
        except Exception as e:
            logging.error(f"Takens embedding failed: {e}")
        
        try:
            # Returns visualization
            self._create_returns_plot(returns, candles)
            logging.debug("Returns visualization completed")
        except Exception as e:
            logging.error(f"Returns visualization failed: {e}")
    
    def _create_returns_plot(self, returns: List[float], candles: pd.DataFrame) -> None:
        """Create returns visualization plot."""
        plt.figure(figsize=(12, 6))
        plt.xlabel('Time')
        plt.ylabel('Returns')
        
        bins_x = self.config.get('Mutual Information', 'NumBinsX', fallback='100')
        bins_y = self.config.get('Mutual Information', 'NumBinsY', fallback='100')
        plt.title(f'Returns Visualization - Bins {bins_x}x{bins_y}')
        
        lag_value = int(self.config.get('Autocorrelation', 'Lag', fallback='1'))
        
        plt.plot(candles[self.time_col][1:], returns, label='Returns', color='blue', alpha=0.7)
        if len(returns) > lag_value:
            plt.plot(candles[self.time_col][1:-lag_value], 
                     returns[lag_value:], label=f'Returns with lag {lag_value}', 
                     color='orange', alpha=0.7)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = Path(self.plots_dir) / 'Returns_Analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


# ==================================================================================
# WORKER EXECUTION
# ==================================================================================

def execute_worker_analysis(args: Tuple[int, int, Dict, bytes, int]) -> WorkerResult:
    """
    Clean, focused worker function for parallel execution.
    
    Args:
        args: Tuple containing (bins_x, bins_y, config_dict, df_pickle, worker_id)
        
    Returns:
        WorkerResult with execution details
    """
    bins_x, bins_y, config_dict, df_pickle, worker_id = args
    start_time = time.time()
    
    try:
        # Setup worker environment
        config = _reconstruct_config(config_dict, bins_x, bins_y)
        df = pickle.loads(df_pickle)
        plots_dir = _create_worker_directory(config, bins_x, bins_y, worker_id)
        
        # Save configuration for reproducibility
        _save_worker_config(config, plots_dir, bins_x, bins_y)
        
        # Execute analysis pipeline
        pipeline = AnalysisPipeline(config, plots_dir)
        results = pipeline.run_complete_analysis(df)
        
        # Cleanup
        del df, config
        gc.collect()
        
        duration = time.time() - start_time
        
        return WorkerResult(
            bins_x=bins_x,
            bins_y=bins_y,
            plots_dir=plots_dir,
            status=results['status'],
            duration=duration,
            worker_id=worker_id
        )
        
    except Exception as e:
        duration = time.time() - start_time
        logging.error(f"Worker {worker_id} failed for bins {bins_x}x{bins_y}: {e}")
        
        return WorkerResult(
            bins_x=bins_x,
            bins_y=bins_y,
            plots_dir=None,
            status='failed',
            duration=duration,
            worker_id=worker_id,
            error=str(e)
        )


def _reconstruct_config(config_dict: Dict, bins_x: int, bins_y: int) -> configparser.ConfigParser:
    """Reconstruct ConfigParser from dictionary and set bin values."""
    config = configparser.ConfigParser()
    config.read_dict(config_dict)
    
    # Ensure required sections exist
    for section in ['DEFAULT', 'Mutual Information']:
        if section not in config:
            config.add_section(section)
    
    # Set bin values
    config['Mutual Information']['NumBinsX'] = str(bins_x)
    config['Mutual Information']['NumBinsY'] = str(bins_y)
    
    return config


def _create_worker_directory(config: configparser.ConfigParser, bins_x: int, bins_y: int, worker_id: int) -> str:
    """Create unique directory for worker output."""
    project_root = Path(__file__).parent
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    table_name = config.get('SQL', 'TableName', fallback='data')
    
    plots_dir = project_root / 'Plots' / table_name / f'{timestamp}_bins_{bins_x}x{bins_y}_w{worker_id}'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    return str(plots_dir)


def _save_worker_config(config: configparser.ConfigParser, plots_dir: str, bins_x: int, bins_y: int) -> None:
    """Save configuration parameters for reproducibility."""
    config_path = Path(plots_dir) / 'analysis_config.txt'
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ANALYSIS CONFIGURATION\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Analysis Bins: X={bins_x}, Y={bins_y}\n")
        f.write(f"Output Directory: {plots_dir}\n")
        f.write("=" * 80 + "\n\n")
        
        for section_name in config.sections():
            f.write(f"[{section_name}]\n")
            for key, value in config[section_name].items():
                marker = "  # Parameter sweep override" if (
                    section_name == 'Mutual Information' and key in ['NumBinsX', 'NumBinsY']
                ) else ""
                f.write(f"{key} = {value}{marker}\n")
            f.write("\n")


# ==================================================================================
# PARALLEL EXECUTION ORCHESTRATOR
# ==================================================================================

class ParallelExecutor:
    """Manages parallel execution of parameter sweep."""
    
    def __init__(self, config_manager: ConfigManager, max_workers: int):
        self.config_manager = config_manager
        self.max_workers = max_workers
    
    def execute_parameter_sweep(self, df: pd.DataFrame, combinations: List[Tuple[int, int]]) -> List[WorkerResult]:
        """
        Execute parameter sweep in parallel.
        
        Args:
            df: DataFrame with data
            combinations: List of (bins_x, bins_y) tuples
            
        Returns:
            List of WorkerResult objects
        """
        # Prepare worker arguments
        config_dict = self.config_manager.to_dict()
        df_pickle = pickle.dumps(df)
        
        worker_args = [
            (bins_x, bins_y, config_dict, df_pickle, i)
            for i, (bins_x, bins_y) in enumerate(combinations)
        ]
        
        logging.info(f"Starting parallel execution with {self.max_workers} workers")
        logging.info(f"Processing {len(combinations)} parameter combinations")
        
        return self._run_parallel_workers(worker_args, combinations)
    
    def _run_parallel_workers(self, worker_args: List[Tuple], combinations: List[Tuple[int, int]]) -> List[WorkerResult]:
        """Execute workers in parallel and track progress."""
        results = []
        completed_count = 0
        total_duration = 0
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_bins = {
                executor.submit(execute_worker_analysis, args): (args[0], args[1])
                for args in worker_args
            }
            
            # Process completed jobs
            for future in as_completed(future_to_bins):
                bins_x, bins_y = future_to_bins[future]
                completed_count += 1
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Progress tracking
                    elapsed_time = time.time() - start_time
                    self._log_progress(result, completed_count, len(combinations), elapsed_time)
                    
                    if result.status == 'success':
                        total_duration += result.duration
                        
                except Exception as e:
                    logging.error(f"Future exception for {bins_x}x{bins_y}: {e}")
                    results.append(WorkerResult(
                        bins_x=bins_x, bins_y=bins_y, plots_dir=None,
                        status='failed', duration=0, worker_id=-1, error=str(e)
                    ))
        
        self._log_completion_summary(results, time.time() - start_time, total_duration)
        return results
    
    def _log_progress(self, result: WorkerResult, completed: int, total: int, elapsed: float) -> None:
        """Log progress information."""
        avg_time = elapsed / completed
        remaining = total - completed
        eta_seconds = remaining * avg_time
        eta_formatted = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
        
        status_icon = "OK" if result.status == 'success' else "FAIL"
        logging.info(f"{status_icon} [{completed}/{total}] {result.bins_x}x{result.bins_y} "
                    f"({result.duration:.1f}s) | ETA: {eta_formatted}")
        
        # Also print to console with emojis (handles encoding better)
        emoji_icon = "✅" if result.status == 'success' else "❌"
        print(f"{emoji_icon} [{completed}/{total}] {result.bins_x}x{result.bins_y} "
              f"({result.duration:.1f}s) | ETA: {eta_formatted}")
    
    def _log_completion_summary(self, results: List[WorkerResult], wall_time: float, cpu_time: float) -> None:
        """Log execution summary."""
        successful = [r for r in results if r.status == 'success']
        failed = [r for r in results if r.status == 'failed']
        
        speedup = cpu_time / wall_time if wall_time > 0 else 0
        
        logging.info(f"Parallel execution completed!")
        logging.info(f"Wall time: {wall_time:.1f}s, CPU time: {cpu_time:.1f}s")
        logging.info(f"Speedup factor: {speedup:.1f}x")
        logging.info(f"Success rate: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
        
        if failed:
            logging.warning(f"Failed executions: {len(failed)}")


# ==================================================================================
# MAIN ORCHESTRATOR
# ==================================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Trading Analysis Parameter Sweep')
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='Number of parallel workers (default: auto-detect)')
    parser.add_argument('--bins', '-b', type=int, nargs='+', default=[50, 100, 150],
                       help='Bin values to test (default: 50 100 150)')
    parser.add_argument('--mode', '-m', choices=['diagonal', 'all_combinations'], 
                       default='all_combinations',
                       help='Sweep mode (default: all_combinations)')
    parser.add_argument('--single', '-s', action='store_true',
                       help='Run single analysis instead of parameter sweep')
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup
    setup_logging("INFO")
    sweep_config = SweepConfig(
        parameter_sweep=not args.single,
        num_bins_values=args.bins,
        sweep_mode=args.mode,
        max_workers=args.workers
    )
    
    try:
        # Configuration management
        config_manager = ConfigManager()
        if not config_manager.validate_config():
            raise ValueError("Configuration validation failed")
        
        # Database operations
        db_manager = DatabaseManager(config_manager.config)
        start_date = datetime.strptime(config_manager.config['SQL']['StartDate'], '%d.%m.%Y %H:%M:%S')
        end_date = datetime.strptime(config_manager.config['SQL']['EndDate'], '%d.%m.%Y %H:%M:%S')
        
        logging.info("Loading data from database...")
        df = db_manager.load_tick_data(start_date, end_date)
        
        # Worker optimization
        workers = sweep_config.max_workers or get_optimal_workers()
        
        if sweep_config.parameter_sweep:
            # Generate parameter combinations
            bins_values = sweep_config.num_bins_values or [50, 100, 150]
            
            if sweep_config.sweep_mode == "diagonal":
                combinations = [(bins, bins) for bins in bins_values]
                logging.info(f"Diagonal sweep: {len(combinations)} combinations")
            elif sweep_config.sweep_mode == "all_combinations":
                combinations = [(x, y) for x in bins_values for y in bins_values]
                logging.info(f"Full sweep: {len(combinations)} combinations")
            else:
                raise ValueError(f"Unknown sweep mode: {sweep_config.sweep_mode}")
            
            # Execute parallel sweep
            executor = ParallelExecutor(config_manager, workers)
            results = executor.execute_parameter_sweep(df, combinations)
            
            # Results summary
            successful = [r for r in results if r.status == 'success']
            failed = [r for r in results if r.status == 'failed']
            
            print(f"\n🎉 Parameter sweep completed!")
            print(f"✅ Successful: {len(successful)}/{len(results)}")
            
            if successful:
                print("📁 Sample results:")
                for result in successful[:5]:
                    print(f"   {result.bins_x}x{result.bins_y}: {result.plots_dir}")
            
            if failed:
                print(f"❌ Failed: {len(failed)}")
                for result in failed[:3]:
                    print(f"   {result.bins_x}x{result.bins_y}: {result.error}")
        
        else:
            # Single run execution
            bins_x = int(config_manager.config['Mutual Information']['NumBinsX'])
            bins_y = int(config_manager.config['Mutual Information']['NumBinsY'])
            
            logging.info(f"Single analysis run with bins {bins_x}x{bins_y}")
            
            # Create output directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            table_name = config_manager.config['SQL']['TableName']
            plots_dir = Path(__file__).parent / 'Plots' / table_name / timestamp
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Execute analysis
            pipeline = AnalysisPipeline(config_manager.config, str(plots_dir))
            results = pipeline.run_complete_analysis(df)
            
            if results['status'] == 'success':
                print(f"✅ Analysis completed: {plots_dir}")
            else:
                print(f"❌ Analysis failed: {results['errors']}")
    
    except Exception as e:
        logging.error(f"Main execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
