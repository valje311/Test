"""
Plot Collection Organizer
========================

This script organizes plots from parameter sweep results into collections by plot type.
It extracts plots from timestamped folders and renames them with bin parameters.

Example:
- Source: 20250713_210133_bins_50x50/False_Nearest_Neighbors.png
- Target: PlotCollection/False_Nearest_Neighbors/False_Nearest_Neighbors_bins_50x50.png

Author: Created for Trading Analysis
Date: July 2025
"""

import os
import shutil
import re
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from datetime import datetime


# ==================================================================================
# CONFIGURATION
# ==================================================================================

# Plot types to collect
PLOT_TYPES = [
    'False_Nearest_Neighbors',
    'Mutual_Information', 
    'Phase_Space_2D_Reconstruction',
    'Phase_Space_3D_Reconstruction',
    'Autocorrelation',
    'candlestick_chart',
    'Returns_Analysis'
]

# Joint histogram pattern (Joint_Histogram1 to Joint_Histogram100)
JOINT_HISTOGRAM_PATTERN = r'Joint_Histogram\d+'

# Source and target directories
SOURCE_DIR = r"H:\Rave\WR_VL\Ohne Glättung"
TARGET_DIR = "PlotCollection"

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


# ==================================================================================
# UTILITY FUNCTIONS
# ==================================================================================

def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("plot_collector.log")
        ]
    )


def extract_bin_info(folder_name: str) -> str:
    """
    Extract bin information from folder name.
    
    Args:
        folder_name: Folder name like '20250713_210133_bins_50x50'
        
    Returns:
        Bin info like 'bins_50x50' or empty string if not found
    """
    # Pattern to match bins_XxY
    pattern = r'(bins_\d+x\d+)'
    match = re.search(pattern, folder_name)
    
    if match:
        return match.group(1)
    else:
        logging.warning(f"Could not extract bin info from folder: {folder_name}")
        return "unknown_bins"


def is_joint_histogram(filename: str) -> bool:
    """Check if filename matches Joint_Histogram pattern."""
    return bool(re.match(JOINT_HISTOGRAM_PATTERN, Path(filename).stem))


def get_plot_category(filename: str) -> str:
    """
    Determine the plot category for organization.
    
    Args:
        filename: Name of the plot file
        
    Returns:
        Category name for organizing plots
    """
    file_stem = Path(filename).stem
    
    # Check for joint histogram
    if is_joint_histogram(filename):
        return "Joint_Histogram"
    
    # Check for standard plot types
    for plot_type in PLOT_TYPES:
        if file_stem == plot_type:
            return plot_type
    
    # Unknown plot type
    logging.warning(f"Unknown plot type: {filename}")
    return "Other"


# ==================================================================================
# MAIN COLLECTION LOGIC
# ==================================================================================

class PlotCollector:
    """Organizes plots from parameter sweep results."""
    
    def __init__(self, source_dir: str, target_dir: str):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.stats = {
            'folders_processed': 0,
            'plots_collected': 0,
            'plots_skipped': 0,
            'errors': 0
        }
    
    def collect_all_plots(self) -> Dict[str, int]:
        """
        Main function to collect and organize all plots.
        
        Returns:
            Dictionary with collection statistics
        """
        logging.info(f"Starting plot collection from {self.source_dir}")
        logging.info(f"Target directory: {self.target_dir}")
        
        # Create target directory
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all parameter sweep folders
        sweep_folders = self._find_sweep_folders()
        logging.info(f"Found {len(sweep_folders)} parameter sweep folders")
        
        # Process each folder
        for folder in sweep_folders:
            try:
                self._process_folder(folder)
                self.stats['folders_processed'] += 1
            except Exception as e:
                logging.error(f"Error processing folder {folder}: {e}")
                self.stats['errors'] += 1
        
        # Log final statistics
        self._log_statistics()
        
        return self.stats
    
    def _find_sweep_folders(self) -> List[Path]:
        """Find all folders that contain parameter sweep results."""
        sweep_folders = []
        
        # Look for folders with timestamp and bins pattern
        if not self.source_dir.exists():
            logging.error(f"Source directory does not exist: {self.source_dir}")
            return []
        
        # Search directly in the source directory and subdirectories
        for item in self.source_dir.rglob("*"):
            if item.is_dir() and self._is_sweep_folder(item.name):
                sweep_folders.append(item)
        
        return sorted(sweep_folders)
    
    def _is_sweep_folder(self, folder_name: str) -> bool:
        """Check if folder name matches parameter sweep pattern."""
        # Pattern: timestamp_bins_XxY or similar
        pattern = r'\d{8}_\d{6}.*bins_\d+x\d+'
        return bool(re.search(pattern, folder_name))
    
    def _process_folder(self, folder: Path) -> None:
        """Process a single parameter sweep folder."""
        logging.info(f"Processing folder: {folder.name}")
        
        bin_info = extract_bin_info(folder.name)
        
        # Find all plot files in the folder
        plot_files = self._find_plot_files(folder)
        
        for plot_file in plot_files:
            try:
                self._collect_plot(plot_file, bin_info)
                self.stats['plots_collected'] += 1
            except Exception as e:
                logging.error(f"Error collecting plot {plot_file}: {e}")
                self.stats['plots_skipped'] += 1
    
    def _find_plot_files(self, folder: Path) -> List[Path]:
        """Find all relevant plot files in a folder."""
        plot_files = []
        
        # Look for PNG files
        for file in folder.glob("*.png"):
            if self._is_target_plot(file.name):
                plot_files.append(file)
        
        return plot_files
    
    def _is_target_plot(self, filename: str) -> bool:
        """Check if file is a target plot type."""
        file_stem = Path(filename).stem
        
        # Check standard plot types
        if file_stem in PLOT_TYPES:
            return True
        
        # Check joint histogram
        if is_joint_histogram(filename):
            return True
        
        return False
    
    def _collect_plot(self, plot_file: Path, bin_info: str) -> None:
        """Collect a single plot file to the appropriate category folder."""
        category = get_plot_category(plot_file.name)
        
        # Create category folder
        category_dir = self.target_dir / category
        category_dir.mkdir(exist_ok=True)
        
        # Generate new filename
        file_stem = plot_file.stem
        file_ext = plot_file.suffix
        new_filename = f"{file_stem}_{bin_info}{file_ext}"
        
        # Target file path
        target_file = category_dir / new_filename
        
        # Handle existing files
        if target_file.exists():
            logging.warning(f"File already exists, overwriting: {target_file}")
        
        # Copy the file
        shutil.copy2(plot_file, target_file)
        logging.debug(f"Copied: {plot_file} -> {target_file}")
    
    def _log_statistics(self) -> None:
        """Log collection statistics."""
        logging.info("="*60)
        logging.info("PLOT COLLECTION COMPLETED")
        logging.info("="*60)
        logging.info(f"Folders processed: {self.stats['folders_processed']}")
        logging.info(f"Plots collected: {self.stats['plots_collected']}")
        logging.info(f"Plots skipped: {self.stats['plots_skipped']}")
        logging.info(f"Errors encountered: {self.stats['errors']}")
        
        # Show category breakdown
        if self.target_dir.exists():
            logging.info("\nPlot categories created:")
            for category_dir in self.target_dir.iterdir():
                if category_dir.is_dir():
                    plot_count = len(list(category_dir.glob("*.png")))
                    logging.info(f"  {category_dir.name}: {plot_count} plots")


# ==================================================================================
# COMMAND LINE INTERFACE
# ==================================================================================

def main():
    """Main execution function."""
    setup_logging("INFO")
    
    print("🎨 Plot Collection Organizer")
    print("="*50)
    
    # Initialize collector
    collector = PlotCollector(SOURCE_DIR, TARGET_DIR)
    
    # Check if source directory exists
    if not Path(SOURCE_DIR).exists():
        print(f"❌ Source directory not found: {SOURCE_DIR}")
        print("Please ensure you run this script from the correct directory.")
        return
    
    # Run collection
    try:
        stats = collector.collect_all_plots()
        
        # Print summary
        print(f"\n✅ Collection completed!")
        print(f"📁 Processed folders: {stats['folders_processed']}")
        print(f"🖼️  Collected plots: {stats['plots_collected']}")
        
        if stats['plots_skipped'] > 0:
            print(f"⚠️  Skipped plots: {stats['plots_skipped']}")
        
        if stats['errors'] > 0:
            print(f"❌ Errors: {stats['errors']}")
        
        print(f"\n📂 Results saved to: {TARGET_DIR}")
        
        # Show what was created
        target_path = Path(TARGET_DIR)
        if target_path.exists():
            categories = [d.name for d in target_path.iterdir() if d.is_dir()]
            if categories:
                print(f"\n📊 Categories created: {', '.join(categories)}")
    
    except Exception as e:
        logging.error(f"Collection failed: {e}")
        print(f"❌ Collection failed: {e}")
        raise


if __name__ == "__main__":
    main()
