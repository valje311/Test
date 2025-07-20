"""
Modern GUI for Plot Collection Organizer
========================================

A sleek, user-friendly interface for organizing plots from parameter sweep results.
Optimized for 4K displays with modern design elements.

Features:
- Source and target directory selection
- Real-time progress tracking
- Statistics display
- Dark/Light theme support
- 4K display optimization

Author: Created for Trading Analysis
Date: July 2025
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import os
import sys
from pathlib import Path
from datetime import datetime
import logging
import json

# Import the main collection logic
from PlotCollector import PlotCollector, setup_logging


# ==================================================================================
# MODERN GUI CONFIGURATION
# ==================================================================================

class ModernTheme:
    """Modern color scheme and styling configuration."""
    
    # Dark theme colors
    DARK_BG = "#1e1e1e"
    DARK_FG = "#ffffff"
    DARK_SURFACE = "#2d2d2d"
    DARK_PRIMARY = "#0078d4"
    DARK_SUCCESS = "#107c10"
    DARK_WARNING = "#ff8c00"
    DARK_ERROR = "#d13438"
    DARK_BORDER = "#404040"
    
    # Light theme colors
    LIGHT_BG = "#f5f5f5"
    LIGHT_FG = "#000000"
    LIGHT_SURFACE = "#ffffff"
    LIGHT_PRIMARY = "#0078d4"
    LIGHT_SUCCESS = "#107c10"
    LIGHT_WARNING = "#ff8c00"
    LIGHT_ERROR = "#d13438"
    LIGHT_BORDER = "#d1d1d1"
    
    # 4K optimized sizes
    WINDOW_WIDTH = 1400
    WINDOW_HEIGHT = 900
    FONT_SIZE = 12
    HEADER_FONT_SIZE = 16
    PADDING = 20
    BUTTON_HEIGHT = 45
    ENTRY_HEIGHT = 35


# ==================================================================================
# PROGRESS TRACKING
# ==================================================================================

class ProgressTracker:
    """Tracks and reports progress for GUI updates."""
    
    def __init__(self, gui_queue: queue.Queue):
        self.gui_queue = gui_queue
        self.total_folders = 0
        self.processed_folders = 0
        self.collected_plots = 0
        self.errors = 0
        
    def set_total_folders(self, total: int):
        """Set total number of folders to process."""
        self.total_folders = total
        self.gui_queue.put(('progress_total', total))
    
    def update_progress(self, folder_name: str, plots_count: int):
        """Update progress with folder completion."""
        self.processed_folders += 1
        self.collected_plots += plots_count
        
        progress_data = {
            'folder_name': folder_name,
            'processed': self.processed_folders,
            'total': self.total_folders,
            'plots_collected': self.collected_plots,
            'errors': self.errors
        }
        
        self.gui_queue.put(('progress_update', progress_data))
    
    def report_error(self, error_msg: str):
        """Report an error."""
        self.errors += 1
        self.gui_queue.put(('error', error_msg))
    
    def report_completion(self, stats: dict):
        """Report final completion stats."""
        self.gui_queue.put(('completion', stats))


# ==================================================================================
# CUSTOM WIDGETS
# ==================================================================================

class ModernButton(tk.Button):
    """Custom button with modern styling."""
    
    def __init__(self, parent, text, command=None, style="primary", **kwargs):
        theme = ModernTheme()
        
        # Style configurations
        styles = {
            "primary": {
                "bg": theme.DARK_PRIMARY,
                "fg": "white",
                "activebackground": "#106ebe",
                "activeforeground": "white"
            },
            "success": {
                "bg": theme.DARK_SUCCESS,
                "fg": "white",
                "activebackground": "#0f6b0f",
                "activeforeground": "white"
            },
            "secondary": {
                "bg": theme.DARK_SURFACE,
                "fg": theme.DARK_FG,
                "activebackground": "#3d3d3d",
                "activeforeground": theme.DARK_FG
            }
        }
        
        style_config = styles.get(style, styles["primary"])
        
        # Use a no-op function if command is None
        if command is None:
            command = lambda: None
        
        super().__init__(
            parent,
            text=text,
            command=command,
            font=("Segoe UI", theme.FONT_SIZE),
            height=2,
            relief="flat",
            borderwidth=0,
            cursor="hand2",
            **style_config,
            **kwargs
        )
        
        # Hover effects
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        
        self.default_bg = style_config["bg"]
        self.hover_bg = style_config["activebackground"]
    
    def _on_enter(self, event):
        self.config(bg=self.hover_bg)
    
    def _on_leave(self, event):
        self.config(bg=self.default_bg)


class ModernEntry(tk.Entry):
    """Custom entry with modern styling."""
    
    def __init__(self, parent, placeholder="", **kwargs):
        theme = ModernTheme()
        
        super().__init__(
            parent,
            font=("Segoe UI", theme.FONT_SIZE),
            bg=theme.DARK_SURFACE,
            fg=theme.DARK_FG,
            insertbackground=theme.DARK_FG,
            relief="flat",
            borderwidth=1,
            highlightthickness=2,
            highlightcolor=theme.DARK_PRIMARY,
            highlightbackground=theme.DARK_BORDER,
            **kwargs
        )
        
        self.placeholder = placeholder
        self.placeholder_active = False
        
        if placeholder:
            self.insert(0, placeholder)
            self.config(fg="#888888")
            self.placeholder_active = True
        
        self.bind("<FocusIn>", self._on_focus_in)
        self.bind("<FocusOut>", self._on_focus_out)
    
    def _on_focus_in(self, event):
        if self.placeholder_active:
            self.delete(0, tk.END)
            self.config(fg=ModernTheme.DARK_FG)
            self.placeholder_active = False
    
    def _on_focus_out(self, event):
        if not self.get():
            self.insert(0, self.placeholder)
            self.config(fg="#888888")
            self.placeholder_active = True


# ==================================================================================
# MAIN GUI APPLICATION
# ==================================================================================

class PlotCollectorGUI:
    """Modern GUI for Plot Collection Organizer."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.theme = ModernTheme()
        self.gui_queue = queue.Queue()
        self.progress_tracker = None
        self.collection_thread = None
        self.is_collecting = False
        
        # Configuration
        self.config_file = "plot_collector_config.json"
        self.config = self.load_config()
        
        self.setup_window()
        self.setup_styles()
        self.create_widgets()
        
        # Load saved paths and update preview
        self.load_saved_paths()
        self.update_filename_preview()
        
        # Start GUI update loop
        self.process_queue()
    
    def setup_window(self):
        """Configure the main window."""
        self.root.title("Plot Sammlung Organizer")
        self.root.geometry(f"{self.theme.WINDOW_WIDTH}x{self.theme.WINDOW_HEIGHT}")
        self.root.configure(bg=self.theme.DARK_BG)
        self.root.resizable(True, True)
        
        # Center window on screen
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.theme.WINDOW_WIDTH // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.theme.WINDOW_HEIGHT // 2)
        self.root.geometry(f"{self.theme.WINDOW_WIDTH}x{self.theme.WINDOW_HEIGHT}+{x}+{y}")
        
        # Configure window icon if available
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass
    
    def setup_styles(self):
        """Configure ttk styles for modern appearance."""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure progress bar style
        self.style.configure(
            "Modern.Horizontal.TProgressbar",
            background=self.theme.DARK_PRIMARY,
            troughcolor=self.theme.DARK_SURFACE,
            borderwidth=0,
            lightcolor=self.theme.DARK_PRIMARY,
            darkcolor=self.theme.DARK_PRIMARY
        )
    
    def create_widgets(self):
        """Create and layout all GUI widgets."""
        # Main container
        main_frame = tk.Frame(self.root, bg=self.theme.DARK_BG)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=self.theme.PADDING, pady=self.theme.PADDING)
        
        # Header
        self.create_header(main_frame)
        
        # Path selection section
        self.create_path_section(main_frame)
        
        # Control buttons
        self.create_control_section(main_frame)
        
        # Progress section
        self.create_progress_section(main_frame)
        
        # Log section
        self.create_log_section(main_frame)
        
        # Status bar
        self.create_status_bar(main_frame)
    
    def create_header(self, parent):
        """Create the header section."""
        header_frame = tk.Frame(parent, bg=self.theme.DARK_BG)
        header_frame.pack(fill=tk.X, pady=(0, self.theme.PADDING))
        
        # Title
        title_label = tk.Label(
            header_frame,
            text="🎨 Plot Sammlung Organizer",
            font=("Segoe UI", self.theme.HEADER_FONT_SIZE + 4, "bold"),
            bg=self.theme.DARK_BG,
            fg=self.theme.DARK_FG
        )
        title_label.pack(side=tk.LEFT)
        
        # Version info
        version_label = tk.Label(
            header_frame,
            text="v2.0 - 4K Optimiert",
            font=("Segoe UI", self.theme.FONT_SIZE - 1),
            bg=self.theme.DARK_BG,
            fg="#888888"
        )
        version_label.pack(side=tk.RIGHT)
    
    def create_path_section(self, parent):
        """Create the path selection section."""
        path_frame = tk.LabelFrame(
            parent,
            text="📁 Verzeichnis Konfiguration",
            font=("Segoe UI", self.theme.FONT_SIZE, "bold"),
            bg=self.theme.DARK_BG,
            fg=self.theme.DARK_FG,
            relief="flat",
            bd=1
        )
        path_frame.pack(fill=tk.X, pady=(0, self.theme.PADDING))
        
        # Source directory
        source_frame = tk.Frame(path_frame, bg=self.theme.DARK_BG)
        source_frame.pack(fill=tk.X, padx=15, pady=10)
        
        tk.Label(
            source_frame,
            text="Quellverzeichnis (Parameter Sweep Ordner):",
            font=("Segoe UI", self.theme.FONT_SIZE),
            bg=self.theme.DARK_BG,
            fg=self.theme.DARK_FG
        ).pack(anchor=tk.W)
        
        source_input_frame = tk.Frame(source_frame, bg=self.theme.DARK_BG)
        source_input_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.source_entry = ModernEntry(
            source_input_frame,
            placeholder="Wählen Sie das Quellverzeichnis mit den Parameter Sweep Ordnern...",
            width=60
        )
        self.source_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        ModernButton(
            source_input_frame,
            text="Durchsuchen",
            command=self.browse_source_directory,
            style="secondary"
        ).pack(side=tk.RIGHT)
        
        # Target directory
        target_frame = tk.Frame(path_frame, bg=self.theme.DARK_BG)
        target_frame.pack(fill=tk.X, padx=15, pady=10)
        
        tk.Label(
            target_frame,
            text="Zielverzeichnis (Organisierte Plots Ausgabe):",
            font=("Segoe UI", self.theme.FONT_SIZE),
            bg=self.theme.DARK_BG,
            fg=self.theme.DARK_FG
        ).pack(anchor=tk.W)
        
        target_input_frame = tk.Frame(target_frame, bg=self.theme.DARK_BG)
        target_input_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.target_entry = ModernEntry(
            target_input_frame,
            placeholder="Wählen Sie das Zielverzeichnis für die organisierte Plot Sammlung...",
            width=60
        )
        self.target_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        ModernButton(
            target_input_frame,
            text="Durchsuchen",
            command=self.browse_target_directory,
            style="secondary"
        ).pack(side=tk.RIGHT)
        
        # File naming section
        naming_frame = tk.Frame(path_frame, bg=self.theme.DARK_BG)
        naming_frame.pack(fill=tk.X, padx=15, pady=10)
        
        tk.Label(
            naming_frame,
            text="Dateiname Zusatz (optional):",
            font=("Segoe UI", self.theme.FONT_SIZE),
            bg=self.theme.DARK_BG,
            fg=self.theme.DARK_FG
        ).pack(anchor=tk.W)
        
        naming_input_frame = tk.Frame(naming_frame, bg=self.theme.DARK_BG)
        naming_input_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.suffix_entry = ModernEntry(
            naming_input_frame,
            placeholder="z.B. _analyse_v2 oder _final (wird an jeden Dateinamen angehängt)",
            width=60
        )
        self.suffix_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        # Bind event to update preview
        self.suffix_entry.bind('<KeyRelease>', self.update_filename_preview)
        
        # Filename preview
        preview_frame = tk.Frame(path_frame, bg=self.theme.DARK_BG)
        preview_frame.pack(fill=tk.X, padx=15, pady=(5, 10))
        
        tk.Label(
            preview_frame,
            text="Vorschau Dateiname:",
            font=("Segoe UI", self.theme.FONT_SIZE),
            bg=self.theme.DARK_BG,
            fg=self.theme.DARK_FG
        ).pack(anchor=tk.W)
        
        self.preview_label = tk.Label(
            preview_frame,
            text="False_Nearest_Neighbors_bins_150x200.png",
            font=("Consolas", self.theme.FONT_SIZE - 1),
            bg=self.theme.DARK_SURFACE,
            fg=self.theme.DARK_PRIMARY,
            anchor=tk.W,
            relief="flat",
            padx=10,
            pady=5
        )
        self.preview_label.pack(fill=tk.X, pady=(5, 0))
    
    def create_control_section(self, parent):
        """Create the control buttons section."""
        control_frame = tk.Frame(parent, bg=self.theme.DARK_BG)
        control_frame.pack(fill=tk.X, pady=(0, self.theme.PADDING))
        
        # Main action button
        self.start_button = ModernButton(
            control_frame,
            text="🚀 Sammlung Starten",
            command=self.start_collection,
            style="primary"
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Stop button
        self.stop_button = ModernButton(
            control_frame,
            text="⏹️ Stoppen",
            command=self.stop_collection,
            style="secondary",
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Config rename button
        self.rename_config_button = ModernButton(
            control_frame,
            text="🏷️ Config ID's Setzen",
            command=self.start_config_rename,
            style="secondary"
        )
        self.rename_config_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Clear log button
        ModernButton(
            control_frame,
            text="🧹 Log Leeren",
            command=self.clear_log,
            style="secondary"
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        # Open target folder button
        self.open_target_button = ModernButton(
            control_frame,
            text="📂 Ergebnisse Öffnen",
            command=self.open_target_folder,
            style="secondary"
        )
        self.open_target_button.pack(side=tk.RIGHT)
    
    def create_progress_section(self, parent):
        """Create the progress tracking section."""
        progress_frame = tk.LabelFrame(
            parent,
            text="📊 Fortschritt",
            font=("Segoe UI", self.theme.FONT_SIZE, "bold"),
            bg=self.theme.DARK_BG,
            fg=self.theme.DARK_FG,
            relief="flat",
            bd=1
        )
        progress_frame.pack(fill=tk.X, pady=(0, self.theme.PADDING))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            style="Modern.Horizontal.TProgressbar"
        )
        self.progress_bar.pack(fill=tk.X, padx=15, pady=10)
        
        # Progress labels
        progress_labels_frame = tk.Frame(progress_frame, bg=self.theme.DARK_BG)
        progress_labels_frame.pack(fill=tk.X, padx=15, pady=(0, 10))
        
        self.progress_label = tk.Label(
            progress_labels_frame,
            text="Bereit für die Sammlung",
            font=("Segoe UI", self.theme.FONT_SIZE),
            bg=self.theme.DARK_BG,
            fg=self.theme.DARK_FG
        )
        self.progress_label.pack(side=tk.LEFT)
        
        # Statistics
        stats_frame = tk.Frame(progress_frame, bg=self.theme.DARK_BG)
        stats_frame.pack(fill=tk.X, padx=15, pady=(0, 10))
        
        self.stats_label = tk.Label(
            stats_frame,
            text="📁 Ordner: 0 | 🖼️ Plots: 0 | ⚠️ Fehler: 0",
            font=("Segoe UI", self.theme.FONT_SIZE),
            bg=self.theme.DARK_BG,
            fg="#888888"
        )
        self.stats_label.pack(side=tk.LEFT)
    
    def create_log_section(self, parent):
        """Create the log display section."""
        log_frame = tk.LabelFrame(
            parent,
            text="📝 Sammlung Log",
            font=("Segoe UI", self.theme.FONT_SIZE, "bold"),
            bg=self.theme.DARK_BG,
            fg=self.theme.DARK_FG,
            relief="flat",
            bd=1
        )
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, self.theme.PADDING))
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=12,
            font=("Consolas", self.theme.FONT_SIZE - 1),
            bg=self.theme.DARK_SURFACE,
            fg=self.theme.DARK_FG,
            insertbackground=self.theme.DARK_FG,
            relief="flat",
            wrap=tk.WORD
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Configure text tags for colored output
        self.log_text.tag_configure("info", foreground=self.theme.DARK_FG)
        self.log_text.tag_configure("success", foreground=self.theme.DARK_SUCCESS)
        self.log_text.tag_configure("warning", foreground=self.theme.DARK_WARNING)
        self.log_text.tag_configure("error", foreground=self.theme.DARK_ERROR)
    
    def create_status_bar(self, parent):
        """Create the status bar."""
        status_frame = tk.Frame(parent, bg=self.theme.DARK_SURFACE, height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(
            status_frame,
            text="Bereit",
            font=("Segoe UI", self.theme.FONT_SIZE - 1),
            bg=self.theme.DARK_SURFACE,
            fg=self.theme.DARK_FG,
            anchor=tk.W
        )
        self.status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Time display
        self.time_label = tk.Label(
            status_frame,
            text="",
            font=("Segoe UI", self.theme.FONT_SIZE - 1),
            bg=self.theme.DARK_SURFACE,
            fg="#888888",
            anchor=tk.E
        )
        self.time_label.pack(side=tk.RIGHT, padx=10, pady=5)
        
        self.update_time()
    
    def update_time(self):
        """Update the time display."""
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.config(text=current_time)
        self.root.after(1000, self.update_time)
    
    def load_config(self):
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {}
    
    def save_config(self):
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except:
            pass
    
    def load_saved_paths(self):
        """Load previously saved paths."""
        if 'source_path' in self.config and self.config['source_path']:
            self.source_entry.delete(0, tk.END)
            self.source_entry.insert(0, self.config['source_path'])
            self.source_entry.config(fg=self.theme.DARK_FG)
            self.source_entry.placeholder_active = False
        
        if 'target_path' in self.config and self.config['target_path']:
            self.target_entry.delete(0, tk.END)
            self.target_entry.insert(0, self.config['target_path'])
            self.target_entry.config(fg=self.theme.DARK_FG)
            self.target_entry.placeholder_active = False
        
        if 'suffix' in self.config and self.config['suffix']:
            self.suffix_entry.delete(0, tk.END)
            self.suffix_entry.insert(0, self.config['suffix'])
            self.suffix_entry.config(fg=self.theme.DARK_FG)
            self.suffix_entry.placeholder_active = False
    
    def update_filename_preview(self, event=None):
        """Update the filename preview based on the suffix input."""
        suffix = self.get_suffix()
        
        # Example filename components
        plot_name = "False_Nearest_Neighbors"
        config_id = "047370"  # Example 6-digit config ID
        bins = "150x200"
        
        # Build preview filename with new format: plotname_suffix_ID_XXXXXX_bins_info.png
        if suffix and not suffix.startswith('_'):
            suffix = '_' + suffix
        
        preview_filename = f"{plot_name}{suffix}_ID_{config_id}_bins_{bins}.png"
        
        # Update preview label
        self.preview_label.config(text=preview_filename)
    
    def get_suffix(self):
        """Get the suffix from entry."""
        suffix = self.suffix_entry.get()
        if self.suffix_entry.placeholder_active:
            return ""
        return suffix.strip()
    
    def browse_source_directory(self):
        """Browse for source directory."""
        directory = filedialog.askdirectory(
            title="Quellverzeichnis auswählen",
            initialdir=self.config.get('source_path', '')
        )
        
        if directory:
            self.source_entry.delete(0, tk.END)
            self.source_entry.insert(0, directory)
            self.source_entry.config(fg=self.theme.DARK_FG)
            self.source_entry.placeholder_active = False
            
            self.config['source_path'] = directory
            self.save_config()
            
            self.log_message(f"Quellverzeichnis ausgewählt: {directory}", "info")
    
    def browse_target_directory(self):
        """Browse for target directory."""
        directory = filedialog.askdirectory(
            title="Zielverzeichnis auswählen",
            initialdir=self.config.get('target_path', '')
        )
        
        if directory:
            self.target_entry.delete(0, tk.END)
            self.target_entry.insert(0, directory)
            self.target_entry.config(fg=self.theme.DARK_FG)
            self.target_entry.placeholder_active = False
            
            self.config['target_path'] = directory
            self.save_config()
            
            self.log_message(f"Zielverzeichnis ausgewählt: {directory}", "info")
    
    def get_source_path(self):
        """Get the source path from entry."""
        path = self.source_entry.get()
        if self.source_entry.placeholder_active:
            return ""
        return path
    
    def get_target_path(self):
        """Get the target path from entry."""
        path = self.target_entry.get()
        if self.target_entry.placeholder_active:
            return ""
        return path
    
    def validate_paths(self):
        """Validate the selected paths."""
        source_path = self.get_source_path()
        target_path = self.get_target_path()
        
        if not source_path:
            messagebox.showerror("Fehler", "Bitte wählen Sie ein Quellverzeichnis aus.")
            return False
        
        if not target_path:
            messagebox.showerror("Fehler", "Bitte wählen Sie ein Zielverzeichnis aus.")
            return False
        
        if not os.path.exists(source_path):
            messagebox.showerror("Fehler", f"Quellverzeichnis existiert nicht:\n{source_path}")
            return False
        
        if source_path == target_path:
            messagebox.showerror("Fehler", "Quell- und Zielverzeichnis dürfen nicht identisch sein.")
            return False
        
        return True
    
    def start_collection(self):
        """Start the plot collection process."""
        if not self.validate_paths():
            return
        
        if self.is_collecting:
            messagebox.showwarning("Warnung", "Die Sammlung läuft bereits.")
            return
        
        # Update UI state
        self.is_collecting = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.open_target_button.config(state=tk.DISABLED)
        
        # Clear previous log
        self.log_text.delete(1.0, tk.END)
        
        # Reset progress
        self.progress_var.set(0)
        self.progress_label.config(text="Sammlung wird gestartet...")
        self.status_label.config(text="Sammlung läuft...")
        
        # Start collection in separate thread
        self.collection_thread = threading.Thread(target=self.run_collection, daemon=True)
        self.collection_thread.start()
        
        self.log_message("🚀 Sammlung gestartet!", "success")
    
    def start_config_rename(self):
        """Start the config.ini renaming process."""
        if not self.validate_paths():
            return
        
        if self.is_collecting:
            messagebox.showwarning("Warnung", "Es läuft bereits ein Prozess.")
            return
        
        # Ask for confirmation
        result = messagebox.askyesno(
            "Config Umbenennung",
            "Diese Funktion benennt alle config.ini Dateien in den Parameter-Ordnern um.\n\n"
            "Format: config_ID_XXXX.ini\n\n"
            "Möchten Sie fortfahren?"
        )
        
        if not result:
            return
        
        # Update UI state
        self.is_collecting = True
        self.start_button.config(state=tk.DISABLED)
        self.rename_config_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.open_target_button.config(state=tk.DISABLED)
        
        # Clear previous log
        self.log_text.delete(1.0, tk.END)
        
        # Reset progress
        self.progress_var.set(0)
        self.progress_label.config(text="Config Umbenennung wird gestartet...")
        self.status_label.config(text="Config Umbenennung läuft...")
        
        # Start renaming in separate thread
        self.collection_thread = threading.Thread(target=self.run_config_rename, daemon=True)
        self.collection_thread.start()
        
        self.log_message("🏷️ Config Umbenennung gestartet!", "success")
    
    def run_config_rename(self):
        """Run the config renaming process in a separate thread."""
        try:
            source_path = self.get_source_path()
            
            # Create progress tracker
            self.progress_tracker = ProgressTracker(self.gui_queue)
            
            # Create config renamer
            renamer = GUIConfigRenamer(source_path, self.progress_tracker)
            
            # Run renaming
            stats = renamer.rename_all_configs()
            
            # Report completion
            self.progress_tracker.report_completion(stats)
            
        except Exception as e:
            self.gui_queue.put(('error', f"Config Umbenennung fehlgeschlagen: {str(e)}"))
            self.gui_queue.put(('collection_complete', False))
    
    def run_collection(self):
        """Run the collection process in a separate thread."""
        try:
            source_path = self.get_source_path()
            target_path = self.get_target_path()
            suffix = self.get_suffix()
            
            # Create progress tracker
            self.progress_tracker = ProgressTracker(self.gui_queue)
            
            # Create collector with GUI progress tracking
            collector = GUIPlotCollector(source_path, target_path, self.progress_tracker, suffix)
            
            # Run collection
            stats = collector.collect_all_plots()
            
            # Report completion
            self.progress_tracker.report_completion(stats)
            
        except Exception as e:
            self.gui_queue.put(('error', f"Sammlung fehlgeschlagen: {str(e)}"))
            self.gui_queue.put(('collection_complete', False))
    
    def stop_collection(self):
        """Stop the collection process."""
        if self.is_collecting:
            self.is_collecting = False
            self.gui_queue.put(('collection_stopped', None))
            self.log_message("🛑 Sammlung vom Benutzer gestoppt", "warning")
    
    def clear_log(self):
        """Clear the log display."""
        self.log_text.delete(1.0, tk.END)
        self.log_message("Log geleert", "info")
    
    def open_target_folder(self):
        """Open the target folder in file explorer."""
        target_path = self.get_target_path()
        if target_path and os.path.exists(target_path):
            os.startfile(target_path)
        else:
            messagebox.showwarning("Warnung", "Zielverzeichnis existiert nicht oder ist nicht angegeben.")
    
    def log_message(self, message, level="info"):
        """Add a message to the log display."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, formatted_message, level)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def process_queue(self):
        """Process messages from the collection thread."""
        try:
            while True:
                message_type, data = self.gui_queue.get_nowait()
                
                if message_type == 'progress_total':
                    self.progress_label.config(text=f"Verarbeite {data} Ordner...")
                
                elif message_type == 'progress_update':
                    processed = data['processed']
                    total = data['total']
                    plots_collected = data['plots_collected']
                    errors = data['errors']
                    
                    # Update progress bar
                    progress_percent = (processed / total) * 100 if total > 0 else 0
                    self.progress_var.set(progress_percent)
                    
                    # Update labels
                    self.progress_label.config(text=f"Verarbeite Ordner {processed}/{total}: {data['folder_name']}")
                    self.stats_label.config(text=f"📁 Ordner: {processed}/{total} | 🖼️ Plots: {plots_collected} | ⚠️ Fehler: {errors}")
                    
                    # Log progress
                    self.log_message(f"Verarbeitet {data['folder_name']} - {plots_collected} Plots insgesamt gesammelt", "info")
                
                elif message_type == 'error':
                    self.log_message(f"❌ Fehler: {data}", "error")
                
                elif message_type == 'completion':
                    self.handle_completion(data)
                
                elif message_type == 'collection_stopped':
                    self.handle_collection_stopped()
                
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.process_queue)
    
    def handle_completion(self, stats):
        """Handle completion of the collection process."""
        self.is_collecting = False
        self.start_button.config(state=tk.NORMAL)
        self.rename_config_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.open_target_button.config(state=tk.NORMAL)
        
        # Update progress
        self.progress_var.set(100)
        
        # Check if this was a config rename operation
        if 'configs_renamed' in stats:
            self.progress_label.config(text="Config Umbenennung abgeschlossen!")
            self.log_message("✅ Config Umbenennung erfolgreich abgeschlossen!", "success")
            self.log_message(f"📊 Endstatistiken:", "info")
            self.log_message(f"   📁 Ordner verarbeitet: {stats['folders_processed']}", "info")
            self.log_message(f"   🏷️ Configs umbenannt: {stats['configs_renamed']}", "info")
            self.log_message(f"   ⚠️ Fehler: {stats['errors']}", "info")
            
            # Show completion message
            messagebox.showinfo(
                "Config Umbenennung Abgeschlossen",
                f"Config-Umbenennung erfolgreich abgeschlossen!\n\n"
                f"📁 Ordner verarbeitet: {stats['folders_processed']}\n"
                f"🏷️ Configs umbenannt: {stats['configs_renamed']}\n"
                f"⚠️ Fehler: {stats['errors']}\n\n"
                f"Verarbeitet in: {self.get_source_path()}"
            )
        else:
            # Regular plot collection completion
            self.progress_label.config(text="Sammlung abgeschlossen!")
            self.log_message("✅ Sammlung erfolgreich abgeschlossen!", "success")
            self.log_message(f"📊 Endstatistiken:", "info")
            self.log_message(f"   📁 Ordner verarbeitet: {stats['folders_processed']}", "info")
            self.log_message(f"   🖼️ Plots gesammelt: {stats['plots_collected']}", "info")
            self.log_message(f"   ⚠️ Fehler: {stats['errors']}", "info")
            
            # Show completion message
            messagebox.showinfo(
                "Sammlung Abgeschlossen",
                f"Plot-Sammlung erfolgreich abgeschlossen!\n\n"
                f"📁 Ordner verarbeitet: {stats['folders_processed']}\n"
                f"🖼️ Plots gesammelt: {stats['plots_collected']}\n"
                f"⚠️ Fehler: {stats['errors']}\n\n"
                f"Ergebnisse gespeichert in: {self.get_target_path()}"
            )
        
        self.status_label.config(text="Bereit")
    
    def handle_collection_stopped(self):
        """Handle manual stop of collection."""
        self.is_collecting = False
        self.start_button.config(state=tk.NORMAL)
        self.rename_config_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.open_target_button.config(state=tk.NORMAL)
        
        self.progress_label.config(text="Prozess gestoppt")
        self.status_label.config(text="Bereit")
    
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


# ==================================================================================
# GUI-INTEGRATED PLOT COLLECTOR
# ==================================================================================

class GUIPlotCollector(PlotCollector):
    """Plot collector with GUI progress reporting."""
    
    def __init__(self, source_dir: str, target_dir: str, progress_tracker: ProgressTracker, suffix: str = ""):
        super().__init__(source_dir, target_dir)
        self.progress_tracker = progress_tracker
        self.suffix = suffix.strip()
        # Ensure suffix starts with underscore if not empty
        if self.suffix and not self.suffix.startswith('_'):
            self.suffix = '_' + self.suffix
        
        # Ensure paths are Path objects
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        
        # Initialize stats if not done by parent
        if not hasattr(self, 'stats'):
            self.stats = {
                'folders_processed': 0,
                'plots_collected': 0,
                'errors': 0
            }
    
    def _collect_plot(self, plot_file: Path, bin_info: str) -> None:
        """Collect a single plot file with custom suffix and config ID to the appropriate category folder."""
        from pathlib import Path
        import shutil
        
        # Get config ID from the folder
        config_id = self._get_config_id_from_folder(plot_file.parent)
        
        # Use enhanced plot category function
        category = self._get_enhanced_plot_category(plot_file.name)
        
        # Create category folder
        category_dir = self.target_dir / category
        category_dir.mkdir(exist_ok=True)
        
        # Generate new filename with suffix and config ID
        file_stem = plot_file.stem
        file_ext = plot_file.suffix
        
        # Format: plotname_suffix_ID_XXXXXX_bins_info.png
        new_filename = f"{file_stem}{self.suffix}_ID_{config_id}_{bin_info}{file_ext}"
        
        # Target file path
        target_file = category_dir / new_filename
        
        # Handle existing files
        if target_file.exists():
            logging.warning(f"File already exists, overwriting: {target_file}")
        
        # Copy the file
        shutil.copy2(plot_file, target_file)
        logging.debug(f"Copied: {plot_file} -> {target_file}")
    
    def _get_enhanced_plot_category(self, filename: str) -> str:
        """Enhanced plot categorization including new plot types."""
        filename_lower = filename.lower()
        
        # Joint Histogram plots (check first since they're very specific)
        if 'joint_histogram' in filename_lower:
            return 'Joint_Histograms'
        
        # Returns Analysis plots
        elif 'returns_histogram' in filename_lower:
            return 'Returns_Histograms'
        elif 'returns_timeseries' in filename_lower:
            return 'Returns_Timeseries'
        elif 'returns_analysis' in filename_lower:
            return 'Returns_Analysis'
        
        # Existing categories
        elif 'false_nearest' in filename_lower or 'fnn' in filename_lower:
            return 'False_Nearest_Neighbors'
        elif 'mutual_information' in filename_lower or 'mi_' in filename_lower:
            return 'Mutual_Information'
        elif 'autocorrelation' in filename_lower or 'acf' in filename_lower:
            return 'Autocorrelation'
        elif 'phase_space' in filename_lower or 'embedding' in filename_lower:
            return 'Phase_Space'
        elif 'candlestick' in filename_lower or 'candle' in filename_lower:
            return 'Candlestick_Charts'
        elif 'perona_malik' in filename_lower or 'smoothing' in filename_lower:
            return 'Smoothing'
        else:
            return 'Other_Plots'
    
    def _get_config_id_from_folder(self, folder_path: Path) -> str:
        """Extract config ID from config file in the folder or generate one."""
        # First, try to find a config file with ID
        for config_file in folder_path.glob('config_ID_*.ini'):
            # Extract ID from filename like config_ID_123456.ini
            import re
            match = re.search(r'config_ID_([A-Za-z0-9]{6})\.ini', config_file.name)
            if match:
                return match.group(1)
        
        # If no ID config found, try regular config.ini and generate ID
        config_file = folder_path / 'config.ini'
        if config_file.exists():
            # Generate ID from folder name (same logic as in renamer)
            folder_name = folder_path.name
            import re
            
            # Look for sequential patterns first (for parameter sweeps)
            sequential_patterns = [
                r'run_(\d{6})',
                r'config_(\d{6})', 
                r'sweep_(\d{6})',
                r'param_(\d{6})',
                r'_(\d{6})_',
                r'_(\d{6})$',
                r'^(\d{6})_',
                r'^(\d{6})$'
            ]
            
            for pattern in sequential_patterns:
                match = re.search(pattern, folder_name)
                if match:
                    return match.group(1)
            
            # Try timestamp extraction
            timestamp_match = re.search(r'(\d{8}_\d{6})', folder_name)
            if timestamp_match:
                timestamp = timestamp_match.group(1)
                return timestamp[-6:]
            else:
                import hashlib
                hash_object = hashlib.md5(folder_name.encode())
                # Convert to number and take modulo to get 6 digits
                hex_hash = hash_object.hexdigest()
                return str(int(hex_hash, 16) % 1000000).zfill(6)
        
        # Fallback: use folder name hash
        import hashlib
        hash_object = hashlib.md5(folder_path.name.encode())
        hex_hash = hash_object.hexdigest()
        return str(int(hex_hash, 16) % 1000000).zfill(6)
    
    def collect_all_plots(self) -> dict:
        """Collect plots with GUI progress reporting."""
        # Find all sweep folders
        sweep_folders = self._find_sweep_folders()
        self.progress_tracker.set_total_folders(len(sweep_folders))
        
        # Create target directory
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each folder
        for folder in sweep_folders:
            try:
                plots_before = self.stats['plots_collected']
                self._process_folder(folder)
                plots_added = self.stats['plots_collected'] - plots_before
                
                self.stats['folders_processed'] += 1
                self.progress_tracker.update_progress(folder.name, plots_added)
                
            except Exception as e:
                error_msg = f"Error processing folder {folder}: {e}"
                self.progress_tracker.report_error(error_msg)
                self.stats['errors'] += 1
        
        return self.stats
    
    def _find_sweep_folders(self):
        """Find all parameter sweep folders containing plots."""
        folders = []
        try:
            # Look for directories that contain plot files
            for item in self.source_dir.iterdir():
                if item.is_dir():
                    # Check if this directory contains any plot files
                    plot_files = list(item.glob("*.png")) + list(item.glob("*.jpg")) + list(item.glob("*.jpeg"))
                    if plot_files:
                        folders.append(item)
                        
            # Also check subdirectories recursively
            for item in self.source_dir.rglob("*"):
                if item.is_dir() and item not in folders:
                    plot_files = list(item.glob("*.png")) + list(item.glob("*.jpg")) + list(item.glob("*.jpeg"))
                    if plot_files:
                        folders.append(item)
                        
        except Exception as e:
            logging.error(f"Error finding sweep folders: {e}")
        
        return sorted(set(folders))  # Remove duplicates and sort
    
    def _process_folder(self, folder: Path):
        """Process a single folder and collect plots."""
        plot_files = list(folder.glob("*.png")) + list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg"))
        
        for plot_file in plot_files:
            try:
                # Extract bin info from filename or use default
                bin_info = self._extract_bin_info(plot_file.name)
                self._collect_plot(plot_file, bin_info)
                self.stats['plots_collected'] += 1
                
            except Exception as e:
                logging.error(f"Error processing plot {plot_file}: {e}")
                self.stats['errors'] += 1
    
    def _extract_bin_info(self, filename: str) -> str:
        """Extract bin information from filename."""
        import re
        
        # Look for patterns like "150-150", "150x200", "bins_150x200", etc.
        patterns = [
            r'(\d+x\d+)',          # 150x200
            r'(\d+-\d+)',          # 150-150
            r'bins[_-](\d+x\d+)',  # bins_150x200
            r'bins[_-](\d+-\d+)',  # bins_150-150
            r'dim(\d+)',           # dim15
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return f"bins_{match.group(1)}"
        
        # Default if no pattern found
        return "bins_none"


# ==================================================================================
# GUI-INTEGRATED CONFIG RENAMER
# ==================================================================================

class GUIConfigRenamer:
    """Config.ini renamer with GUI progress reporting."""
    
    def __init__(self, source_dir: str, progress_tracker: ProgressTracker):
        self.source_dir = Path(source_dir)
        self.progress_tracker = progress_tracker
        self.stats = {
            'folders_processed': 0,
            'configs_renamed': 0,
            'errors': 0
        }
    
    def _find_sweep_folders(self):
        """Find all parameter sweep folders."""
        folders = []
        try:
            for item in self.source_dir.iterdir():
                if item.is_dir():
                    # Look for config.ini files in subdirectories
                    for config_file in item.rglob('config.ini'):
                        if config_file.parent not in folders:
                            folders.append(config_file.parent)
        except Exception as e:
            logging.error(f"Error finding sweep folders: {e}")
        
        return sorted(folders)
    
    def _generate_config_id(self, folder_path: Path) -> str:
        """Generate a unique ID for the config file based on folder structure."""
        # Use sequential number from folder name if available
        folder_name = folder_path.name
        
        # Try to extract sequential number from folder name (e.g., "run_000001", "config_000001", etc.)
        import re
        
        # Look for patterns like run_123456, config_123456, sweep_123456, etc.
        sequential_patterns = [
            r'run_(\d{6})',
            r'config_(\d{6})', 
            r'sweep_(\d{6})',
            r'param_(\d{6})',
            r'_(\d{6})_',
            r'_(\d{6})$',
            r'^(\d{6})_',
            r'^(\d{6})$'
        ]
        
        for pattern in sequential_patterns:
            match = re.search(pattern, folder_name)
            if match:
                return match.group(1)  # Return the 6-digit number
        
        # Try to extract timestamp from folder name
        timestamp_match = re.search(r'(\d{8}_\d{6})', folder_name)
        if timestamp_match:
            timestamp = timestamp_match.group(1)
            # Extract last 6 digits as ID
            config_id = timestamp[-6:]
        else:
            # Generate 6-digit ID from folder name hash
            import hashlib
            hash_object = hashlib.md5(folder_name.encode())
            # Take first 6 characters and ensure they're digits
            hex_hash = hash_object.hexdigest()
            # Convert to number and take modulo to get 6 digits
            config_id = str(int(hex_hash, 16) % 1000000).zfill(6)
        
        return config_id
    
    def _rename_config_file(self, config_file: Path) -> bool:
        """Rename a single config.ini file with ID."""
        try:
            folder_path = config_file.parent
            config_id = self._generate_config_id(folder_path)
            
            # New filename
            new_filename = f"config_ID_{config_id}.ini"
            new_path = folder_path / new_filename
            
            # Check if already renamed
            if config_file.name.startswith('config_ID_'):
                logging.info(f"Config already renamed: {config_file}")
                return False
            
            # Check if target exists
            if new_path.exists():
                logging.warning(f"Target config file already exists: {new_path}")
                return False
            
            # Rename the file
            config_file.rename(new_path)
            logging.info(f"Renamed: {config_file.name} -> {new_filename}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error renaming config file {config_file}: {e}")
            return False
    
    def rename_all_configs(self) -> dict:
        """Rename all config.ini files with GUI progress reporting."""
        # Find all folders with config.ini
        config_folders = self._find_sweep_folders()
        self.progress_tracker.set_total_folders(len(config_folders))
        
        if not config_folders:
            logging.warning("No config.ini files found in subdirectories")
            return self.stats
        
        # Process each folder
        for folder in config_folders:
            try:
                config_file = folder / 'config.ini'
                
                if config_file.exists():
                    renamed = self._rename_config_file(config_file)
                    if renamed:
                        self.stats['configs_renamed'] += 1
                
                self.stats['folders_processed'] += 1
                self.progress_tracker.update_progress(folder.name, self.stats['configs_renamed'])
                
            except Exception as e:
                error_msg = f"Error processing folder {folder}: {e}"
                self.progress_tracker.report_error(error_msg)
                self.stats['errors'] += 1
        
        return self.stats


# ==================================================================================
# MAIN APPLICATION
# ==================================================================================

def main():
    """Main application entry point."""
    try:
        # Configure logging
        setup_logging("INFO")
        
        # Create and run GUI
        app = PlotCollectorGUI()
        app.run()
        
    except Exception as e:
        messagebox.showerror("Application Error", f"Failed to start application:\n{str(e)}")


if __name__ == "__main__":
    main()
