#!/usr/bin/env python3
"""
ConfigFinder GUI - Fast ID-based Config File Finder
==================================================

Ein Tool zum schnellen Auffinden von Config-Dateien basierend auf IDs:
- Scannt Verzeichnisse nach Ordnernamen mit IDs
- Blitzschnelle Suche durch Indexierung
- GUI für einfache Bedienung
- Direktes Öffnen gefundener Configs

Author: Trading Analysis System
Date: Juli 2025
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import sys

class ConfigFinder:
    """Core Logic für das direkte Suchen von Config-Dateien"""
    
    def __init__(self):
        self.progress_callback = None
        self.status_callback = None
        self.cancel_flag = False
    
    def set_callbacks(self, progress_callback, status_callback):
        """Callbacks für Progress und Status Updates setzen."""
        self.progress_callback = progress_callback
        self.status_callback = status_callback
    
    def cancel_operation(self):
        """Operation abbrechen."""
        self.cancel_flag = True
    
    def find_config_by_id(self, root_dir: str, search_id: str) -> Optional[str]:
        """Findet Config-Datei durch direktes Pfad-Testing."""
        if not root_dir or not os.path.exists(root_dir):
            return None
        
        if not search_id:
            return None
            
        # ID normalisieren (nur Zahlen)
        clean_id = re.sub(r'\D', '', search_id)
        
        if not clean_id:
            return None
        
        self.cancel_flag = False
        
        if self.status_callback:
            self.status_callback(f"🔍 Searching for config_ID_{clean_id}.ini...")
        
        try:
            # Alle direkten Unterordner im Hauptverzeichnis
            all_folders = []
            for item in os.listdir(root_dir):
                item_path = os.path.join(root_dir, item)
                if os.path.isdir(item_path):
                    all_folders.append(item)
            
            total_folders = len(all_folders)
            
            if self.status_callback:
                self.status_callback(f"📊 Testing {total_folders} folders for config_ID_{clean_id}.ini...")
            
            # Jeden Ordner durchprobieren
            for i, folder_name in enumerate(all_folders):
                if self.cancel_flag:
                    break
                
                # Progress Update
                if self.progress_callback and total_folders > 0:
                    progress = (i + 1) / total_folders * 100
                    self.progress_callback(progress)
                
                # Pfad zusammenbauen: Hauptverzeichnis + Ordnername + config_ID_XXXX.ini
                config_filename = f"config_ID_{clean_id}.ini"
                full_config_path = os.path.join(root_dir, folder_name, config_filename)
                
                # Prüfen ob Datei existiert
                if os.path.exists(full_config_path):
                    if self.status_callback:
                        self.status_callback(f"✅ Found: {config_filename} in {folder_name}")
                    return full_config_path
                
                # Status Update alle 100 Ordner
                if (i + 1) % 100 == 0 and self.status_callback:
                    self.status_callback(f"🔍 Checked {i + 1}/{total_folders} folders...")
            
            # Nicht gefunden
            if self.status_callback:
                self.status_callback(f"❌ config_ID_{clean_id}.ini not found in any folder")
            
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"❌ Error during search: {str(e)}")
        
        return None

class ConfigFinderGUI:
    """GUI für den ConfigFinder"""
    
    def __init__(self):
        self.config_finder = ConfigFinder()
        self.config_finder.set_callbacks(self.update_progress, self.update_status)
        self.search_thread = None  # Für die Suche
        self.setup_gui()
    
    def setup_gui(self):
        """GUI Setup"""
        self.root = tk.Tk()
        self.root.title("ConfigFinder - Fast ID-based Config Search")
        self.root.geometry("900x750")
        
        # Main Frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=tk.W+tk.E+tk.N+tk.S)
        
        # Configure grid weights for responsive design
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(5, weight=1)  # Results section expandable
        
        # Title
        title_label = ttk.Label(main_frame, text="🔍 ConfigFinder - Fast ID Search", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Source Directory
        ttk.Label(main_frame, text="📁 Source Directory:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.source_var = tk.StringVar(value=r"H:\Rave\WR_VL\full sweep bis 5303")
        source_entry = ttk.Entry(main_frame, textvariable=self.source_var, width=70)
        source_entry.grid(row=1, column=1, sticky=tk.W+tk.E, padx=(10, 5), pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_source).grid(row=1, column=2, pady=5)
        
        # Scan Button - entfernt, da kein Scannen mehr nötig
        # Stattdessen nur Info-Text
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=2, column=0, columnspan=3, pady=10)
        
        info_label = ttk.Label(info_frame, text="� No scanning required - just enter an ID to search!", 
                              font=('Arial', 10), foreground='green')
        info_label.grid(row=0, column=0)
        
        # Search Section
        search_frame = ttk.LabelFrame(main_frame, text="🔎 ID Search", padding="10")
        search_frame.grid(row=3, column=0, columnspan=3, sticky=tk.W+tk.E, pady=10)
        search_frame.columnconfigure(1, weight=1)
        
        # Search ID Input
        ttk.Label(search_frame, text="🎯 Search ID:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=20, font=('Consolas', 12))
        search_entry.grid(row=0, column=1, sticky=tk.W, padx=10, pady=5)
        
        # Search Button
        ttk.Button(search_frame, text="🔍 Find Config", command=self.find_config).grid(row=0, column=2, padx=10)
        
        # Bind Enter key to search
        search_entry.bind('<Return>', lambda e: self.find_config())
        
        # Found Config Path Display
        path_frame = ttk.LabelFrame(main_frame, text="📁 Found Path", padding="10")
        path_frame.grid(row=4, column=0, columnspan=3, sticky=tk.W+tk.E, pady=10)
        path_frame.columnconfigure(1, weight=1)
        
        ttk.Label(path_frame, text="📄 Path:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.config_path_var = tk.StringVar(value="Enter an ID and click 'Find Config' to search")
        config_path_entry = ttk.Entry(path_frame, textvariable=self.config_path_var, state='readonly', font=('Consolas', 9))
        config_path_entry.grid(row=0, column=1, sticky=tk.W+tk.E, padx=10, pady=5)
        
        # Copy Path Button
        ttk.Button(path_frame, text="📋 Copy Path", command=self.copy_config_path).grid(row=0, column=2, padx=5)
        
        # Optional Direct Open (with checkbox control)
        open_control_frame = ttk.Frame(path_frame)
        open_control_frame.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        self.enable_direct_open = tk.BooleanVar(value=False)  # Standardmäßig deaktiviert
        open_checkbox = ttk.Checkbutton(open_control_frame, text="Enable direct open buttons", 
                                       variable=self.enable_direct_open, command=self.toggle_open_buttons)
        open_checkbox.grid(row=0, column=0, sticky=tk.W)
        
        # Direct Open Buttons (initially disabled)
        self.open_buttons_frame = ttk.Frame(open_control_frame)
        self.open_buttons_frame.grid(row=0, column=1, padx=20)
        
        self.open_config_btn = ttk.Button(self.open_buttons_frame, text="📂 Open Config", 
                                         command=self.open_config, state='disabled')
        self.open_config_btn.grid(row=0, column=0, padx=2)
        
        self.open_folder_btn = ttk.Button(self.open_buttons_frame, text="📁 Open Folder", 
                                         command=self.open_folder, state='disabled')
        self.open_folder_btn.grid(row=0, column=1, padx=2)
        
        # Results Section
        results_frame = ttk.LabelFrame(main_frame, text="📋 Search Results", padding="10")
        results_frame.grid(row=5, column=0, columnspan=3, sticky=tk.W+tk.E+tk.N+tk.S, pady=10)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
        # Results Text
        self.results_text = tk.Text(results_frame, height=12, width=90, font=('Consolas', 10))
        self.results_text.grid(row=0, column=0, sticky=tk.W+tk.E+tk.N+tk.S, pady=5)
        
        # Scrollbar for results
        results_scroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        results_scroll.grid(row=0, column=1, sticky=tk.N+tk.S)
        self.results_text.configure(yscrollcommand=results_scroll.set)
        
        # Statistics
        stats_frame = ttk.Frame(results_frame)
        stats_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W+tk.E, pady=5)
        
        self.stats_var = tk.StringVar(value="Ready to search - enter an ID above")
        stats_label = ttk.Label(stats_frame, textvariable=self.stats_var)
        stats_label.grid(row=0, column=0, sticky=tk.W)
        
        # Progress Section
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=6, column=0, columnspan=3, sticky=tk.W+tk.E, pady=10)
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=0, sticky=tk.W+tk.E, pady=5)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to search - enter an ID above and click 'Find Config'")
        status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        status_label.grid(row=1, column=0, sticky=tk.W)
        
        # Set responsive design
        status_label.grid(row=1, column=0, sticky=tk.W)
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def browse_source(self):
        """Quellverzeichnis auswählen."""
        directory = filedialog.askdirectory(
            title="Select Source Directory to scan for configs",
            initialdir=self.source_var.get() if os.path.exists(self.source_var.get()) else None
        )
        if directory:
            self.source_var.set(directory)
    
    def find_config(self):
        """Config nach ID suchen."""
        search_id = self.search_var.get().strip()
        source_dir = self.source_var.get()
        
        if not search_id:
            messagebox.showwarning("Warning", "Please enter an ID to search for!")
            return
        
        if not source_dir or not os.path.exists(source_dir):
            messagebox.showwarning("Warning", "Please select a valid source directory!")
            return
        
        # Search in separatem Thread starten
        self.search_thread = threading.Thread(
            target=self._search_thread,
            args=(source_dir, search_id)
        )
        self.search_thread.daemon = True
        self.search_thread.start()
    
    def _search_thread(self, source_dir: str, search_id: str):
        """Search-Thread."""
        try:
            # Config suchen
            config_path = self.config_finder.find_config_by_id(source_dir, search_id)
            
            # UI zurücksetzen
            self.root.after(0, self._search_finished, config_path, search_id)
            
        except Exception as e:
            self.root.after(0, self._search_error, str(e))
    
    def _search_finished(self, config_path: Optional[str], search_id: str):
        """Search abgeschlossen."""
        if config_path:
            folder_path = os.path.dirname(config_path)
            folder_name = os.path.basename(folder_path)
            relative_folder_path = os.path.relpath(folder_path, self.source_var.get())
            
            # Pfad im Feld anzeigen
            self.config_path_var.set(config_path)
            
            # Ergebnisse anzeigen
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"🎯 Search Results for ID: {search_id}\n")
            self.results_text.insert(tk.END, "=" * 50 + "\n\n")
            self.results_text.insert(tk.END, f"✅ FOUND!\n\n")
            self.results_text.insert(tk.END, f"📁 Folder: {folder_name}\n")
            self.results_text.insert(tk.END, f"📂 Relative Path: {relative_folder_path}\n")
            clean_id = re.sub(r'\D', '', search_id)
            self.results_text.insert(tk.END, f"📄 Config File: config_ID_{clean_id}.ini\n")
            self.results_text.insert(tk.END, f"📄 Full Path: {config_path}\n\n")
            
            self.results_text.insert(tk.END, f"💡 The full path is displayed above and can be copied with the 'Copy Path' button.\n")
            self.results_text.insert(tk.END, f"📋 Enable direct open buttons with the checkbox if needed.")
            
            self.update_status(f"✅ Found config_ID_{clean_id}.ini")
            
        else:
            # Kein Pfad gefunden
            self.config_path_var.set("Config file not found")
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"🎯 Search Results for ID: {search_id}\n")
            self.results_text.insert(tk.END, "=" * 50 + "\n\n")
            self.results_text.insert(tk.END, f"❌ NOT FOUND\n\n")
            
            clean_id = re.sub(r'\D', '', search_id)
            self.results_text.insert(tk.END, f"config_ID_{clean_id}.ini was not found in any folder.\n\n")
            self.results_text.insert(tk.END, f"💡 Tips:\n")
            self.results_text.insert(tk.END, f"   - Make sure the source directory is correct\n")
            self.results_text.insert(tk.END, f"   - Check if the ID is correct\n")
            self.results_text.insert(tk.END, f"   - The file should be named exactly: config_ID_{clean_id}.ini")
            
            self.update_status(f"❌ config_ID_{clean_id}.ini not found")
    
    def _search_error(self, error_msg: str):
        """Search-Fehler."""
        self.config_path_var.set("Search failed")
        messagebox.showerror("Search Error", f"Search failed:\\n\\n{error_msg}")
    
    def open_config(self):
        """Config-Datei öffnen (nur wenn aktiviert)."""
        if not self.enable_direct_open.get():
            messagebox.showwarning("Warning", "Direct open is disabled. Enable it with the checkbox first!")
            return
        
        path = self.config_path_var.get()
        
        if not path or path in ["Enter an ID and click 'Find Config' to search", "Config file not found", "Search failed"]:
            messagebox.showwarning("Warning", "No valid path available!")
            return
        
        # Prüfen ob es ein config_ID_*.ini Pfad ist oder ein Ordner-Pfad
        if path.endswith('.ini') and os.path.exists(path):
            config_path = path
        else:
            messagebox.showwarning("Warning", f"Path is not a config file:\n{path}")
            return
        
        try:
            if sys.platform == "win32":
                os.startfile(config_path)
            elif sys.platform == "darwin":  # macOS
                subprocess.run(["open", config_path])
            else:  # Linux
                subprocess.run(["xdg-open", config_path])
            
            search_id = self.search_var.get().strip()
            self.update_status(f"📄 Opened config file for ID {search_id}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not open config file:\n{str(e)}")
    
    def open_folder(self):
        """Ordner öffnen (nur wenn aktiviert)."""
        if not self.enable_direct_open.get():
            messagebox.showwarning("Warning", "Direct open is disabled. Enable it with the checkbox first!")
            return
        
        path = self.config_path_var.get()
        
        if not path or path in ["Enter an ID and click 'Find Config' to search", "Config file not found", "Search failed"]:
            messagebox.showwarning("Warning", "No valid path available!")
            return
        
        # Prüfen ob es ein config_ID_*.ini Pfad ist
        if path.endswith('.ini'):
            folder_path = os.path.dirname(path)
        else:
            messagebox.showwarning("Warning", f"Path is not a config file:\n{path}")
            return
        
        if os.path.exists(folder_path):
            try:
                if sys.platform == "win32":
                    os.startfile(folder_path)
                elif sys.platform == "darwin":  # macOS
                    subprocess.run(["open", folder_path])
                else:  # Linux
                    subprocess.run(["xdg-open", folder_path])
                
                search_id = self.search_var.get().strip()
                self.update_status(f"📁 Opened folder for ID {search_id}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not open folder:\n{str(e)}")
        else:
            messagebox.showerror("Error", f"Folder does not exist:\n{folder_path}")
    
    def copy_config_path(self):
        """Pfad in Zwischenablage kopieren."""
        path = self.config_path_var.get()
        
        if path in ["Enter an ID and click 'Find Config' to search", "Config file not found", "Search failed"] or not path:
            messagebox.showwarning("Warning", "No path to copy!")
            return
        
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(path)
            self.root.update()  # Ensures clipboard is updated
            self.update_status(f"📋 Path copied to clipboard")
            messagebox.showinfo("Copied", f"Path copied to clipboard:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not copy path to clipboard:\n{str(e)}")
    
    def toggle_open_buttons(self):
        """Direct Open Buttons aktivieren/deaktivieren."""
        if self.enable_direct_open.get():
            # Buttons aktivieren
            self.open_config_btn.configure(state='normal')
            self.open_folder_btn.configure(state='normal')
            self.update_status("📂 Direct open buttons enabled")
        else:
            # Buttons deaktivieren
            self.open_config_btn.configure(state='disabled')
            self.open_folder_btn.configure(state='disabled')
            self.update_status("🔒 Direct open buttons disabled")

    def update_progress(self, value):
        """Progress Bar aktualisieren."""
        self.root.after(0, lambda: self.progress_var.set(value))
    
    def update_status(self, message):
        """Status-Nachricht aktualisieren."""
        self.root.after(0, lambda: self.status_var.set(message))
    
    def on_closing(self):
        """Beim Schließen des Fensters."""
        if self.search_thread and self.search_thread.is_alive():
            result = messagebox.askokcancel(
                "Quit", 
                "Search is in progress. Do you want to cancel and quit?"
            )
            if result:
                self.config_finder.cancel_operation()
                self.root.after(1000, self.root.destroy)
        else:
            self.root.destroy()
    
    def run(self):
        """GUI starten."""
        self.root.mainloop()

def main():
    """Hauptfunktion."""
    try:
        app = ConfigFinderGUI()
        app.run()
    except Exception as e:
        print(f"❌ Error starting ConfigFinder: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
