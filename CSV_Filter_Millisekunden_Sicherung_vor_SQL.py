import tkinter as tk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import pandas as pd
import os
import numpy as np
from datetime import datetime

class CSVToolApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Erweitertes CSV Tool")
        self.root.geometry("800x600") # Startgröße

        # Modernes Theme anwenden
        style = ttk.Style()
        style.theme_use('clam') # 'clam', 'alt', 'default', 'classic'

        self.df = None
        self.filepath = None
        self.column_order_in_listbox = [] # Hält die aktuelle Reihenfolge in der Listbox

        # --- Hauptlayout mit PanedWindow ---
        self.main_paned_window = ttk.PanedWindow(root, orient=tk.VERTICAL)
        self.main_paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Oberer Bereich (Einstellungen & Aktionen) ---
        self.top_frame = ttk.Frame(self.main_paned_window, padding="5")
        self.main_paned_window.add(self.top_frame, weight=0) # Nicht stark vergrößern

        # --- Unterer Bereich (Vorschau & Spalten) ---
        self.bottom_paned_window = ttk.PanedWindow(self.top_frame, orient=tk.HORIZONTAL)
        self.bottom_paned_window.pack(fill=tk.BOTH, expand=True)

        # --- Linker Bereich unten (Spaltenauswahl & Dezimalstellen) ---
        self.left_frame = ttk.Frame(self.bottom_paned_window, padding="5")
        self.bottom_paned_window.add(self.left_frame, weight=1)

        # --- Rechter Bereich unten (Datenvorschau) ---
        self.right_frame = ttk.Frame(self.bottom_paned_window, padding="5")
        self.bottom_paned_window.add(self.right_frame, weight=3) # Vorschau bekommt mehr Platz

        # --- Statusleiste ---
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding="2 5")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.set_status("Bereit.")

        # --- UI Elemente erstellen ---
        self._create_load_options_ui(self.top_frame)
        self._create_column_management_ui(self.left_frame)
        self._create_decimal_settings_ui(self.left_frame)
        self._create_export_options_ui(self.left_frame)
        self._create_data_preview_ui(self.right_frame)

        # Konfiguration für Größenanpassung
        self.top_frame.columnconfigure(1, weight=1) # Button-Spalte soll sich ausdehnen
        self.left_frame.columnconfigure(0, weight=1)
        self.left_frame.rowconfigure(0, weight=1) # Spaltenauswahl-Frame soll wachsen
        self.right_frame.columnconfigure(0, weight=1)
        self.right_frame.rowconfigure(1, weight=1) # Treeview soll wachsen


    def set_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks() # UI sofort aktualisieren

    def _create_load_options_ui(self, parent):
        frame = ttk.LabelFrame(parent, text="Datei Laden & Optionen", padding="10")
        frame.pack(fill=tk.X, pady=5)

        ttk.Button(frame, text="CSV Datei laden", command=self.load_csv).grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        ttk.Label(frame, text="Trennzeichen:").grid(row=0, column=1, padx=5, pady=5, sticky="e")
        self.load_separator_var = tk.StringVar(value=';')
        sep_combo = ttk.Combobox(frame, textvariable=self.load_separator_var, values=[';', ',', '\\t', '|', ' '], width=5)
        sep_combo.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        ttk.Label(frame, text="Dezimal:").grid(row=0, column=3, padx=5, pady=5, sticky="e")
        self.load_decimal_var = tk.StringVar(value='.')
        dec_combo = ttk.Combobox(frame, textvariable=self.load_decimal_var, values=['.', ','], width=5)
        dec_combo.grid(row=0, column=4, padx=5, pady=5, sticky="w")

        self.file_info_label = ttk.Label(frame, text="Keine Datei geladen.")
        self.file_info_label.grid(row=0, column=5, padx=15, pady=5, sticky="w")

        frame.columnconfigure(5, weight=1) # Label soll übrigen Platz einnehmen


    def _create_column_management_ui(self, parent):
        frame = ttk.LabelFrame(parent, text="Spaltenauswahl und Reihenfolge", padding="10")
        # frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew") # Positioniert durch parent Layout Manager
        frame.pack(fill=tk.BOTH, expand=True, pady=(0,5))

        # Listbox für die Spaltenauswahl
        self.columns_listbox = tk.Listbox(frame, selectmode=tk.EXTENDED, exportselection=0) # EXTENDED für Shift/Ctrl Auswahl
        self.columns_listbox.grid(row=0, column=0, rowspan=4, padx=5, pady=5, sticky="nsew")

        # Scrollbar für die Listbox
        listbox_scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.columns_listbox.yview)
        listbox_scrollbar.grid(row=0, column=1, rowspan=4, pady=5, sticky="ns")
        self.columns_listbox['yscrollcommand'] = listbox_scrollbar.set

        # Buttons zum Verschieben der Spalten
        self.up_button = ttk.Button(frame, text="↑", command=self.move_up, width=3, state=tk.DISABLED)
        self.up_button.grid(row=0, column=2, padx=5, sticky="n")
        self.down_button = ttk.Button(frame, text="↓", command=self.move_down, width=3, state=tk.DISABLED)
        self.down_button.grid(row=1, column=2, padx=5, sticky="s")

        # Button zum Auswählen aller Spalten
        self.select_all_button = ttk.Button(frame, text="Alle ausw.", command=self.select_all, width=10, state=tk.DISABLED)
        self.select_all_button.grid(row=2, column=2, padx=5, pady=5, sticky="ew")
        # Button zum Abwählen aller Spalten
        self.deselect_all_button = ttk.Button(frame, text="Alle abw.", command=self.deselect_all, width=10, state=tk.DISABLED)
        self.deselect_all_button.grid(row=3, column=2, padx=5, pady=5, sticky="ew")


        # Konfiguration für Größenanpassung im Frame
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1) # Listbox soll wachsen

    def _create_decimal_settings_ui(self, parent):
        # Äußeres LabelFrame
        decimal_outer_frame = ttk.LabelFrame(parent, text="Nachkommastellen (Export)", padding="10")
        decimal_outer_frame.pack(fill=tk.X, pady=5)
        
        # Canvas und Scrollbar für scrollbaren Inhalt (Höhe auf 200 erhöht für mehr sichtbare Einträge)
        canvas = tk.Canvas(decimal_outer_frame, height=200, highlightthickness=0)
        scrollbar = ttk.Scrollbar(decimal_outer_frame, orient="vertical", command=canvas.yview)
        
        # Inneres Frame für die tatsächlichen Widgets
        self.decimal_settings_frame = ttk.Frame(canvas)
        
        # Konfiguration des Canvas
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Layout
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Frame im Canvas platzieren
        canvas_frame = canvas.create_window((0, 0), window=self.decimal_settings_frame, anchor="nw")
        
        # Canvas-Größe an Inhalt anpassen
        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(canvas_frame, width=canvas.winfo_width())
            
        # Mausrad-Unterstützung für Scrollen hinzufügen
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            
        self.decimal_settings_frame.bind("<Configure>", configure_scroll_region)
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(canvas_frame, width=e.width))
        canvas.bind_all("<MouseWheel>", _on_mousewheel)  # Für Windows
        
        # Sicherstellen, dass der Canvas nach dem Hinzufügen von Widgets aktualisiert wird
        def update_canvas_later():
            canvas.configure(scrollregion=canvas.bbox("all"))
            
        parent.after(100, update_canvas_later)
        
        self.decimal_widgets = {}
        # Inhalt wird dynamisch in populate_ui_after_load gefüllt

    def _create_export_options_ui(self, parent):
        frame = ttk.LabelFrame(parent, text="Export Einstellungen", padding="10")
        frame.pack(fill=tk.X, pady=5)

        ttk.Label(frame, text="Trennzeichen:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.export_separator_var = tk.StringVar(value=';')
        exp_sep_combo = ttk.Combobox(frame, textvariable=self.export_separator_var, values=[';', ',', '\\t', '|', ' '], width=5)
        exp_sep_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(frame, text="Dezimal:").grid(row=0, column=2, padx=5, pady=5, sticky="e")
        self.export_decimal_var = tk.StringVar(value=',') # Standard für deutschen Export oft Komma
        exp_dec_combo = ttk.Combobox(frame, textvariable=self.export_decimal_var, values=[',', '.'], width=5)
        exp_dec_combo.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        
        # Formatierungsoptionen für Forex-Daten
        self.forex_frame = ttk.LabelFrame(frame, text="Forex-Daten Optionen", padding="5")
        self.forex_frame.grid(row=1, column=0, columnspan=4, padx=5, pady=5, sticky="ew")
        self.forex_frame.grid_remove()  # Standardmäßig ausblenden
        
        # Checkbox für MACD-Signalfilterung
        self.macd_signal_var = tk.BooleanVar(value=False)
        self.macd_signal_check = ttk.Checkbutton(self.forex_frame, text="Nur MACD Kreuzungen exportieren", 
                                                variable=self.macd_signal_var)
        self.macd_signal_check.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # Export-Buttons
        self.export_button = ttk.Button(frame, text="Auswahl Exportieren", command=self.export_csv, state=tk.DISABLED)
        self.export_button.grid(row=2, column=0, columnspan=2, padx=5, pady=10, sticky="ew")
        
        self.export_vis_button = ttk.Button(frame, text="Visualisierung", command=self.show_visualization_options, state=tk.DISABLED)
        self.export_vis_button.grid(row=2, column=2, columnspan=2, padx=5, pady=10, sticky="ew")

        frame.columnconfigure(1, weight=1) # Platz flexibel verteilen
        frame.columnconfigure(3, weight=1)

    def _create_data_preview_ui(self, parent):
        frame = ttk.LabelFrame(parent, text="Datenvorschau (keine Daten geladen)", padding="10")
        # frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew") # Positioniert durch parent
        frame.pack(fill=tk.BOTH, expand=True)

        # Button-Frame für Aktionen oberhalb der Vorschau
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        # Aktualisieren-Button
        self.refresh_preview_button = ttk.Button(
            button_frame, 
            text="Vorschau aktualisieren", 
            command=self.refresh_data_preview,
            state=tk.DISABLED
        )
        self.refresh_preview_button.pack(side=tk.LEFT, padx=5)

        # Dropdown für die Anzahl der anzuzeigenden Zeilen
        ttk.Label(button_frame, text="Zeilen:").pack(side=tk.LEFT, padx=(15, 5))
        self.preview_rows_var = tk.StringVar(value="50")
        rows_combo = ttk.Combobox(button_frame, textvariable=self.preview_rows_var, 
                                values=["10", "25", "50", "100", "200", "500"], width=5)
        rows_combo.pack(side=tk.LEFT)

        # Treeview für Datenvorschau
        self.preview_tree = ttk.Treeview(frame, show='headings') # Nur Überschriften anzeigen

        # Scrollbars für Treeview
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.preview_tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self.preview_tree.xview)
        self.preview_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.grid(row=1, column=1, sticky='ns')
        hsb.grid(row=2, column=0, sticky='ew')
        self.preview_tree.grid(row=1, column=0, sticky='nsew')

        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)  # Treeview soll wachsen


    def detect_and_preprocess_forex_data(self):
        """Erkennt spezifische Spalten der Tick_Aufzeichnung.mq4 Ausgabedatei und bereitet sie vor."""
        if self.df is None:
            return False
            
        # Prüfen, ob die typischen Spalten vorhanden sind
        expected_columns = ["Timestamp", "Open", "High", "Low", "Close", "Volume", 
                           "RSI", "Momentum", "MACD_Main", "MACD_Signal"]
                           
        # Zählen wie viele der erwarteten Spalten vorhanden sind
        matching_cols = sum(1 for col in expected_columns if col in self.df.columns)
        
        # Wenn mindestens 7 der 10 erwarteten Spalten vorhanden sind, dann handelt es sich wahrscheinlich
        # um die Ausgabe unseres MT4-EAs
        if matching_cols >= 7:
            self.set_status("Forex-Daten erkannt. Führe spezifische Vorverarbeitung durch...")
            
            # MACD Cross in kategorische Werte umwandeln, falls vorhanden
            if 'MACD_Cross' in self.df.columns:
                cross_values = {
                    1: "Bullisch",
                    -1: "Bärisch",
                    0: "Keine Kreuzung"
                }
                try:
                    self.df['MACD_Cross_Text'] = self.df['MACD_Cross'].map(cross_values)
                    
                    # Stelle die neue Spalte direkt nach der numerischen Spalte
                    cols = list(self.df.columns)
                    macd_cross_idx = cols.index('MACD_Cross')
                    cols.insert(macd_cross_idx + 1, 'MACD_Cross_Text')
                    cols.remove('MACD_Cross_Text')
                    self.df = self.df[cols]
                except Exception as e:
                    print(f"Fehler bei der MACD-Umwandlung: {str(e)}")
            
            # Forex-spezifische UI-Elemente einblenden
            self.forex_frame.grid()
            self.export_vis_button.config(state=tk.NORMAL)
            
            # MACD-Signale zählen und zusammenfassen
            if 'MACD_Cross' in self.df.columns:
                bullish_signals = sum(self.df['MACD_Cross'] == 1)
                bearish_signals = sum(self.df['MACD_Cross'] == -1)
                
                summary = (f"Die geladene Datei enthält Forex-Handelsdaten aus dem MT4 Expert Advisor.\n\n"
                          f"MACD-Signale Zusammenfassung:\n"
                          f"• {bullish_signals} bullische Signale (MACD kreuzt Signal nach oben)\n"
                          f"• {bearish_signals} bärische Signale (MACD kreuzt Signal nach unten)\n\n"
                          f"Erweiterte Funktionen für die Datenanalyse wurden aktiviert.")
            else:
                summary = ("Die geladene Datei enthält Forex-Handelsdaten aus dem MT4 Expert Advisor.\n\n"
                          "Erweiterte Funktionen für die Datenanalyse wurden aktiviert.")
                
            messagebox.showinfo("Forex-Daten erkannt", summary)
            
            return True
            
        return False

    def add_milliseconds_to_timestamps(self):
        """
        Erweitert Zeitstempel um Millisekunden.
        Bei einmaligen Zeitstempeln: fügt .000 hinzu
        Bei mehrfachen Zeitstempeln: verteilt Millisekunden im Bereich 0-100ms
        nach der Formel: schrittweite = 100ms / (anzahl_gleicher_zeitstempel + 1)
        """
        if self.df is None or len(self.df) == 0:
            self.set_status("Keine Daten zum Erweitern der Zeitstempel vorhanden.")
            return
            
        # Zeitstempelspalten identifizieren
        timestamp_candidates = ['Timestamp', 'Datetime', 'Zeit', 'Date']
        timestamp_col = next((col for col in timestamp_candidates if col in self.df.columns), None)
        
        if not timestamp_col:
            self.set_status("Keine Zeitstempelspalte gefunden.")
            return
            
        self.set_status(f"Erweitere Zeitstempel in Spalte '{timestamp_col}' um Millisekunden...")
        
        try:
            # Dataframe nach Zeitstempel sortieren
            self.df.sort_values(by=timestamp_col, inplace=True)
            
            # Temporäre Spalte für neue Zeitstempel
            self.df['Timestamp_with_ms'] = self.df[timestamp_col].astype(str)
            
            # Gruppiere nach identischen Zeitstempeln
            grouped = self.df.groupby(timestamp_col)
            
            # Zähler für verarbeitete Zeitstempel
            processed_count = 0
            duplicate_groups_count = 0
            
            # Für jede Gruppe von identischen Zeitstempeln
            for timestamp, group in grouped:
                count = len(group)
                processed_count += count
                
                if count == 1:
                    # Einzelner Zeitstempel: füge .000 hinzu
                    idx = group.index[0]
                    current_ts = str(self.df.at[idx, 'Timestamp_with_ms'])
                    # Prüfe, ob bereits Millisekunden vorhanden sind
                    if '.' not in current_ts.split(' ')[-1]:
                        self.df.at[idx, 'Timestamp_with_ms'] = current_ts + '.000'
                else:
                    # Mehrere identische Zeitstempel: verteile Millisekunden
                    duplicate_groups_count += 1
                    ms_step = 100 / (count + 1)
                    
                    for i, idx in enumerate(group.index):
                        # Berechne Millisekunden mit (i+1) für Werte von 1 bis count
                        ms_value = int(ms_step * (i + 1))
                        # Formatiere mit führenden Nullen als .020, .040 usw.
                        ms_str = f'.{ms_value:03d}'
                        
                        current_ts = str(self.df.at[idx, 'Timestamp_with_ms'])
                        if '.' not in current_ts.split(' ')[-1]:
                            self.df.at[idx, 'Timestamp_with_ms'] = current_ts + ms_str
                        else:
                            # Falls bereits Millisekunden vorhanden, ersetze diese
                            base_ts = current_ts.split('.')[0]
                            self.df.at[idx, 'Timestamp_with_ms'] = base_ts + ms_str
            
            # Ursprüngliche Zeitstempelspalte mit der neuen Version ersetzen
            self.df[timestamp_col] = self.df['Timestamp_with_ms']
            # Temporäre Spalte entfernen
            self.df.drop(columns=['Timestamp_with_ms'], inplace=True)
            
            # Detaillierte Statusmeldung
            duplicate_msg = f", {duplicate_groups_count} Gruppen mit Duplikaten" if duplicate_groups_count > 0 else ""
            self.set_status(f"Zeitstempel erfolgreich erweitert: {processed_count} Einträge bearbeitet{duplicate_msg}")
            
        except Exception as e:
            self.set_status(f"Fehler beim Erweitern der Zeitstempel: {str(e)}")
            # Nur löschen, wenn die temporäre Spalte existiert
            if 'Timestamp_with_ms' in self.df.columns:
                self.df.drop(columns=['Timestamp_with_ms'], inplace=True)

    def load_csv(self):
        filepath = filedialog.askopenfilename(
            title="CSV Datei öffnen",
            filetypes=[("CSV Dateien", "*.csv"), ("Textdateien", "*.txt"), ("Alle Dateien", "*.*")]
        )
        if not filepath:
            return

        self.filepath = filepath
        load_sep = self.load_separator_var.get()
        if load_sep == '\\t': load_sep = '\t' # Escape-Sequenz interpretieren
        load_dec = self.load_decimal_var.get()

        # Sichere Verwendung von os.path.basename
        filename = os.path.basename(filepath) if filepath else "Unbekannte Datei"
        self.set_status(f"Lade {filename}...")
        
        try:
            self.df = pd.read_csv(filepath, sep=load_sep, decimal=load_dec)
            self.set_status(f"Datei {filename} geladen.")
            self.file_info_label.config(text=f"{filename} ({self.df.shape[0]} Zeilen, {self.df.shape[1]} Spalten)")

            # --- Zeitstempel um Millisekunden erweitern ---
            self.add_milliseconds_to_timestamps()

            # --- Spezifische Vorverarbeitung für Forex-Daten durchführen ---
            is_forex_data = self.detect_and_preprocess_forex_data()
            
            # --- Nach Datum/Uhrzeit Spalten suchen und zusammenführen ---
            self.check_and_merge_datetime()

            # --- UI mit den geladenen Daten aktualisieren ---
            self.populate_ui_after_load()

            # Buttons aktivieren
            self.export_button.config(state=tk.NORMAL)
            self.up_button.config(state=tk.NORMAL)
            self.down_button.config(state=tk.NORMAL)
            self.select_all_button.config(state=tk.NORMAL)
            self.deselect_all_button.config(state=tk.NORMAL)
            self.refresh_preview_button.config(state=tk.NORMAL)


        except FileNotFoundError:
            messagebox.showerror("Fehler", f"Datei nicht gefunden: {filepath}")
            self.set_status("Fehler beim Laden.")
            self.file_info_label.config(text="Keine Datei geladen.")
        except Exception as e:
            messagebox.showerror("Fehler beim Lesen der CSV",
                                 f"Konnte die Datei nicht lesen.\n"
                                 f"Stellen Sie sicher, dass das korrekte Trennzeichen ('{load_sep}') "
                                 f"und Dezimalzeichen ('{load_dec}') gewählt ist.\n\n"
                                 f"Fehlerdetails: {str(e)}")
            self.set_status("Fehler beim Laden.")
            self.df = None # Sicherstellen, dass kein alter DataFrame übrig bleibt
            self.clear_ui() # UI zurücksetzen

    def check_and_merge_datetime(self):
        if self.df is None:
            return
            
        # Prüfen, ob bereits eine Timestamp-Spalte vorhanden ist
        if 'Timestamp' in self.df.columns:
            try:
                # Bestimmen des Zeitstempelformats durch Prüfung eines Beispielwerts
                sample_timestamp = self.df['Timestamp'].iloc[0] if not self.df['Timestamp'].empty else ""
                self.set_status(f"Prüfe Timestamp-Format: {sample_timestamp}")
                
                # Typische MT4-Timestamp-Formate
                if isinstance(sample_timestamp, (int, float)):
                    # Unix-Timestamp (Sekunden seit 1970-01-01)
                    self.df['Datetime'] = pd.to_datetime(self.df['Timestamp'], unit='s', errors='coerce')
                elif isinstance(sample_timestamp, str):
                    if '.' in sample_timestamp and len(sample_timestamp.split('.')[-1]) >= 3:
                        # Format mit Millisekunden wie "2023.04.28 12:30:45.000"
                        self.df['Datetime'] = pd.to_datetime(self.df['Timestamp'], format='%Y.%m.%d %H:%M:%S.%f', errors='coerce')
                    elif '.' in sample_timestamp:
                        # Format wie "2023.04.28 12:30:45"
                        self.df['Datetime'] = pd.to_datetime(self.df['Timestamp'], format='%Y.%m.%d %H:%M:%S', errors='coerce')
                    else:
                        # Fallback auf automatische Erkennung
                        self.df['Datetime'] = pd.to_datetime(self.df['Timestamp'], errors='coerce')
                else:
                    # Fallback auf automatische Erkennung
                    self.df['Datetime'] = pd.to_datetime(self.df['Timestamp'], errors='coerce')
                
                # Überprüfen, ob die Konvertierung erfolgreich war
                if not self.df['Datetime'].isnull().all():
                    # Fragen, ob die Konvertierung übernommen werden soll
                    convert = messagebox.askyesno("Timestamp gefunden",
                                            f"Die Spalte 'Timestamp' wurde gefunden und kann zu einem Datetime-Format konvertiert werden.\n"
                                            f"Möchten Sie die Konvertierung durchführen?")
                    if convert:
                        # Timestamp-Spalte an den Anfang bringen
                        dt_col = self.df.pop('Datetime')
                        self.df.insert(0, 'Datetime', dt_col)
                        # Originalspalte entfernen, falls gewünscht
                        remove_orig = messagebox.askyesno("Originalspalte entfernen?",
                                                   "Soll die ursprüngliche Timestamp-Spalte entfernt werden?")
                        if remove_orig:
                            self.df.drop(columns=['Timestamp'], inplace=True)
                        return
                    else:
                        # Konvertierte Spalte wieder entfernen
                        if 'Datetime' in self.df.columns:
                            del self.df['Datetime']
                else:
                    if 'Datetime' in self.df.columns:
                        del self.df['Datetime']
            except Exception as e:
                messagebox.showwarning("Konvertierungsproblem", 
                                     f"Die Timestamp-Spalte konnte nicht konvertiert werden: {str(e)}")
                if 'Datetime' in self.df.columns:
                    del self.df['Datetime']

        # Traditionelle Methode mit Datum/Uhrzeit-Spalten, falls keine Timestamp vorhanden
        date_cols = [col for col in self.df.columns if col.lower() in ['datum', 'date']]
        time_cols = [col for col in self.df.columns if col.lower() in ['uhrzeit', 'zeit', 'time']]

        if date_cols and time_cols:
            date_col = date_cols[0] # Nimm die erste gefundene
            time_col = time_cols[0] # Nimm die erste gefundene

            merge = messagebox.askyesno("Datum/Uhrzeit gefunden",
                                        f"Spalten '{date_col}' und '{time_col}' gefunden.\n"
                                        f"Sollen diese zu einer 'Datetime'-Spalte zusammengeführt werden?\n"
                                        f"(Die Originalspalten werden danach entfernt)")

            if merge:
                self.set_status(f"Führe '{date_col}' und '{time_col}' zusammen...")
                try:
                    # Versuche, die Spalten als Strings zu kombinieren
                    datetime_str = self.df[date_col].astype(str) + ' ' + self.df[time_col].astype(str)

                    # Versuche, das kombinierte Format zu parsen
                    # Hier könnten bei Bedarf flexiblere Formate oder Benutzereingaben hinzugefügt werden
                    self.df['Datetime'] = pd.to_datetime(datetime_str, errors='coerce', dayfirst=True) # dayfirst=True für TT.MM.JJJJ

                    # Überprüfen, ob das Parsen fehlgeschlagen ist (NaT-Werte erzeugt)
                    if self.df['Datetime'].isnull().all():
                         # Fallback oder alternativer Versuch ohne explizites Format
                         self.df['Datetime'] = pd.to_datetime(datetime_str, errors='coerce', format='mixed')

                    # Erneute Prüfung nach Fallback
                    if self.df['Datetime'].isnull().any():
                        failed_count = self.df['Datetime'].isnull().sum()
                        if failed_count == len(self.df):
                            messagebox.showerror("Fehler beim Zusammenführen",
                                                 "Konnte Datum und Uhrzeit nicht kombinieren. Überprüfen Sie das Format in den Spalten.")
                            # Mache die Änderung rückgängig, falls alles fehlschlug
                            if 'Datetime' in self.df.columns: del self.df['Datetime']
                            return # Nicht weitermachen mit dem Löschen der Originale
                        else:
                             messagebox.showwarning("Warnung beim Zusammenführen",
                                                 f"{failed_count} von {len(self.df)} Datum/Zeit-Werten konnten nicht korrekt interpretiert werden (NaT).")


                    # Originalspalten löschen und Datetime an den Anfang stellen
                    cols_to_drop = [date_col, time_col]
                    self.df.drop(columns=cols_to_drop, inplace=True)
                    # 'Datetime' Spalte nach vorne bringen
                    dt_col = self.df.pop('Datetime')
                    self.df.insert(0, 'Datetime', dt_col)

                    self.set_status(f"'{date_col}' und '{time_col}' zu 'Datetime' zusammengeführt.")
                    filename = os.path.basename(self.filepath) if self.filepath else "Unbekannte Datei"
                    self.file_info_label.config(text=f"{filename} ({self.df.shape[0]} Zeilen, {self.df.shape[1]} Spalten)")


                except Exception as e:
                    messagebox.showerror("Fehler beim Zusammenführen", f"Fehler beim Konvertieren zu Datetime:\n{str(e)}")
                    # Sicherstellen, dass eine halbfertige 'Datetime'-Spalte entfernt wird
                    if 'Datetime' in self.df.columns: del self.df['Datetime']


    def populate_ui_after_load(self):
        if self.df is None:
            self.clear_ui()
            return

        # 1. Spaltenliste füllen
        self.columns_listbox.delete(0, tk.END)
        self.column_order_in_listbox = list(self.df.columns) # Initiale Reihenfolge aus DF
        for col in self.column_order_in_listbox:
            self.columns_listbox.insert(tk.END, col)
        self.select_all() # Standardmäßig alle auswählen

        # 2. Dezimalstellen-Einstellungen aktualisieren
        # Alte Widgets entfernen
        for widget in self.decimal_settings_frame.winfo_children():
            widget.destroy()
        self.decimal_widgets.clear()

        # Neue Widgets für numerische Spalten erstellen
        row_idx = 0
        numeric_cols = self.df.select_dtypes(include='number').columns
        if not numeric_cols.empty:
            # Header-Frame für Überschriften und den "Alle setzen"-Button
            header_frame = ttk.Frame(self.decimal_settings_frame)
            header_frame.grid(row=row_idx, column=0, columnspan=3, sticky="ew", padx=5, pady=2)
            
            # Überschriften im Header-Frame
            ttk.Label(header_frame, text="Spalte", width=15).pack(side=tk.LEFT, padx=(0, 10))
            ttk.Label(header_frame, text="Stellen", width=8).pack(side=tk.LEFT, padx=(0, 10))
            
            # Button für alle Dezimalstellen auf einmal setzen
            ttk.Button(header_frame, text="Alle setzen", 
                     command=self.set_all_decimals).pack(side=tk.LEFT, padx=5)
            
            row_idx += 1
            
            # Trennlinie
            separator = ttk.Separator(self.decimal_settings_frame, orient='horizontal')
            separator.grid(row=row_idx, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
            row_idx += 1
            
            # Für jede numerische Spalte ein Eingabefeld erstellen
            for col in self.column_order_in_listbox: # Reihenfolge aus Listbox beibehalten
                if col in numeric_cols:
                   # Frame für jede Zeile
                   row_frame = ttk.Frame(self.decimal_settings_frame)
                   row_frame.grid(row=row_idx, column=0, columnspan=3, sticky="ew", padx=5, pady=2)
                   
                   # Dezimalstellen bestimmen (automatisch erkennen)
                   try:
                       sample = self.df[col].dropna().iloc[0] if not self.df[col].empty and len(self.df[col].dropna()) > 0 else 0
                       decimal_places = len(str(sample).split('.')[-1]) if '.' in str(sample) else 2
                       decimal_places = min(decimal_places, 6)  # Max. 6 Stellen
                   except Exception as e:
                       self.set_status(f"Info: Konnte automatische Dezimalstellen für {col} nicht bestimmen: {str(e)}")
                       decimal_places = 2  # Fallback
                    
                   # Spaltenname mit fester Breite
                   name_label = ttk.Label(row_frame, text=f"{col}:", width=15, anchor="e")
                   name_label.pack(side=tk.LEFT, padx=(0, 10))
                   
                   # Spinbox für Dezimalstellen
                   entry = ttk.Spinbox(row_frame, width=5, from_=0, to=10, 
                                     increment=1, wrap=True)
                   entry.insert(0, str(decimal_places)) # Automatisch erkannte Anzahl Nachkommastellen
                   entry.pack(side=tk.LEFT, padx=(0, 10))
                   self.decimal_widgets[col] = entry
                   
                   # Beispielwert anzeigen
                   try:
                       sample_value = self.df[col].dropna().iloc[0] if not self.df[col].empty and len(self.df[col].dropna()) > 0 else 0
                       # Formatiere den Wert entsprechend der eingestellten Dezimalstellen
                       formatted_value = "{:.{prec}f}".format(float(sample_value), prec=decimal_places)
                       example_label = ttk.Label(row_frame, text=f"z.B.: {formatted_value}")
                   except Exception as e:
                       self.set_status(f"Info: Formatierungsproblem bei {col}: {str(e)}")
                       example_label = ttk.Label(row_frame, text="z.B.: ---")
                   
                   example_label.pack(side=tk.LEFT)
                   
                   row_idx += 1
        else:
             ttk.Label(self.decimal_settings_frame, text="Keine numerischen Spalten gefunden.").grid(row=0, column=0, padx=5, pady=5, columnspan=3)


        # 3. Datenvorschau aktualisieren
        self.update_data_preview(self.df.head(50)) # Zeige erste 50 Zeilen
        
        # 4. Titel des Datenvorschau-Frames aktualisieren
        preview_frame_text = f"Datenvorschau (50 von {len(self.df)} Zeilen)"
        for child in self.right_frame.winfo_children():
            if isinstance(child, ttk.LabelFrame):
                child.configure(text=preview_frame_text)

    def clear_ui(self):
        """Setzt die UI-Elemente zurück, die vom DataFrame abhängen."""
        # Mousewheel-Binding entfernen, um Fehler zu vermeiden
        try:
            self.root.unbind_all("<MouseWheel>")
        except:
            pass
            
        self.columns_listbox.delete(0, tk.END)
        self.column_order_in_listbox = []
        for widget in self.decimal_settings_frame.winfo_children():
            widget.destroy()
        self.decimal_widgets.clear()
        self.clear_data_preview()
        self.export_button.config(state=tk.DISABLED)
        self.up_button.config(state=tk.DISABLED)
        self.down_button.config(state=tk.DISABLED)
        self.select_all_button.config(state=tk.DISABLED)
        self.deselect_all_button.config(state=tk.DISABLED)
        self.refresh_preview_button.config(state=tk.DISABLED)
        self.file_info_label.config(text="Keine Datei geladen.")
        
        # Titel der Datenvorschau zurücksetzen
        for child in self.right_frame.winfo_children():
            if isinstance(child, ttk.LabelFrame):
                child.configure(text="Datenvorschau (keine Daten geladen)")


    def update_data_preview(self, preview_df):
        self.clear_data_preview()
        if preview_df is None or preview_df.empty:
            return

        cols = list(preview_df.columns)
        self.preview_tree["columns"] = cols
        self.preview_tree["show"] = "headings"

        for col in cols:
            self.preview_tree.heading(col, text=col, anchor=tk.W)
            
            # Optimale Spaltenbreite basierend auf Überschrift und Daten berechnen
            max_width = len(col) * 8  # Basisbreite für die Überschrift
            
            # Beispielwerte zur Breitenberechnung verwenden
            sample_values = preview_df[col].dropna().head(10).astype(str)
            if not sample_values.empty:
                # Max. Länge der Beispielwerte berechnen (begrenzt auf 50 Zeichen)
                sample_width = min(50, max([len(str(v)) for v in sample_values])) * 7
                max_width = max(max_width, sample_width)
            
            self.preview_tree.column(col, anchor=tk.W, width=max_width, stretch=tk.NO)

            # Spezielle Formatierung für MACD_Cross
            if col == 'MACD_Cross':
                # Farbliche Markierung einrichten
                self.preview_tree.tag_configure('bullish', background='#d4ffcc')  # Leichtes Grün
                self.preview_tree.tag_configure('bearish', background='#ffccd4')  # Leichtes Rot

        # Daten einfügen (als Strings konvertieren für die Anzeige)
        for index, row in preview_df.iterrows():
            # Konvertiere jede Zelle sicher in einen String
            # Falls die Zelle bereits ein String mit formatierter Zahl ist, nutze diesen direkt
            values = []
            for col in cols:
                cell_value = row[col]
                if pd.notna(cell_value):
                    # Überprüfen, ob die Zelle bereits ein formatierter String ist
                    if isinstance(cell_value, str) and any(c.isdigit() for c in cell_value):
                        values.append(cell_value)
                    else:
                        values.append(str(cell_value))
                else:
                    values.append("")
            
            # Prüfen auf MACD_Cross Werte für farbliche Markierung
            item_tags = ()
            if 'MACD_Cross' in preview_df.columns:
                macd_idx = cols.index('MACD_Cross')
                try:
                    # Wenn der Wert bereits als String formatiert ist, müssen wir ihn zuerst zurückkonvertieren
                    macd_cell = row['MACD_Cross']
                    macd_value = int(float(macd_cell)) if isinstance(macd_cell, str) else int(macd_cell)
                    if macd_value == 1:
                        item_tags = ('bullish',)
                    elif macd_value == -1:
                        item_tags = ('bearish',)
                except (ValueError, TypeError):
                    pass
                    
            self.preview_tree.insert("", tk.END, values=values, tags=item_tags)


    def clear_data_preview(self):
        # Lösche alte Spalten und Daten
        self.preview_tree.delete(*self.preview_tree.get_children())
        if isinstance(self.preview_tree["columns"], (list, tuple)): # Prüfen ob Spalten gesetzt sind
             for col in self.preview_tree["columns"]:
                 # Versuch, die Spaltendefinition zu löschen (kann Fehler werfen, wenn nicht vorhanden)
                try:
                    self.preview_tree.heading(col, text="")
                    self.preview_tree.column(col, width=0) # Verstecken statt löschen
                except tk.TclError:
                    pass # Spalte existiert evtl. nicht mehr
        self.preview_tree["columns"] = []



    def move_up(self):
        selections = self.columns_listbox.curselection()
        if not selections:
            return

        # Sortiere die Indizes, damit wir von oben nach unten arbeiten
        selections = sorted(list(selections))

        for i in selections:
            if i > 0: # Kann nicht das oberste Element nach oben verschieben
                # Tausche in der Datenstruktur (column_order_in_listbox)
                self.column_order_in_listbox[i], self.column_order_in_listbox[i-1] = \
                    self.column_order_in_listbox[i-1], self.column_order_in_listbox[i]
                # Tausche in der Listbox-Anzeige
                text = self.columns_listbox.get(i)
                self.columns_listbox.delete(i)
                self.columns_listbox.insert(i-1, text)
                # Aktualisiere die Selektion, damit sie mit dem Element wandert
                self.columns_listbox.selection_set(i-1)
                # Alte Selektion aufheben (wichtig bei Mehrfachauswahl)
                self.columns_listbox.selection_clear(i+1 if i+1 < self.columns_listbox.size() else i)


    def move_down(self):
        selections = self.columns_listbox.curselection()
        if not selections:
            return

        # Sortiere die Indizes rückwärts, damit wir von unten nach oben arbeiten
        selections = sorted(list(selections), reverse=True)

        for i in selections:
            if i < self.columns_listbox.size() - 1: # Kann nicht das unterste Element nach unten verschieben
                # Tausche in der Datenstruktur (column_order_in_listbox)
                self.column_order_in_listbox[i], self.column_order_in_listbox[i+1] = \
                    self.column_order_in_listbox[i+1], self.column_order_in_listbox[i]
                # Tausche in der Listbox-Anzeige
                text = self.columns_listbox.get(i)
                self.columns_listbox.delete(i)
                self.columns_listbox.insert(i+1, text)
                # Aktualisiere die Selektion
                self.columns_listbox.selection_set(i+1)
                # Alte Selektion aufheben
                self.columns_listbox.selection_clear(i-1 if i-1 >= 0 else i)


    # --- Hilfsfunktionen für Selektion ---
    def select_all(self):
        self.columns_listbox.selection_set(0, tk.END)

    def deselect_all(self):
         self.columns_listbox.selection_clear(0, tk.END)

    def set_all_decimals(self):
        """Setzt alle Dezimalstellen-Einstellungen auf einen Wert."""
        if not self.decimal_widgets:
            return

        # Dialog zum Eingeben des Wertes
        dialog = tk.Toplevel(self.root)
        dialog.title("Nachkommastellen einstellen")
        dialog.geometry("250x120")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()  # Modal machen
        
        # Zentrieren im Hauptfenster
        x = self.root.winfo_x() + self.root.winfo_width() // 2 - 125
        y = self.root.winfo_y() + self.root.winfo_height() // 2 - 60
        dialog.geometry(f"+{x}+{y}")
        
        # UI-Elemente
        ttk.Label(dialog, text="Anzahl der Nachkommastellen für alle\nnumerischen Spalten:").pack(pady=10)
        
        value_var = tk.StringVar(value="2")
        spinner = ttk.Spinbox(dialog, from_=0, to=10, increment=1, width=5, textvariable=value_var)
        spinner.pack(pady=5)
        
        def apply_and_close():
            try:
                decimal_places = int(value_var.get())
                for widget in self.decimal_widgets.values():
                    widget.delete(0, tk.END)
                    widget.insert(0, str(decimal_places))
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Fehler", "Bitte geben Sie eine gültige Zahl ein.")

        ttk.Button(dialog, text="OK", command=apply_and_close).pack(pady=5)


    def show_visualization_options(self):
        """Zeigt Optionen zur Visualisierung der Forex-Daten an."""
        if self.df is None:
            messagebox.showwarning("Keine Daten", "Bitte zuerst eine CSV-Datei laden.")
            return
            
        # Prüfen, ob Pandas verfügbar ist (sollte es sein)
        try:
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except ImportError:
            messagebox.showwarning("Modul fehlt", 
                                 "Die Visualisierung erfordert matplotlib. "
                                 "Bitte installieren Sie es mit 'pip install matplotlib'.")
            return
        
        # Neues Fenster für Visualisierung erstellen
        vis_win = tk.Toplevel(self.root)
        vis_win.title("Forex Daten Visualisierung")
        vis_win.geometry("1000x600")
        
        # Rahmen für Steuerelemente
        control_frame = ttk.Frame(vis_win, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Dropdown für Visualisierungstyp
        ttk.Label(control_frame, text="Visualisierungstyp:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        vis_type_var = tk.StringVar(value="Candlestick mit MACD")
        vis_combo = ttk.Combobox(control_frame, textvariable=vis_type_var, 
                               values=["Candlestick mit MACD", "RSI und Momentum", "OHLC mit Moving Averages"])
        vis_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Frame für Zeitbereichsauswahl
        time_frame = ttk.LabelFrame(control_frame, text="Zeitbereich", padding="5")
        time_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        # Start und Ende
        ttk.Label(time_frame, text="Start:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Label(time_frame, text="Ende:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        
        # Hier sollte eigentlich ein DateEntry Widget sein, aber wir simulieren es mit Comboboxen
        # In einer vollständigen Implementierung würde ich das tkcalendar-Paket nutzen
        
        # Platzhalter für Visualisierung (wird später mit aktuellem Plot ersetzt)
        fig_frame = ttk.Frame(vis_win)
        fig_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Dummy-Plot als Platzhalter
        fig = Figure(figsize=(10, 6), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=fig_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Button zum Aktualisieren der Visualisierung
        ttk.Button(control_frame, text="Visualisierung aktualisieren", 
                  command=lambda: self.update_visualization(fig, canvas, vis_type_var.get())).grid(
                      row=0, column=2, rowspan=2, padx=10, pady=10, sticky="nsew")
        
        # Initial visualisieren
        self.update_visualization(fig, canvas, vis_type_var.get())
        
    def update_visualization(self, fig, canvas, vis_type):
        """Aktualisiert die Visualisierung basierend auf dem ausgewählten Typ."""
        
        # Prüfen ob die benötigten Daten existieren
        req_cols = ['Close']
        timestamp_candidates = ['Timestamp', 'Datetime', 'Zeit', 'Date']
        
        # Überprüfen, ob self.df existiert und mindestens eine Timestamp-Spalte vorhanden ist
        has_timestamp = False
        if self.df is not None:
            has_timestamp = any(col in self.df.columns for col in timestamp_candidates)
        
        if self.df is None or not has_timestamp or 'Close' not in self.df.columns:
            messagebox.showwarning("Fehlende Daten", 
                                "Die für diese Visualisierung benötigten Spalten (Zeitstempel und Close) sind nicht vorhanden.")
            return
            
        # DataFrame für Visualisierung vorbereiten
        vis_df = self.df.copy()
        
        # Timestamp-Spalte bestimmen
        timestamp_col = next((col for col in timestamp_candidates if col in vis_df.columns), None)
        
        # Datetime-Konvertierung versuchen
        if timestamp_col and timestamp_col != 'Datetime':
            try:
                vis_df['Datetime'] = pd.to_datetime(vis_df[timestamp_col], errors='coerce')
                # Prüfen ob Konvertierung erfolgreich war
                if not vis_df['Datetime'].isnull().all():
                    timestamp_col = 'Datetime'
            except Exception as e:
                print(f"Fehler bei der Datetime-Konvertierung: {str(e)}")
                # Behalte das Original, wenn die Konvertierung fehlschlägt
        
        # Figure leeren
        fig.clear()
        
        # Je nach ausgewähltem Visualisierungstyp plotten
        if 'MACD' in vis_type:
            self.plot_candlestick_with_macd(fig, vis_df, timestamp_col)
        elif 'RSI' in vis_type:
            self.plot_rsi_momentum(fig, vis_df, timestamp_col)
        else:
            self.plot_ohlc_with_ma(fig, vis_df, timestamp_col)
            
        # Canvas aktualisieren
        canvas.draw()
            
    def plot_candlestick_with_macd(self, fig, df, timestamp_col):
        """Plot mit Candlestick und MACD erstellen."""
        # In einer echten Implementierung würden hier mpl_finance oder mplfinance
        # für Candlestick-Charts verwendet werden
        
        # Hier nur ein einfaches Beispiel:
        ax1 = fig.add_subplot(211)  # Oberer Plot
        ax2 = fig.add_subplot(212)  # Unterer Plot
        
        # Preisdaten plotten
        ax1.plot(df[timestamp_col], df['Close'], label='Close')
        
        # SMA hinzufügen, falls vorhanden
        if 'SMA20' in df.columns and 'SMA50' in df.columns:
            ax1.plot(df[timestamp_col], df['SMA20'], label='SMA20')
            ax1.plot(df[timestamp_col], df['SMA50'], label='SMA50')
        
        ax1.set_title('Preisdaten')
        ax1.legend()
        
        # MACD plotten
        if 'MACD_Main' in df.columns and 'MACD_Signal' in df.columns:
            ax2.plot(df[timestamp_col], df['MACD_Main'], label='MACD')
            ax2.plot(df[timestamp_col], df['MACD_Signal'], label='Signal')
            
            # MACD-Kreuzungen hervorheben
            if 'MACD_Cross' in df.columns:
                # Bullische Kreuzungen (1)
                bullish = df[df['MACD_Cross'] == 1]
                if not bullish.empty:
                    ax2.scatter(bullish[timestamp_col], bullish['MACD_Main'], 
                             color='green', marker='^', s=100, label='Bullisch')
                
                # Bärische Kreuzungen (-1)
                bearish = df[df['MACD_Cross'] == -1]
                if not bearish.empty:
                    ax2.scatter(bearish[timestamp_col], bearish['MACD_Main'], 
                             color='red', marker='v', s=100, label='Bärisch')
        
        ax2.set_title('MACD')
        ax2.legend()
        
        fig.tight_layout()
    
    def plot_rsi_momentum(self, fig, df, timestamp_col):
        """Plot mit RSI und Momentum erstellen."""
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        # RSI plotten
        if 'RSI' in df.columns:
            ax1.plot(df[timestamp_col], df['RSI'], color='purple')
            ax1.axhline(y=30, color='green', linestyle='-')
            ax1.axhline(y=70, color='red', linestyle='-')
            ax1.set_title('RSI (14)')
            ax1.set_ylim(0, 100)
        
        # Momentum plotten
        if 'Momentum' in df.columns:
            ax2.plot(df[timestamp_col], df['Momentum'], color='blue')
            ax2.axhline(y=100, color='black', linestyle='-')
            ax2.set_title('Momentum (14)')
        
        fig.tight_layout()
        
    def plot_ohlc_with_ma(self, fig, df, timestamp_col):
        """Plot mit OHLC-Daten und Moving Averages erstellen."""
        ax = fig.add_subplot(111)
        
        # Close-Preis plotten
        ax.plot(df[timestamp_col], df['Close'], label='Close')
        
        # Moving Averages plotten
        if 'SMA20' in df.columns:
            ax.plot(df[timestamp_col], df['SMA20'], label='SMA20')
        if 'SMA50' in df.columns:
            ax.plot(df[timestamp_col], df['SMA50'], label='SMA50')
        
        # Bollinger Bands plotten
        if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
            ax.plot(df[timestamp_col], df['BB_Upper'], 'r--', label='BB Upper')
            ax.plot(df[timestamp_col], df['BB_Lower'], 'r--', label='BB Lower')
            # Füllung zwischen den Bändern
            ax.fill_between(df[timestamp_col], df['BB_Lower'], df['BB_Upper'], alpha=0.1, color='gray')
        
        ax.set_title('OHLC mit Moving Averages und Bollinger Bands')
        ax.legend()
        
        fig.tight_layout()

    def refresh_data_preview(self):
        """Aktualisiert die Datenvorschau mit der aktuellen Anzahl von Zeilen, 
        berücksichtigt ausgewählte Spalten und Dezimalstellen."""
        if self.df is None:
            messagebox.showwarning("Keine Daten", "Bitte zuerst eine CSV-Datei laden.")
            return
            
        try:
            # Anzahl der Zeilen aus dem Dropdown holen
            rows_to_show = int(self.preview_rows_var.get())
            # Begrenze auf die tatsächliche Anzahl von Zeilen im DataFrame
            rows_to_show = min(rows_to_show, len(self.df))
            
            # Ausgewählte Spalten in der aktuellen Reihenfolge aus der Listbox holen
            selected_indices = self.columns_listbox.curselection()
            if not selected_indices:
                # Wenn keine Spalten ausgewählt sind, zeige alle an
                preview_df = self.df.head(rows_to_show)
            else:
                # Nur die ausgewählten Spalten in der gewünschten Reihenfolge anzeigen
                selected_cols = [self.columns_listbox.get(i) for i in selected_indices]
                preview_df = self.df[selected_cols].head(rows_to_show)
            
            # Dezimalstellen formatieren (nur für die Anzeige, nicht im DataFrame selbst)
            formatted_df = preview_df.copy()
            for col in formatted_df.select_dtypes(include='number').columns:
                if col in self.decimal_widgets:
                    try:
                        dec_places_str = self.decimal_widgets[col].get().strip()
                        if dec_places_str:  # Falls nicht leer
                            dec_places = int(dec_places_str)
                            # Bei 0 Nachkommastellen Ganzzahl-Format verwenden, sonst Dezimalformat
                            if dec_places == 0:
                                formatted_df[col] = formatted_df[col].apply(
                                    lambda x: f"{int(x)}" if pd.notna(x) else ""
                                )
                            else:
                                formatted_df[col] = formatted_df[col].apply(
                                    lambda x: f"{x:.{dec_places}f}" if pd.notna(x) else ""
                                )
                    except (ValueError, TypeError) as e:
                        self.set_status(f"Warnung: Konnte Dezimalstellen für {col} nicht anwenden: {str(e)}")
            
            self.set_status(f"Aktualisiere Datenvorschau mit {rows_to_show} Zeilen...")
            self.update_data_preview(formatted_df)
            
            # Status-Update mit Spalteninfo
            spalteninfo = f"{len(selected_indices) if selected_indices else len(self.df.columns)} Spalten"
            # Status-Update mit Details
            self.set_status(f"Datenvorschau aktualisiert: {rows_to_show} Zeilen, {spalteninfo} angezeigt.")
            
            # Titel der Datenvorschau aktualisieren
            preview_frame_text = f"Datenvorschau ({rows_to_show} von {len(self.df)} Zeilen)"
            for child in self.right_frame.winfo_children():
                if isinstance(child, ttk.LabelFrame):
                    child.configure(text=preview_frame_text)
        except ValueError:
            messagebox.showerror("Fehler", "Ungültige Anzahl von Zeilen. Bitte geben Sie eine ganze Zahl ein.")
            self.preview_rows_var.set("50")  # Zurück zum Standardwert
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Aktualisieren der Vorschau: {str(e)}")

    def export_csv(self):
        if self.df is None:
            messagebox.showwarning("Keine Daten", "Bitte zuerst eine CSV-Datei laden.")
            return

        # 1. Ausgewählte Spalten in der aktuellen Reihenfolge aus der Listbox holen
        selected_indices = self.columns_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Keine Auswahl", "Bitte wählen Sie mindestens eine Spalte für den Export aus.")
            return
        # Wichtig: Holen die Spaltennamen basierend auf der Reihenfolge in der Listbox!
        selected_cols_in_order = [self.columns_listbox.get(i) for i in selected_indices]


        # 2. Speicherort abfragen
        export_path = filedialog.asksaveasfilename(
            title="Ausgewählte Daten speichern unter...",
            defaultextension=".csv",
            filetypes=[("CSV Dateien", "*.csv"), ("Textdateien", "*.txt"), ("Alle Dateien", "*.*")],
            initialfile=f"export_{os.path.basename(self.filepath)}" if self.filepath else "export.csv"
        )
        if not export_path:
            return

        # 3. Export-Einstellungen holen
        export_sep = self.export_separator_var.get()
        if export_sep == '\\t': export_sep = '\t'
        export_dec = self.export_decimal_var.get()

        self.set_status(f"Exportiere nach {os.path.basename(export_path)}...")

        try:
            # 4. DataFrame für den Export vorbereiten
            # Nur die ausgewählten Spalten in der gewünschten Reihenfolge nehmen
            export_df = self.df[selected_cols_in_order].copy()
            
            # MACD-Signal-Filter anwenden, falls aktiviert
            if hasattr(self, 'macd_signal_var') and self.macd_signal_var.get() and 'MACD_Cross' in export_df.columns:
                pre_filter_count = len(export_df)
                export_df = export_df[export_df['MACD_Cross'] != 0].copy()
                post_filter_count = len(export_df)
                
                self.set_status(f"MACD-Signal-Filter angewendet: {pre_filter_count - post_filter_count} Einträge gefiltert")
                if post_filter_count == 0:
                    messagebox.showwarning("Keine Daten", 
                                          "Nach Anwendung des MACD-Signal-Filters sind keine Daten mehr übrig.\n"
                                          "Der Export wird abgebrochen.")
                    return

            # 5. Dezimalstellen anwenden (auf numerische Spalten im Export-DF)
            rounded_cols_count = 0
            for col in export_df.select_dtypes(include='number').columns:
                if col in self.decimal_widgets:
                    try:
                        dec_places_str = self.decimal_widgets[col].get().strip()
                        if dec_places_str:  # Falls nicht leer
                            dec_places = int(dec_places_str)
                            if dec_places > 0:
                                # Wende Rundung auf die Spalte an
                                export_df[col] = export_df[col].round(dec_places)
                                rounded_cols_count += 1
                            elif dec_places == 0:
                                # Bei 0 Nachkommastellen zu Ganzzahl konvertieren
                                export_df[col] = export_df[col].astype(int)
                                rounded_cols_count += 1
                    except ValueError as e:
                        # Nur eine stille Warnung in die Statusleiste, nicht als Popup
                        self.set_status(f"Info: Konnte Dezimalstellen für {col} nicht anwenden: {str(e)}")
                    except Exception as e:
                        # Allgemeine Ausnahmen fangen, aber den Export nicht abbrechen
                        self.set_status(f"Warnung bei Dezimalrundung für Spalte {col}: {str(e)}")
            
            # Meldung über erfolgreiche Rundung
            if rounded_cols_count > 0:
                self.set_status(f"{rounded_cols_count} Spalten mit den angegebenen Dezimalstellen gerundet.")
            else:
                self.set_status("Keine Spalten wurden gerundet.")

            # 6. Speichern
            export_df.to_csv(
                export_path,
                sep=export_sep,
                decimal=export_dec,
                index=False,          # Normalerweise keinen Index mitspeichern
                encoding='utf-8-sig' # Gut für Kompatibilität mit Excel (bzgl. Umlauten etc.)
            )
            self.set_status(f"Datei erfolgreich gespeichert: {os.path.basename(export_path)}")
            messagebox.showinfo("Erfolg", f"Ausgewählte Daten wurden erfolgreich nach\n{export_path}\ngespeichert.")

        except Exception as e:
            self.set_status("Fehler beim Export.")
            messagebox.showerror("Exportfehler", f"Ein Fehler ist beim Speichern der Datei aufgetreten:\n{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = CSVToolApp(root)
    root.mainloop()
