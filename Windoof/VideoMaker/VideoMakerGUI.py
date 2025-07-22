#!/usr/bin/env python3
"""
VideoMaker GUI - Convert Plot Images to Video
===========================================

Ein Tool zum Erstellen von Videos aus Plot-Bildern mit:
- GUI ähnlich dem PlotCollector
- Konfigurierbare FPS
- ID-Einblendung in roter Schrift (oben rechts)
- Automatische Bildgrößen-Anpassung
- Progress-Tracking

Author: Trading Analysis System
Date: Juli 2025
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import re
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import threading
import time

class VideoMaker:
    """Core Video Creation Logic"""
    
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
    
    def extract_id_from_filename(self, filename: str) -> Optional[str]:
        """Extrahiert 6-stellige ID aus Dateinamen."""
        # Suche nach 6-stelligen Zahlen (Timestamp-basierte IDs)
        patterns = [
            r'(\d{6})',  # 6-stellige Zahl
            r'_(\d{6})_',  # 6-stellige Zahl zwischen Underscores
            r'(\d{6})\.',  # 6-stellige Zahl vor Dateierweiterung
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(1)
        
        # Fallback: Versuche beliebige Zahlen zu finden
        numbers = re.findall(r'\d+', filename)
        if numbers:
            # Nehme die längste Zahl als ID
            longest_number = max(numbers, key=len)
            if len(longest_number) >= 4:  # Mindestens 4 Stellen
                return longest_number[:6]  # Maximal 6 Stellen
        
        return None
    
    def get_image_files(self, source_dir: str, extensions: Optional[List[str]] = None) -> List[Tuple[str, str]]:
        """Holt alle Bilddateien mit IDs aus dem Quellverzeichnis."""
        if extensions is None:
            extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        
        image_files = []
        
        if not os.path.exists(source_dir):
            return image_files
        
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    full_path = os.path.join(root, file)
                    image_id = self.extract_id_from_filename(file)
                    if image_id:  # Nur Dateien mit erkennbarer ID
                        image_files.append((full_path, image_id))
        
        # Nach ID sortieren (numerisch)
        try:
            image_files.sort(key=lambda x: int(x[1]))
        except ValueError:
            # Fallback: alphabetisch sortieren
            image_files.sort(key=lambda x: x[1])
        
        return image_files
    
    def add_id_overlay(self, image: np.ndarray, image_id: str, font_scale: float = 1.0) -> np.ndarray:
        """Fügt ID-Overlay in roter Schrift oben rechts hinzu."""
        # Kopie des Bildes erstellen
        overlay_image = image.copy()
        height, width = overlay_image.shape[:2]
        
        # Font-Parameter
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = max(2, int(font_scale * 2))
        
        # Text-Größe berechnen
        (text_width, text_height), baseline = cv2.getTextSize(
            f"ID: {image_id}", font, font_scale, font_thickness
        )
        
        # Position oben rechts mit Margin
        margin = 20
        text_x = width - text_width - margin
        text_y = margin + text_height
        
        # Hintergrund-Box für bessere Lesbarkeit
        box_padding = 10
        box_x1 = text_x - box_padding
        box_y1 = text_y - text_height - box_padding
        box_x2 = text_x + text_width + box_padding
        box_y2 = text_y + box_padding
        
        # Semi-transparente schwarze Box
        cv2.rectangle(overlay_image, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
        
        # Transparenz anwenden
        alpha = 0.7
        cv2.addWeighted(overlay_image, alpha, image, 1 - alpha, 0, overlay_image)
        
        # Roten Text hinzufügen
        cv2.putText(
            overlay_image, 
            f"ID: {image_id}", 
            (text_x, text_y), 
            font, 
            font_scale, 
            (0, 0, 255),  # Rot in BGR
            font_thickness,
            cv2.LINE_AA
        )
        
        return overlay_image
    
    def create_video(self, source_dir: str, output_file: str, fps: int = 30, 
                    target_resolution: Optional[Tuple[int, int]] = None) -> bool:
        """Erstellt Video aus Bildern."""
        try:
            self.cancel_flag = False
            
            if self.status_callback:
                self.status_callback("🔍 Scanning for image files...")
            
            # Bilddateien sammeln
            image_files = self.get_image_files(source_dir)
            
            if not image_files:
                if self.status_callback:
                    self.status_callback("❌ No images with recognizable IDs found!")
                return False
            
            total_images = len(image_files)
            if self.status_callback:
                self.status_callback(f"📊 Found {total_images} images with IDs")
            
            # Erste Bildgröße ermitteln für Video-Setup
            first_image_path = image_files[0][0]
            first_image = cv2.imread(first_image_path)
            if first_image is None:
                if self.status_callback:
                    self.status_callback(f"❌ Cannot read first image: {first_image_path}")
                return False

            # KRITISCH: Bestimme die finale Video-Auflösung BEVOR Video Writer erstellt wird
            # Returns_Histograms haben oft unterschiedliche Breiten!
            
            # Analysiere alle Bildgrößen für konsistente Video-Auflösung
            if self.status_callback:
                self.status_callback("📐 Analyzing image dimensions for consistent video resolution...")
            
            image_dimensions = []
            sample_size = min(10, len(image_files))  # Analysiere erste 10 Bilder
            
            for i in range(sample_size):
                sample_path, _ = image_files[i]
                sample_img = cv2.imread(sample_path)
                if sample_img is not None:
                    h, w = sample_img.shape[:2]
                    image_dimensions.append((w, h))
            
            if not image_dimensions:
                if self.status_callback:
                    self.status_callback("❌ No valid images found!")
                return False
            
            # Bestimme optimale Video-Auflösung
            widths = [dim[0] for dim in image_dimensions]
            heights = [dim[1] for dim in image_dimensions]
            
            # Für Returns_Histograms: Nehme die größte gemeinsame Auflösung oder skaliere einheitlich
            max_width = max(widths)
            max_height = max(heights)
            min_width = min(widths)
            min_height = min(heights)
            
            if self.status_callback:
                self.status_callback(f"📊 Image dimensions analysis:")
                self.status_callback(f"   Width range: {min_width}px - {max_width}px")
                self.status_callback(f"   Height range: {min_height}px - {max_height}px")
            
            # Entscheide finale Video-Auflösung
            if max_width != min_width or max_height != min_height:
                # Unterschiedliche Größen -> Skaliere auf einheitliche Größe
                if any('returns_histogram' in f[0].lower() for f in image_files[:5]):
                    # Für Returns_Histograms: Intelligente Skalierung
                    if max_width > 3000 or max_height > 2000:
                        # Sehr hohe Auflösungen -> Skaliere auf 1920x1080
                        video_width = 1920
                        video_height = 1080
                    else:
                        # Moderate Auflösungen -> Verwende größte Dimensionen
                        video_width = max_width
                        video_height = max_height
                else:
                    # Für andere Plot-Typen: Verwende größte Dimensionen
                    video_width = max_width
                    video_height = max_height
                
                if self.status_callback:
                    self.status_callback(f"⚠️  Inconsistent image sizes detected!")
                    self.status_callback(f"📐 Will resize all images to: {video_width}x{video_height}")
            else:
                # Alle Bilder haben gleiche Größe
                video_width = max_width
                video_height = max_height
                if self.status_callback:
                    self.status_callback(f"✅ All images have consistent size: {video_width}x{video_height}")
            
            # Finale Überprüfung für Returns_Histograms
            if any('returns_histogram' in f[0].lower() for f in image_files[:5]):
                if video_width > 3000 or video_height > 2000:
                    # Auto-Skalierung für bessere Media Player Kompatibilität
                    max_width = 1920
                    max_height = 1080
                    scale_factor = min(max_width/video_width, max_height/video_height)
                    video_width = int(video_width * scale_factor)
                    video_height = int(video_height * scale_factor)
                    
                    if self.status_callback:
                        self.status_callback(f"📐 Auto-scaling Returns_Histograms to {video_width}x{video_height} for better compatibility")

            # Target Resolution anwenden wenn vom User gesetzt
            if target_resolution:
                video_width, video_height = target_resolution
                if self.status_callback:
                    self.status_callback(f"🎯 Using user-specified resolution: {video_width}x{video_height}")
            
            # Finale Video-Auflösung
            width, height = video_width, video_height
            
            if self.status_callback:
                self.status_callback(f"🎬 Creating video: {width}x{height} @ {fps}fps")
            
            # Video Writer mit universell kompatiblen Codecs
            # WICHTIG: Für Windows Media Player Kompatibilität priorisiere H.264 und XVID
            codecs_to_try = [
                ('H264', cv2.VideoWriter.fourcc(*'H264')),  # H.264 - beste universelle Kompatibilität
                ('XVID', cv2.VideoWriter.fourcc(*'XVID')),  # XVID - sehr Windows-kompatibel
                ('mp4v', cv2.VideoWriter.fourcc(*'mp4v')),  # MP4V - Standard MP4 Codec
                ('MJPG', cv2.VideoWriter.fourcc(*'MJPG')),  # Motion JPEG - kann Probleme haben
            ]
            
            # Für Returns_Histograms: Spezielle Behandlung wegen großer Datenmengen
            if any('returns_histogram' in f[0].lower() for f in image_files[:5]):
                if self.status_callback:
                    self.status_callback("🎬 Detected Returns_Histograms - using optimized codec selection")
                # Priorisiere Codecs die gut mit Histogramm-Daten umgehen
                codecs_to_try = [
                    ('H264', cv2.VideoWriter.fourcc(*'H264')),  # H.264 - beste Kompression für statische Inhalte
                    ('XVID', cv2.VideoWriter.fourcc(*'XVID')),  # XVID - sehr stabil
                    ('mp4v', cv2.VideoWriter.fourcc(*'mp4v')),  # MP4V - Fallback
                ]
            
            out = None
            successful_codec = None
            
            for codec_name, fourcc in codecs_to_try:
                if self.status_callback:
                    self.status_callback(f"🎬 Trying codec: {codec_name}")
                
                out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
                
                if out.isOpened():
                    successful_codec = codec_name
                    if self.status_callback:
                        self.status_callback(f"✅ Using codec: {codec_name}")
                    break
                else:
                    out.release()
            
            if not out or not out.isOpened():
                if self.status_callback:
                    self.status_callback(f"❌ Cannot create video file with any codec: {output_file}")
                return False
            
            # Font-Scale basierend auf Bildgröße
            font_scale = max(0.7, width / 1000)
            
            # Bilder verarbeiten
            for i, (image_path, image_id) in enumerate(image_files):
                if self.cancel_flag:
                    if self.status_callback:
                        self.status_callback("❌ Operation cancelled by user")
                    out.release()
                    # Unvollständige Datei löschen
                    if os.path.exists(output_file):
                        os.remove(output_file)
                    return False
                
                # Progress Update
                progress = (i + 1) / total_images * 100
                if self.progress_callback:
                    self.progress_callback(progress)
                
                if self.status_callback:
                    self.status_callback(f"🎬 Processing image {i+1}/{total_images}: ID {image_id}")
                
                # Bild laden
                image = cv2.imread(image_path)
                if image is None:
                    if self.status_callback:
                        self.status_callback(f"⚠️  Warning: Cannot read {image_path}")
                    continue
                
                original_height, original_width = image.shape[:2]
                
                # WICHTIG: Jedes Bild MUSS auf die finale Video-Auflösung skaliert werden!
                # Returns_Histograms haben oft unterschiedliche Breiten (4571px vs 2964px)
                if (original_width != width or original_height != height):
                    if self.status_callback and i < 5:  # Nur bei ersten 5 Bildern melden
                        self.status_callback(f"📐 Resizing from {original_width}x{original_height} to {width}x{height}")
                    
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                
                # Spezielle Behandlung für Returns_Histogram Bilder
                if 'returns_histogram' in os.path.basename(image_path).lower():
                    # Validiere, dass das Bild korrekt geladen wurde
                    if image.shape[0] == 0 or image.shape[1] == 0:
                        if self.status_callback:
                            self.status_callback(f"⚠️  Warning: Invalid histogram image dimensions: {image_path}")
                        continue
                    
                    # Konvertiere zu 8-bit falls nötig (Histogramme können 16-bit sein)
                    if image.dtype != np.uint8:
                        image = cv2.convertScaleAbs(image)
                    
                    # Stelle sicher, dass das Bild im korrekten BGR-Format ist
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        # Normalisiere für bessere Kompression
                        image = cv2.convertScaleAbs(image, alpha=1.0, beta=0)
                
                # Finale Validierung: Bild MUSS exakt die Video-Auflösung haben
                current_height, current_width = image.shape[:2]
                if current_width != width or current_height != height:
                    if self.status_callback:
                        self.status_callback(f"🔧 Final resize from {current_width}x{current_height} to {width}x{height}")
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                
                # ID-Overlay hinzufügen
                image_with_id = self.add_id_overlay(image, image_id, font_scale)
                
                # Frame zum Video hinzufügen
                out.write(image_with_id)
            
            # Video schließen
            out.release()
            
            # Video-Datei validieren
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                if file_size < 1024:  # Weniger als 1KB ist verdächtig
                    if self.status_callback:
                        self.status_callback(f"⚠️  Warning: Video file seems too small ({file_size} bytes)")
                else:
                    if self.status_callback:
                        self.status_callback(f"✅ Video created successfully: {output_file}")
                        self.status_callback(f"📊 Video file size: {file_size/1024/1024:.2f} MB")
                        self.status_callback(f"🎬 Used codec: {successful_codec}")
            else:
                if self.status_callback:
                    self.status_callback(f"❌ Video file was not created: {output_file}")
                return False
            
            return True
            
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"❌ Error creating video: {str(e)}")
            return False

class VideoMakerGUI:
    """GUI für den VideoMaker"""
    
    def __init__(self):
        self.video_maker = VideoMaker()
        self.video_maker.set_callbacks(self.update_progress, self.update_status)
        self.processing_thread = None
        self.setup_gui()
    
    def setup_gui(self):
        """GUI Setup"""
        self.root = tk.Tk()
        self.root.title("VideoMaker - Convert Images to Video")
        self.root.geometry("800x700")
        
        # Main Frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=tk.W+tk.E+tk.N+tk.S)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🎬 VideoMaker - Images to Video Converter", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Source Directory
        ttk.Label(main_frame, text="📁 Source Directory (Images):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.source_var = tk.StringVar()
        source_entry = ttk.Entry(main_frame, textvariable=self.source_var, width=60)
        source_entry.grid(row=1, column=1, sticky=tk.W+tk.E, padx=(10, 5), pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_source).grid(row=1, column=2, pady=5)
        
        # Output File
        ttk.Label(main_frame, text="🎥 Output Video File:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.output_var = tk.StringVar()
        output_entry = ttk.Entry(main_frame, textvariable=self.output_var, width=60)
        output_entry.grid(row=2, column=1, sticky=tk.W+tk.E, padx=(10, 5), pady=5)
        ttk.Button(main_frame, text="Save As", command=self.browse_output).grid(row=2, column=2, pady=5)
        
        # Video Settings Frame
        settings_frame = ttk.LabelFrame(main_frame, text="🎛️ Video Settings", padding="10")
        settings_frame.grid(row=3, column=0, columnspan=3, sticky=tk.W+tk.E, pady=10)
        settings_frame.columnconfigure(1, weight=1)
        
        # FPS Setting
        ttk.Label(settings_frame, text="🎬 FPS (Frames per Second):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.fps_var = tk.IntVar(value=10)
        fps_frame = ttk.Frame(settings_frame)
        fps_frame.grid(row=0, column=1, sticky=tk.W+tk.E, padx=10, pady=5)
        
        fps_scale = ttk.Scale(fps_frame, from_=1, to=60, variable=self.fps_var, orient=tk.HORIZONTAL)
        fps_scale.grid(row=0, column=0, sticky=tk.W+tk.E)
        fps_frame.columnconfigure(0, weight=1)
        
        self.fps_label = ttk.Label(fps_frame, text="10 FPS")
        self.fps_label.grid(row=0, column=1, padx=(10, 0))
        
        # FPS Scale Update
        fps_scale.configure(command=self.update_fps_label)
        
        # Resolution Setting
        ttk.Label(settings_frame, text="📐 Resolution:").grid(row=1, column=0, sticky=tk.W, pady=5)
        resolution_frame = ttk.Frame(settings_frame)
        resolution_frame.grid(row=1, column=1, sticky=tk.W+tk.E, padx=10, pady=5)
        
        self.resolution_var = tk.StringVar(value="original")
        resolution_combo = ttk.Combobox(resolution_frame, textvariable=self.resolution_var, 
                                       values=["original", "1920x1080", "1280x720", "854x480", "640x360"])
        resolution_combo.grid(row=0, column=0, sticky=tk.W+tk.E)
        resolution_combo.state(['readonly'])
        resolution_frame.columnconfigure(0, weight=1)
        
        # Image Extensions
        ttk.Label(settings_frame, text="🖼️ Image Types:").grid(row=2, column=0, sticky=tk.W, pady=5)
        extensions_frame = ttk.Frame(settings_frame)
        extensions_frame.grid(row=2, column=1, sticky=tk.W+tk.E, padx=10, pady=5)
        
        self.png_var = tk.BooleanVar(value=True)
        self.jpg_var = tk.BooleanVar(value=True)
        self.bmp_var = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(extensions_frame, text="PNG", variable=self.png_var).grid(row=0, column=0, padx=5)
        ttk.Checkbutton(extensions_frame, text="JPG", variable=self.jpg_var).grid(row=0, column=1, padx=5)
        ttk.Checkbutton(extensions_frame, text="BMP", variable=self.bmp_var).grid(row=0, column=2, padx=5)
        
        # Preview Frame
        preview_frame = ttk.LabelFrame(main_frame, text="📋 Preview", padding="10")
        preview_frame.grid(row=4, column=0, columnspan=3, sticky=tk.W+tk.E, pady=10)
        preview_frame.columnconfigure(0, weight=1)
        
        # Preview Button
        ttk.Button(preview_frame, text="🔍 Preview Images", command=self.preview_images).grid(row=0, column=0, pady=5)
        
        # Preview Text
        self.preview_text = tk.Text(preview_frame, height=8, width=80)
        self.preview_text.grid(row=1, column=0, sticky=tk.W+tk.E, pady=5)
        
        # Scrollbar for preview
        preview_scroll = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.preview_text.yview)
        preview_scroll.grid(row=1, column=1, sticky=tk.N+tk.S)
        self.preview_text.configure(yscrollcommand=preview_scroll.set)
        
        # Control Buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=5, column=0, columnspan=3, pady=20)
        
        self.create_button = ttk.Button(control_frame, text="🎬 Create Video", 
                                       command=self.create_video, style='Accent.TButton')
        self.create_button.grid(row=0, column=0, padx=5)
        
        self.cancel_button = ttk.Button(control_frame, text="❌ Cancel", 
                                       command=self.cancel_operation, state='disabled')
        self.cancel_button.grid(row=0, column=1, padx=5)
        
        ttk.Button(control_frame, text="📂 Open Output Folder", 
                  command=self.open_output_folder).grid(row=0, column=2, padx=5)
        
        # Progress Frame
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=6, column=0, columnspan=3, sticky=tk.W+tk.E, pady=10)
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=0, sticky=tk.W+tk.E, pady=5)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to create video from images")
        status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        status_label.grid(row=1, column=0, sticky=tk.W+tk.E)
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def update_fps_label(self, value):
        """FPS Label aktualisieren."""
        fps = int(float(value))
        self.fps_label.config(text=f"{fps} FPS")
    
    def browse_source(self):
        """Quellverzeichnis auswählen."""
        directory = filedialog.askdirectory(title="Select Source Directory with Images")
        if directory:
            self.source_var.set(directory)
            # Auto-suggest output filename
            if not self.output_var.get():
                base_name = os.path.basename(directory) or "video"
                output_path = os.path.join(directory, f"{base_name}_video.mp4")
                self.output_var.set(output_path)
    
    def browse_output(self):
        """Output-Datei auswählen."""
        file_path = filedialog.asksaveasfilename(
            title="Save Video As",
            defaultextension=".mp4",
            filetypes=[
                ("MP4 Video (H264)", "*.mp4"),
                ("AVI Video (XVID)", "*.avi"), 
                ("MOV Video (H264)", "*.mov"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            # Stelle sicher, dass die Dateiendung korrekt ist
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in ['.mp4', '.avi', '.mov']:
                file_path += '.mp4'  # Standard-Endung für beste Kompatibilität
            self.output_var.set(file_path)
    
    def get_selected_extensions(self):
        """Gewählte Dateierweiterungen zurückgeben."""
        extensions = []
        if self.png_var.get():
            extensions.extend(['.png'])
        if self.jpg_var.get():
            extensions.extend(['.jpg', '.jpeg'])
        if self.bmp_var.get():
            extensions.extend(['.bmp', '.tiff', '.tif'])
        return extensions if extensions else ['.png', '.jpg', '.jpeg']
    
    def get_target_resolution(self) -> Optional[Tuple[int, int]]:
        """Target Resolution aus Combobox."""
        resolution_str = self.resolution_var.get()
        if resolution_str == "original":
            return None
        
        try:
            width, height = map(int, resolution_str.split('x'))
            return (width, height)
        except ValueError:
            return None
    
    def preview_images(self):
        """Bilder-Preview anzeigen."""
        source_dir = self.source_var.get()
        if not source_dir or not os.path.exists(source_dir):
            messagebox.showerror("Error", "Please select a valid source directory first!")
            return
        
        extensions = self.get_selected_extensions()
        image_files = self.video_maker.get_image_files(source_dir, extensions)
        
        self.preview_text.delete(1.0, tk.END)
        
        if not image_files:
            self.preview_text.insert(tk.END, "❌ No images with recognizable IDs found in the selected directory!\n\n")
            self.preview_text.insert(tk.END, "Make sure your image files contain 6-digit IDs in their names.\n")
            self.preview_text.insert(tk.END, "Examples: 'plot_123456.png', 'image_123456_data.jpg', etc.")
            return
        
        self.preview_text.insert(tk.END, f"📊 Found {len(image_files)} images with IDs:\n")
        self.preview_text.insert(tk.END, "=" * 60 + "\n\n")
        
        for i, (file_path, image_id) in enumerate(image_files[:50], 1):  # Show first 50
            relative_path = os.path.relpath(file_path, source_dir)
            self.preview_text.insert(tk.END, f"{i:3d}. ID: {image_id:>6} - {relative_path}\n")
        
        if len(image_files) > 50:
            self.preview_text.insert(tk.END, f"\n... and {len(image_files) - 50} more images\n")
        
        self.preview_text.insert(tk.END, f"\n🎬 Video will be created with {self.fps_var.get()} FPS")
        target_res = self.get_target_resolution()
        if target_res:
            width, height = target_res
            self.preview_text.insert(tk.END, f" at {width}x{height} resolution")
        self.preview_text.insert(tk.END, "\n")
        
        # Scroll to top
        self.preview_text.see(1.0)
    
    def create_video(self):
        """Video erstellen."""
        # Validation
        source_dir = self.source_var.get()
        output_file = self.output_var.get()
        
        if not source_dir or not os.path.exists(source_dir):
            messagebox.showerror("Error", "Please select a valid source directory!")
            return
        
        if not output_file:
            messagebox.showerror("Error", "Please specify an output video file!")
            return
        
        # Check if output directory exists
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                messagebox.showerror("Error", f"Cannot create output directory: {str(e)}")
                return
        
        # Get settings
        fps = self.fps_var.get()
        target_resolution = self.get_target_resolution()
        
        # Disable UI during processing
        self.create_button.configure(state='disabled')
        self.cancel_button.configure(state='normal')
        self.progress_var.set(0)
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._create_video_thread,
            args=(source_dir, output_file, fps, target_resolution)
        )
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def _create_video_thread(self, source_dir: str, output_file: str, fps: int, target_resolution: Optional[Tuple[int, int]]):
        """Video-Erstellung in separatem Thread."""
        try:
            success = self.video_maker.create_video(source_dir, output_file, fps, target_resolution)
            
            # UI zurücksetzen
            self.root.after(0, self._video_creation_finished, success, output_file)
            
        except Exception as e:
            self.root.after(0, self._video_creation_error, str(e))
    
    def _video_creation_finished(self, success: bool, output_file: str):
        """Video-Erstellung abgeschlossen."""
        self.create_button.configure(state='normal')
        self.cancel_button.configure(state='disabled')
        
        if success:
            messagebox.showinfo("Success", f"Video created successfully!\n\n{output_file}")
        else:
            messagebox.showerror("Error", "Video creation failed! Check the status messages for details.")
    
    def _video_creation_error(self, error_msg: str):
        """Video-Erstellung Fehler."""
        self.create_button.configure(state='normal')
        self.cancel_button.configure(state='disabled')
        messagebox.showerror("Error", f"Video creation failed:\n\n{error_msg}")
    
    def cancel_operation(self):
        """Operation abbrechen."""
        if self.processing_thread and self.processing_thread.is_alive():
            self.video_maker.cancel_operation()
            self.update_status("🛑 Cancelling operation...")
    
    def open_output_folder(self):
        """Output-Ordner öffnen."""
        output_file = self.output_var.get()
        if output_file and os.path.exists(output_file):
            output_dir = os.path.dirname(output_file)
            try:
                os.startfile(output_dir)
            except OSError:
                # Fallback für andere Betriebssysteme
                import subprocess
                import sys
                if sys.platform == "darwin":  # macOS
                    subprocess.run(["open", output_dir])
                else:  # Linux
                    subprocess.run(["xdg-open", output_dir])
        elif output_file:
            output_dir = os.path.dirname(output_file)
            if os.path.exists(output_dir):
                try:
                    os.startfile(output_dir)
                except OSError:
                    messagebox.showinfo("Info", f"Output directory: {output_dir}")
            else:
                messagebox.showwarning("Warning", "Output directory does not exist yet!")
        else:
            messagebox.showwarning("Warning", "No output file specified!")
    
    def update_progress(self, value):
        """Progress Bar aktualisieren."""
        self.root.after(0, lambda: self.progress_var.set(value))
    
    def update_status(self, message):
        """Status-Nachricht aktualisieren."""
        self.root.after(0, lambda: self.status_var.set(message))
    
    def on_closing(self):
        """Beim Schließen des Fensters."""
        if self.processing_thread and self.processing_thread.is_alive():
            if messagebox.askokcancel("Quit", "Video creation is in progress. Do you want to cancel and quit?"):
                self.video_maker.cancel_operation()
                self.root.after(1000, self.root.destroy)  # Kurz warten, dann schließen
            else:
                return
        else:
            self.root.destroy()
    
    def run(self):
        """GUI starten."""
        self.root.mainloop()

def main():
    """Hauptfunktion mit Dependency-Check."""
    # Überprüfe ob OpenCV verfügbar ist
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
    except ImportError:
        print("❌ Error: OpenCV not found!")
        print("   Install with: pip install opencv-python")
        return
    
    # GUI starten
    try:
        app = VideoMakerGUI()
        app.run()
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
