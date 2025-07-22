#!/usr/bin/env python3
"""
Returns Histogram Diagnostic Tool
Vergleicht Returns_Histograms mit anderen Plots um den Unterschied zu finden
"""

import cv2
import os
import numpy as np
from pathlib import Path

def analyze_image_properties(image_path):
    """Analysiert die Eigenschaften eines Bildes detailliert"""
    print(f"\n🔍 Analyzing: {os.path.basename(image_path)}")
    
    if not os.path.exists(image_path):
        print("❌ File does not exist!")
        return None
    
    # Datei-Info
    file_size = os.path.getsize(image_path)
    print(f"📁 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    
    # Mit PIL laden (wie matplotlib speichert)
    try:
        from PIL import Image
        pil_img = Image.open(image_path)
        print(f"🖼️  PIL Info:")
        print(f"   Mode: {pil_img.mode}")
        print(f"   Size: {pil_img.size}")
        print(f"   Format: {pil_img.format}")
        if hasattr(pil_img, 'info'):
            print(f"   Info: {pil_img.info}")
    except Exception as e:
        print(f"⚠️  PIL Error: {e}")
    
    # Mit OpenCV laden
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Cannot load with OpenCV!")
        return None
    
    print(f"🔧 OpenCV Info:")
    print(f"   Shape: {img.shape}")
    print(f"   Dtype: {img.dtype}")
    print(f"   Min/Max: {img.min()}/{img.max()}")
    print(f"   Mean: {img.mean():.2f}")
    
    # Channels analysieren
    if len(img.shape) == 3:
        b, g, r = cv2.split(img)
        print(f"   BGR Channels:")
        print(f"     Blue:  min={b.min()}, max={b.max()}, mean={b.mean():.2f}")
        print(f"     Green: min={g.min()}, max={g.max()}, mean={g.mean():.2f}")
        print(f"     Red:   min={r.min()}, max={r.max()}, mean={r.mean():.2f}")
    
    # Prüfe auf spezielle Eigenschaften
    unique_values = len(np.unique(img))
    print(f"   Unique pixel values: {unique_values}")
    
    # Prüfe auf Transparenz-ähnliche Bereiche
    if len(img.shape) == 3:
        # Prüfe auf weiße Bereiche (matplotlib background)
        white_pixels = np.all(img >= 250, axis=2).sum()
        total_pixels = img.shape[0] * img.shape[1]
        white_percentage = (white_pixels / total_pixels) * 100
        print(f"   White pixels: {white_percentage:.1f}%")
        
        # Prüfe auf schwarze Bereiche
        black_pixels = np.all(img <= 5, axis=2).sum()
        black_percentage = (black_pixels / total_pixels) * 100
        print(f"   Black pixels: {black_percentage:.1f}%")
    
    return img

def find_plot_files(base_dir):
    """Findet verschiedene Plot-Typen"""
    plot_types = {
        'returns_histogram': [],
        'candlestick': [],
        'autocorrelation': [],
        'phase_space': [],
        'other': []
    }
    
    if not os.path.exists(base_dir):
        print(f"❌ Directory does not exist: {base_dir}")
        return plot_types
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith('.png'):
                file_path = os.path.join(root, file)
                file_lower = file.lower()
                
                if 'returns_histogram' in file_lower:
                    plot_types['returns_histogram'].append(file_path)
                elif 'candlestick' in file_lower:
                    plot_types['candlestick'].append(file_path)
                elif 'autocorrelation' in file_lower:
                    plot_types['autocorrelation'].append(file_path)
                elif 'phase_space' in file_lower:
                    plot_types['phase_space'].append(file_path)
                else:
                    plot_types['other'].append(file_path)
    
    return plot_types

def compare_plots():
    """Vergleiche verschiedene Plot-Typen"""
    print("🔍 DIAGNOSTIC: Returns_Histogram vs Other Plots")
    print("="*60)
    
    # Spezifischer Pfad zu Returns_Histograms
    returns_histogram_dir = r"H:\Rave\WR_VL\Full Sweep 18072025\11111_PlotCollection\Returns_Histograms"
    
    # Andere Plot-Typen suchen
    base_collection_dir = r"H:\Rave\WR_VL\Full Sweep 18072025\11111_PlotCollection"
    
    plot_types = {
        'returns_histogram': [],
        'candlestick': [],
        'autocorrelation': [],
        'phase_space': [],
        'other': []
    }
    
    # Returns_Histograms sammeln
    if os.path.exists(returns_histogram_dir):
        for file in os.listdir(returns_histogram_dir):
            if file.lower().endswith('.png'):
                plot_types['returns_histogram'].append(os.path.join(returns_histogram_dir, file))
    
    # Andere Plot-Typen sammeln
    if os.path.exists(base_collection_dir):
        for item in os.listdir(base_collection_dir):
            item_path = os.path.join(base_collection_dir, item)
            if os.path.isdir(item_path) and item != "Returns_Histograms":
                for file in os.listdir(item_path):
                    if file.lower().endswith('.png'):
                        file_path = os.path.join(item_path, file)
                        item_lower = item.lower()
                        
                        if 'candlestick' in item_lower:
                            plot_types['candlestick'].append(file_path)
                        elif 'autocorrelation' in item_lower:
                            plot_types['autocorrelation'].append(file_path)
                        elif 'phase' in item_lower:
                            plot_types['phase_space'].append(file_path)
                        else:
                            plot_types['other'].append(file_path)
    
    print("\n📊 Found plot files:")
    for plot_type, files in plot_types.items():
        print(f"   {plot_type}: {len(files)} files")
        if files:
            print(f"      First file: {os.path.basename(files[0])}")
    
    # Analysiere je einen von jedem Typ
    for plot_type, files in plot_types.items():
        if files and len(files) > 0:
            print(f"\n{'='*60}")
            print(f"🎯 ANALYZING {plot_type.upper()}")
            print(f"{'='*60}")
            
            # Nehme das erste File
            sample_file = files[0]
            img_data = analyze_image_properties(sample_file)
            
            if img_data is not None:
                # Teste Video-Erstellung mit diesem einen File
                success = test_video_creation(sample_file, f"test_{plot_type}.mp4")
                
                if plot_type == 'returns_histogram' and not success:
                    print(f"\n❌ RETURNS_HISTOGRAM FAILED - Let's try alternatives...")
                    # Teste verschiedene Lösungsansätze
                    test_returns_histogram_fixes(sample_file)
            
            # Begrenzte Anzahl für schnellere Diagnose
            if plot_type == 'returns_histogram':
                break  # Erstmal nur Returns_Histograms testen

def test_video_creation(image_path, output_path):
    """Teste Video-Erstellung mit einem einzelnen Bild"""
    print(f"\n🎬 Testing video creation with: {os.path.basename(image_path)}")
    
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Cannot load image!")
        return False
    
    height, width = img.shape[:2]
    fps = 10
    
    # Verwende XVID (funktioniert laut Test am besten)
    fourcc = cv2.VideoWriter.fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("❌ Cannot create video writer!")
        return False
    
    print("✅ Video writer created successfully")
    
    # Schreibe 3 identische Frames
    for i in range(3):
        test_img = img.copy()
        # Füge Frame-Nummer hinzu
        cv2.putText(test_img, f"Frame {i+1}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        out.write(test_img)
    
    out.release()
    
    # Validiere das Video
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"✅ Video created: {file_size} bytes")
        
        # Versuche das Video zu lesen
        cap = cv2.VideoCapture(output_path)
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"✅ Video readable: {frame_count} frames")
            cap.release()
            
            # Teste mit Windows Media Player Format
            import subprocess
            try:
                # Versuche Video-Info zu bekommen
                result = subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json', 
                                       '-show_format', '-show_streams', output_path], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print("✅ Video format seems valid (ffprobe success)")
                else:
                    print("⚠️  ffprobe reported issues")
            except:
                print("ℹ️  ffprobe not available - skipping detailed analysis")
            
        else:
            print("❌ Video not readable!")
        
        # Aufräumen
        os.remove(output_path)
        return True
    else:
        print("❌ Video file was not created!")
        return False

def test_returns_histogram_fixes(image_path):
    """Spezielle Tests für Returns_Histogram Probleme"""
    print(f"\n🔧 TESTING RETURNS_HISTOGRAM FIXES")
    print(f"File: {os.path.basename(image_path)}")
    
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Cannot load image!")
        return
    
    height, width = img.shape[:2]
    print(f"Original dimensions: {width}x{height}")
    
    # Test 1: Verschiedene Auflösungen
    print("\n🎯 Test 1: Different resolutions")
    test_resolutions = [(1920, 1080), (1280, 720), (640, 480)]
    
    for test_width, test_height in test_resolutions:
        print(f"  Testing {test_width}x{test_height}...")
        resized = cv2.resize(img, (test_width, test_height))
        
        fourcc = cv2.VideoWriter.fourcc(*'XVID')
        test_file = f"fix_test_{test_width}x{test_height}.avi"
        out = cv2.VideoWriter(test_file, fourcc, 10, (test_width, test_height))
        
        if out.isOpened():
            out.write(resized)
            out.write(resized)  # 2 frames
            out.release()
            
            if os.path.exists(test_file):
                size = os.path.getsize(test_file)
                print(f"    ✅ Success: {size} bytes")
                os.remove(test_file)
            else:
                print(f"    ❌ Failed")
        else:
            print(f"    ❌ Writer failed")
    
    # Test 2: Verschiedene Codecs mit Original-Größe
    print(f"\n🎯 Test 2: Different codecs with original size {width}x{height}")
    codecs = [
        ('XVID', 'avi'),
        ('MJPG', 'avi'), 
        ('mp4v', 'mp4'),
        ('DIVX', 'avi')
    ]
    
    for codec, ext in codecs:
        print(f"  Testing {codec}...")
        fourcc = cv2.VideoWriter.fourcc(*codec)
        test_file = f"fix_test_{codec}.{ext}"
        out = cv2.VideoWriter(test_file, fourcc, 10, (width, height))
        
        if out.isOpened():
            out.write(img)
            out.write(img)
            out.release()
            
            if os.path.exists(test_file):
                size = os.path.getsize(test_file)
                print(f"    ✅ Success: {size} bytes")
                
                # Test playback
                cap = cv2.VideoCapture(test_file)
                if cap.isOpened():
                    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    print(f"    📹 Playable: {frames} frames")
                    cap.release()
                else:
                    print(f"    ⚠️  Not playable")
                
                os.remove(test_file)
            else:
                print(f"    ❌ No file created")
        else:
            print(f"    ❌ Writer failed")
    
    # Test 3: Image preprocessing
    print(f"\n🎯 Test 3: Image preprocessing")
    
    # Test mit verschiedenen Bildbearbeitungen
    preprocessed_images = {
        'original': img,
        'normalized': cv2.normalize(img, np.zeros_like(img), 0, 255, cv2.NORM_MINMAX),
        'converted': cv2.convertScaleAbs(img),
        'gaussian': cv2.GaussianBlur(img, (3, 3), 0)
    }
    
    for prep_name, prep_img in preprocessed_images.items():
        print(f"  Testing {prep_name} preprocessing...")
        fourcc = cv2.VideoWriter.fourcc(*'XVID')
        test_file = f"fix_test_{prep_name}.avi"
        out = cv2.VideoWriter(test_file, fourcc, 10, (width, height))
        
        if out.isOpened():
            out.write(prep_img)
            out.write(prep_img)
            out.release()
            
            if os.path.exists(test_file):
                size = os.path.getsize(test_file)
                print(f"    ✅ Success: {size} bytes")
                os.remove(test_file)
            else:
                print(f"    ❌ Failed")
        else:
            print(f"    ❌ Writer failed")

if __name__ == "__main__":
    compare_plots()
