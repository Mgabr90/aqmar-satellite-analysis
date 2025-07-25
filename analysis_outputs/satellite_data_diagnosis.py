#!/usr/bin/env python3
"""
AQMAR Satellite Data Diagnosis Script
Analyzes satellite imagery files and metadata from AS03 satellite
"""

import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from pathlib import Path

def diagnose_satellite_data():
    """Diagnose satellite data structure and content"""
    data_dir = Path("Data")
    
    print("=== AQMAR Satellite Data Diagnosis ===\n")
    
    # Find extracted directories
    dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    for data_folder in dirs:
        print(f"Analyzing: {data_folder.name}")
        print("-" * 50)
        
        # Parse metadata
        meta_file = data_folder / f"{data_folder.name}.meta.xml"
        if meta_file.exists():
            with open(meta_file, 'r', encoding='utf-8') as f:
                content = f.read()
            tree = ET.fromstring(content)
            root = tree
            
            # Extract key metadata
            satellite = root.find('Satellite').text if root.find('Satellite') is not None else 'Unknown'
            sensor = root.find('Sensor').text if root.find('Sensor') is not None else 'Unknown'
            
            # Image info
            img_info = root.find('ImageInfo')
            if img_info is not None:
                start_time = img_info.find('StartTime').text if img_info.find('StartTime') is not None else 'Unknown'
                center_lat = img_info.find('CenterLocation/Latitude').text if img_info.find('CenterLocation/Latitude') is not None else 'Unknown'
                center_lon = img_info.find('CenterLocation/Longitude').text if img_info.find('CenterLocation/Longitude') is not None else 'Unknown'
                width = img_info.find('NumSamples').text if img_info.find('NumSamples') is not None else 'Unknown'
                height = img_info.find('NumLines').text if img_info.find('NumLines') is not None else 'Unknown'
                cloud_percent = img_info.find('CloudPercent').text if img_info.find('CloudPercent') is not None else 'Unknown'
                
                print(f"Satellite: {satellite}")
                print(f"Sensor: {sensor}")
                print(f"Capture Time: {start_time}")
                print(f"Center Location: {center_lat}N, {center_lon}E")
                print(f"Dimensions: {width} x {height} pixels")
                print(f"Cloud Coverage: {cloud_percent}%")
        
        # Analyze TIFF image
        tiff_file = data_folder / f"{data_folder.name}.tif"
        if tiff_file.exists():
            try:
                with Image.open(tiff_file) as img:
                    print(f"Image Format: {img.format}")
                    print(f"Mode: {img.mode}")
                    print(f"Size: {img.size}")
                    print(f"File Size: {tiff_file.stat().st_size / 1024 / 1024:.2f} MB")
                    
                    # Check if it's a numpy-readable format
                    if hasattr(img, 'getdata'):
                        data = np.array(img)
                        print(f"Data Type: {data.dtype}")
                        print(f"Value Range: {data.min()} - {data.max()}")
                        
            except Exception as e:
                print(f"Error reading TIFF: {e}")
        
        # List other files
        other_files = [f for f in data_folder.iterdir() if f.is_file() and not f.name.endswith('.tif') and not f.name.endswith('.meta.xml')]
        if other_files:
            print(f"Additional Files:")
            for f in other_files:
                print(f"   - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
        
        print("\n")

if __name__ == "__main__":
    diagnose_satellite_data()