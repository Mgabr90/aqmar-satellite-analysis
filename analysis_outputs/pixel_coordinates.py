#!/usr/bin/env python3
"""
Pixel-to-Coordinate Mapping for AQMAR Satellite Data
Converts pixel positions to geographic coordinates (lat/lon)
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path

class PixelCoordinateMapper:
    """Maps pixel coordinates to geographic coordinates for satellite imagery"""
    
    def __init__(self, metadata_file):
        """Initialize with satellite metadata file"""
        self.metadata_file = Path(metadata_file)
        self.parse_metadata()
        
    def parse_metadata(self):
        """Extract georeferencing parameters from metadata XML"""
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            content = f.read()
        root = ET.fromstring(content)
        
        # Image dimensions
        img_info = root.find('ImageInfo')
        self.width = int(img_info.find('NumSamples').text)
        self.height = int(img_info.find('NumLines').text)
        self.pixel_spacing_x = float(img_info.find('SampleSpacing').text)  # meters
        self.pixel_spacing_y = float(img_info.find('LineSpacing').text)    # meters
        
        # Corner coordinates
        corners = img_info.find('Corners')
        self.top_left_lat = float(corners.find('TopLeft/Latitude').text)
        self.top_left_lon = float(corners.find('TopLeft/Longitude').text)
        self.top_right_lat = float(corners.find('TopRight/Latitude').text)
        self.top_right_lon = float(corners.find('TopRight/Longitude').text)
        self.bottom_left_lat = float(corners.find('BottomLeft/Latitude').text)
        self.bottom_left_lon = float(corners.find('BottomLeft/Longitude').text)
        self.bottom_right_lat = float(corners.find('BottomRight/Latitude').text)
        self.bottom_right_lon = float(corners.find('BottomRight/Longitude').text)
        
        # Center coordinate
        center = img_info.find('CenterLocation')
        self.center_lat = float(center.find('Latitude').text)
        self.center_lon = float(center.find('Longitude').text)
        
        print(f"Image dimensions: {self.width} x {self.height}")
        print(f"Pixel spacing: {self.pixel_spacing_x}m x {self.pixel_spacing_y}m")
        print(f"Corner coordinates:")
        print(f"  Top-left: {self.top_left_lat:.6f}, {self.top_left_lon:.6f}")
        print(f"  Top-right: {self.top_right_lat:.6f}, {self.top_right_lon:.6f}")
        print(f"  Bottom-left: {self.bottom_left_lat:.6f}, {self.bottom_left_lon:.6f}")
        print(f"  Bottom-right: {self.bottom_right_lat:.6f}, {self.bottom_right_lon:.6f}")
        
    def pixel_to_coordinates(self, pixel_x, pixel_y):
        """
        Convert pixel coordinates to geographic coordinates using bilinear interpolation
        
        Args:
            pixel_x: Column index (0 to width-1)
            pixel_y: Row index (0 to height-1)
            
        Returns:
            tuple: (latitude, longitude)
        """
        # Normalize pixel coordinates to [0, 1]
        norm_x = pixel_x / (self.width - 1)
        norm_y = pixel_y / (self.height - 1)
        
        # Bilinear interpolation
        # Top edge interpolation
        top_lat = self.top_left_lat + norm_x * (self.top_right_lat - self.top_left_lat)
        top_lon = self.top_left_lon + norm_x * (self.top_right_lon - self.top_left_lon)
        
        # Bottom edge interpolation
        bottom_lat = self.bottom_left_lat + norm_x * (self.bottom_right_lat - self.bottom_left_lat)
        bottom_lon = self.bottom_left_lon + norm_x * (self.bottom_right_lon - self.bottom_left_lon)
        
        # Final interpolation between top and bottom
        lat = top_lat + norm_y * (bottom_lat - top_lat)
        lon = top_lon + norm_y * (bottom_lon - top_lon)
        
        return lat, lon
    
    def coordinates_to_pixel(self, lat, lon):
        """
        Convert geographic coordinates to pixel coordinates (approximate)
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            tuple: (pixel_x, pixel_y)
        """
        # Simple inverse mapping using center point and pixel spacing
        # This is approximate and works best for small areas
        
        # Calculate offset from center in degrees
        lat_offset = lat - self.center_lat
        lon_offset = lon - self.center_lon
        
        # Convert to meters (approximate)
        lat_meters = lat_offset * 111000  # ~111km per degree latitude
        lon_meters = lon_offset * 111000 * np.cos(np.radians(self.center_lat))
        
        # Convert to pixels
        pixel_x = (self.width // 2) + (lon_meters / self.pixel_spacing_x)
        pixel_y = (self.height // 2) - (lat_meters / self.pixel_spacing_y)  # Y axis is flipped
        
        return int(pixel_x), int(pixel_y)
    
    def generate_coordinate_grid(self):
        """
        Generate full coordinate grid for all pixels
        
        Returns:
            tuple: (lat_grid, lon_grid) - 2D arrays of coordinates
        """
        print("Generating coordinate grid...")
        
        lat_grid = np.zeros((self.height, self.width))
        lon_grid = np.zeros((self.height, self.width))
        
        for y in range(self.height):
            for x in range(self.width):
                lat, lon = self.pixel_to_coordinates(x, y)
                lat_grid[y, x] = lat
                lon_grid[y, x] = lon
                
        print(f"Grid generated: {lat_grid.shape}")
        print(f"Latitude range: {lat_grid.min():.6f} to {lat_grid.max():.6f}")
        print(f"Longitude range: {lon_grid.min():.6f} to {lon_grid.max():.6f}")
        
        return lat_grid, lon_grid
    
    def get_pixel_coordinates_batch(self, pixel_coords):
        """
        Get coordinates for multiple pixels at once
        
        Args:
            pixel_coords: List of (x, y) tuples
            
        Returns:
            List of (lat, lon) tuples
        """
        results = []
        for x, y in pixel_coords:
            lat, lon = self.pixel_to_coordinates(x, y)
            results.append((lat, lon))
        return results

def demonstrate_coordinate_mapping():
    """Demonstrate coordinate mapping for both datasets"""
    data_dir = Path("Data")
    
    # Find extracted directories
    dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    for data_folder in dirs:
        print(f"\n{'='*60}")
        print(f"Processing: {data_folder.name}")
        print('='*60)
        
        # Find metadata file
        meta_file = data_folder / f"{data_folder.name}.meta.xml"
        
        if meta_file.exists():
            mapper = PixelCoordinateMapper(meta_file)
            
            # Test specific pixels
            test_pixels = [
                (0, 0),  # Top-left
                (mapper.width-1, 0),  # Top-right
                (0, mapper.height-1),  # Bottom-left
                (mapper.width-1, mapper.height-1),  # Bottom-right
                (mapper.width//2, mapper.height//2),  # Center
            ]
            
            print(f"\nTest pixel coordinates:")
            for i, (x, y) in enumerate(test_pixels):
                lat, lon = mapper.pixel_to_coordinates(x, y)
                labels = ["Top-left", "Top-right", "Bottom-left", "Bottom-right", "Center"]
                print(f"  {labels[i]} ({x}, {y}): {lat:.6f}, {lon:.6f}")
            
            # Test reverse mapping
            print(f"\nReverse mapping test (center point):")
            center_lat, center_lon = mapper.pixel_to_coordinates(mapper.width//2, mapper.height//2)
            back_x, back_y = mapper.coordinates_to_pixel(center_lat, center_lon)
            print(f"  Original pixel: ({mapper.width//2}, {mapper.height//2})")
            print(f"  Coordinates: {center_lat:.6f}, {center_lon:.6f}")
            print(f"  Back to pixel: ({back_x}, {back_y})")
            
            # Save coordinate grids
            print(f"\nGenerating and saving coordinate grids...")
            lat_grid, lon_grid = mapper.generate_coordinate_grid()
            
            np.save(data_folder / "latitude_grid.npy", lat_grid)
            np.save(data_folder / "longitude_grid.npy", lon_grid)
            print(f"Saved coordinate grids to {data_folder}")

if __name__ == "__main__":
    demonstrate_coordinate_mapping()