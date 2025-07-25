#!/usr/bin/env python3
"""
Sentinel-1 SAR Data Handler
Downloads and processes Sentinel-1 C-band SAR data for solar farm analysis
"""

import os
import requests
import zipfile
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class Sentinel1DataHandler:
    """Handler for Sentinel-1 SAR data download and processing"""
    
    def __init__(self, output_dir: str = "sar_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.credentials = None
        
    def setup_credentials(self, username: str = None, password: str = None):
        """Setup Copernicus credentials for data access"""
        if username and password:
            self.credentials = (username, password)
        else:
            print("Note: For automatic download, set up Copernicus credentials")
            print("Register at: https://scihub.copernicus.eu/dhus/#/self-registration")
            
    def get_target_coordinates(self) -> Dict[str, Tuple[float, float, float, float]]:
        """Get target coordinates for Virginia and Taiwan based on AQMAR data"""
        # These coordinates match the AQMAR satellite coverage areas
        targets = {
            'virginia': {
                'name': 'Virginia_USA',
                'bbox': (-77.0, 36.5, -75.5, 37.5),  # (lon_min, lat_min, lon_max, lat_max)
                'description': 'Hampton Roads area, Virginia'
            },
            'taiwan': {
                'name': 'Taiwan_Placeholder',
                'bbox': (120.0, 23.0, 122.0, 25.0),  # Placeholder - adjust based on actual data
                'description': 'Taiwan region (coordinates estimated)'
            }
        }
        return targets
        
    def generate_search_query(self, bbox: Tuple[float, float, float, float], 
                            start_date: str = "2025-03-01", end_date: str = "2025-04-30") -> str:
        """Generate Copernicus Hub search query for Sentinel-1 data"""
        lon_min, lat_min, lon_max, lat_max = bbox
        
        # Format: POLYGON((lon1 lat1, lon2 lat2, ...))
        footprint = f"POLYGON(({lon_min} {lat_min},{lon_max} {lat_min},{lon_max} {lat_max},{lon_min} {lat_max},{lon_min} {lat_min}))"
        
        query = f"""
        platformname:Sentinel-1 AND 
        producttype:GRD AND 
        polarisation:VV+VH AND 
        sensoroperationalmode:IW AND
        beginposition:[{start_date}T00:00:00.000Z TO {end_date}T23:59:59.999Z] AND
        footprint:"Intersects({footprint})"
        """.replace('\n', ' ').strip()
        
        return query
    
    def create_download_instructions(self) -> str:
        """Create instructions for manual Sentinel-1 data download"""
        targets = self.get_target_coordinates()
        
        instructions = """
=== Sentinel-1 Data Download Instructions ===

Since automated download requires Copernicus credentials, follow these steps:

1. Register at: https://scihub.copernicus.eu/dhus/#/self-registration

2. Search and download data for each target area:

"""
        
        for region, info in targets.items():
            lon_min, lat_min, lon_max, lat_max = info['bbox']
            
            instructions += f"""
--- {info['name']} ({info['description']}) ---
Search Parameters:
- Platform: Sentinel-1
- Product Type: GRD (Ground Range Detected)
- Sensor Mode: IW (Interferometric Wide)
- Polarisation: VV+VH (Dual polarization)
- Time Range: 2025-03-01 to 2025-04-30
- Area: {lat_min}°N to {lat_max}°N, {lon_min}°E to {lon_max}°E

Manual Search URL:
https://scihub.copernicus.eu/dhus/#/home

Search Query (copy-paste):
{self.generate_search_query(info['bbox'])}

Recommended files to download:
- Look for recent IW_GRDH products
- Prefer ascending passes for consistency
- Download both VV and VH polarizations if available

Save to: {self.output_dir / region}/

"""
        
        instructions += """
3. Alternative Free Sources:
- Google Earth Engine (requires account): https://earthengine.google.com/
- ASF Data Search (Alaska Satellite Facility): https://search.asf.alaska.edu/
- USGS EarthExplorer: https://earthexplorer.usgs.gov/

4. Once downloaded, place files in the appropriate subdirectories and run the processing pipeline.
"""
        
        return instructions
    
    def create_mock_sar_data(self, region: str, size: Tuple[int, int] = (1000, 1000)) -> Dict:
        """Create mock SAR data for testing when real data is not available"""
        print(f"Creating mock SAR data for {region} (for testing purposes)")
        
        # Simulate realistic SAR backscatter values
        # Typical range: -30 to 0 dB, converted to linear scale
        mock_vv = np.random.gamma(2, 2, size) * 0.01  # VV polarization
        mock_vh = np.random.gamma(1.5, 1.5, size) * 0.005  # VH polarization (typically lower)
        
        # Add some geometric structures that could represent solar installations
        y_center, x_center = size[0] // 2, size[1] // 2
        
        # Create rectangular high-backscatter regions (simulating metal structures)
        for i in range(3):  # Add 3 mock solar farms
            y_offset = np.random.randint(-200, 200)
            x_offset = np.random.randint(-200, 200)
            width = np.random.randint(50, 150)
            height = np.random.randint(30, 100)
            
            y1 = max(0, min(size[0], y_center + y_offset))
            x1 = max(0, min(size[1], x_center + x_offset))
            y2 = max(0, min(size[0], y1 + height))
            x2 = max(0, min(size[1], x1 + width))
            
            # Higher backscatter for metal structures
            mock_vv[y1:y2, x1:x2] *= 3.0
            mock_vh[y1:y2, x1:x2] *= 2.0
        
        # Add noise
        mock_vv += np.random.normal(0, 0.001, size)
        mock_vh += np.random.normal(0, 0.0005, size)
        
        # Ensure positive values
        mock_vv = np.abs(mock_vv)
        mock_vh = np.abs(mock_vh)
        
        return {
            'vv_intensity': mock_vv,
            'vh_intensity': mock_vh,
            'vv_db': 10 * np.log10(mock_vv + 1e-10),
            'vh_db': 10 * np.log10(mock_vh + 1e-10),
            'metadata': {
                'region': region,
                'polarizations': ['VV', 'VH'],
                'resolution': '20m',
                'type': 'mock_data',
                'created': datetime.now().isoformat()
            }
        }
    
    def process_sar_data(self, file_path: str) -> Dict:
        """Process downloaded SAR data (placeholder for real implementation)"""
        print(f"Processing SAR data: {file_path}")
        
        # This would normally use libraries like:
        # - SNAP (Sentinel Application Platform)
        # - pyroSAR
        # - GDAL/rasterio for TIFF files
        
        # For now, return mock data structure
        return {
            'status': 'placeholder',
            'message': 'Real SAR processing requires SNAP or similar tools',
            'file_path': file_path
        }
    
    def save_mock_data(self, region: str, sar_data: Dict):
        """Save mock SAR data for testing"""
        region_dir = self.output_dir / region
        region_dir.mkdir(exist_ok=True)
        
        # Save as numpy arrays
        np.save(region_dir / 'vv_intensity.npy', sar_data['vv_intensity'])
        np.save(region_dir / 'vh_intensity.npy', sar_data['vh_intensity'])
        np.save(region_dir / 'vv_db.npy', sar_data['vv_db'])
        np.save(region_dir / 'vh_db.npy', sar_data['vh_db'])
        
        # Save metadata
        import json
        with open(region_dir / 'metadata.json', 'w') as f:
            json.dump(sar_data['metadata'], f, indent=2)
        
        print(f"Mock SAR data saved to: {region_dir}")
    
    def setup_sar_data(self) -> Dict:
        """Setup SAR data (download instructions + mock data for testing)"""
        # Print download instructions
        instructions = self.create_download_instructions()
        
        # Save instructions to file
        with open(self.output_dir / 'download_instructions.txt', 'w') as f:
            f.write(instructions)
        
        print("SAR data setup complete:")
        print(f"- Instructions saved to: {self.output_dir / 'download_instructions.txt'}")
        
        # Create mock data for immediate testing
        results = {}
        targets = self.get_target_coordinates()
        
        for region in targets.keys():
            mock_data = self.create_mock_sar_data(region)
            self.save_mock_data(region, mock_data)
            results[region] = mock_data
        
        print("- Mock SAR data created for testing")
        
        return results

def main():
    """Main function for SAR data setup"""
    print("=== Sentinel-1 SAR Data Handler ===")
    
    handler = Sentinel1DataHandler()
    
    # Setup SAR data
    results = handler.setup_sar_data()
    
    print(f"\nSAR data setup complete for {len(results)} regions")
    print("Next steps:")
    print("1. Follow download instructions for real Sentinel-1 data")
    print("2. Use mock data for immediate algorithm development")
    print("3. Replace mock data with real SAR data when available")

if __name__ == "__main__":
    main()