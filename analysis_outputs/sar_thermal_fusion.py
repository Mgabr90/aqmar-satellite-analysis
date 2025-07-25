#!/usr/bin/env python3
"""
SAR-Thermal Fusion System for Cell-Level Solar Panel Monitoring
Provides precise panel identification combined with thermal anomaly detection
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Union
import json
from scipy import ndimage, interpolate
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SolarPanel:
    """Individual solar panel with precise location and thermal data"""
    panel_id: int
    row: int
    col: int
    coordinates: Tuple[float, float]  # (x, y) center coordinates in meters
    bounds: Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)
    size: Tuple[float, float] = (2.0, 1.0)  # Standard panel size: 2m x 1m
    temperature: Optional[float] = None
    thermal_std: Optional[float] = None
    is_anomalous: bool = False
    anomaly_severity: str = "normal"  # normal, minor, moderate, severe, critical
    confidence: float = 1.0

@dataclass
class SolarCell:
    """Individual solar cell within a panel (for sub-panel analysis)"""
    cell_id: str  # Format: "Panel_347_Cell_A1"
    panel_id: int
    cell_position: Tuple[int, int]  # (row, col) within panel
    coordinates: Tuple[float, float]  # Absolute coordinates
    temperature: Optional[float] = None
    is_hotspot: bool = False
    hotspot_severity: str = "normal"

@dataclass
class SolarFarm:
    """Complete solar farm with structured panel layout"""
    farm_id: str
    panels: List[SolarPanel] = field(default_factory=list)
    cells: List[SolarCell] = field(default_factory=list)
    layout: Tuple[int, int] = (20, 50)  # 20 rows x 50 panels = 1000 panels
    panel_spacing: Tuple[float, float] = (0.5, 0.5)  # 0.5m spacing between panels
    origin: Tuple[float, float] = (0.0, 0.0)  # Farm origin coordinates
    total_panels: int = 1000

class HighResSARProcessor:
    """High-resolution SAR processor for precise panel boundary detection"""
    
    def __init__(self, resolution: float = 0.5):
        self.resolution = resolution  # meters per pixel
        self.panel_size = (2.0, 1.0)  # Standard panel size in meters
        
    def create_mock_highres_sar(self, farm_layout: Tuple[int, int] = (20, 50), 
                               size: Tuple[int, int] = (2000, 3000)) -> np.ndarray:
        """Create mock high-resolution SAR data with realistic panel structures"""
        print(f"Creating mock high-res SAR data ({self.resolution}m resolution)")
        
        # Initialize background with typical ground backscatter
        sar_image = np.random.gamma(1.5, 0.5, size) * 0.001  # Low backscatter background
        
        # Calculate panel dimensions in pixels
        panel_width_px = int(self.panel_size[0] / self.resolution)  # 2m / 0.5m = 4 pixels
        panel_height_px = int(self.panel_size[1] / self.resolution)  # 1m / 0.5m = 2 pixels
        spacing_x_px = int(0.5 / self.resolution)  # 0.5m spacing = 1 pixel
        spacing_y_px = int(0.5 / self.resolution)
        
        rows, cols = farm_layout
        
        # Calculate farm dimensions
        farm_width_px = cols * (panel_width_px + spacing_x_px) - spacing_x_px
        farm_height_px = rows * (panel_height_px + spacing_y_px) - spacing_y_px
        
        # Center the farm in the image
        start_x = (size[1] - farm_width_px) // 2
        start_y = (size[0] - farm_height_px) // 2
        
        panel_id = 1
        panel_coordinates = []
        
        for row in range(rows):
            for col in range(cols):
                # Calculate panel position
                x_start = start_x + col * (panel_width_px + spacing_x_px)
                y_start = start_y + row * (panel_height_px + spacing_y_px)
                x_end = x_start + panel_width_px
                y_end = y_start + panel_height_px
                
                # Ensure bounds are within image
                if x_end < size[1] and y_end < size[0]:
                    # High backscatter for metal panel frames
                    sar_image[y_start:y_end, x_start:x_end] = np.random.gamma(3, 0.01, 
                                                                            (y_end-y_start, x_end-x_start))
                    
                    # Add panel frame (even higher backscatter)
                    sar_image[y_start:y_start+1, x_start:x_end] *= 2.0  # Top edge
                    sar_image[y_end-1:y_end, x_start:x_end] *= 2.0      # Bottom edge
                    sar_image[y_start:y_end, x_start:x_start+1] *= 2.0  # Left edge
                    sar_image[y_start:y_end, x_end-1:x_end] *= 2.0      # Right edge
                    
                    # Store panel center coordinates (in meters from origin)
                    center_x = (x_start + x_end) / 2 * self.resolution
                    center_y = (y_start + y_end) / 2 * self.resolution
                    panel_coordinates.append((panel_id, row+1, col+1, center_x, center_y, 
                                            x_start, y_start, x_end, y_end))
                    panel_id += 1
        
        # Add some noise and speckle (typical for SAR)
        noise = np.random.gamma(1, 0.0001, size)
        sar_image += noise
        
        # Store panel coordinates for later use
        self.panel_coordinates = panel_coordinates
        
        return sar_image
    
    def detect_panel_boundaries(self, sar_image: np.ndarray, 
                              min_panel_area: int = 6) -> List[SolarPanel]:
        """Detect individual panel boundaries using high-resolution SAR"""
        print("Detecting panel boundaries with high-resolution SAR...")
        
        # Convert to dB scale
        sar_db = 10 * np.log10(sar_image + 1e-10)
        
        # Threshold for high backscatter (metal structures)
        threshold = np.percentile(sar_db, 85)  # Top 15% of backscatter values
        binary_mask = sar_db > threshold
        
        # Morphological operations to clean up the mask
        kernel = np.ones((2, 2), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # Find connected components (individual panels)
        num_labels, labels = cv2.connectedComponents(binary_mask)
        
        detected_panels = []
        
        for label in range(1, num_labels):
            mask = (labels == label)
            area = np.sum(mask)
            
            if area >= min_panel_area:  # Minimum panel size
                # Get bounding box
                coords = np.where(mask)
                y_min, y_max = np.min(coords[0]), np.max(coords[0])
                x_min, x_max = np.min(coords[1]), np.max(coords[1])
                
                # Calculate panel dimensions
                width_px = x_max - x_min
                height_px = y_max - y_min
                width_m = width_px * self.resolution
                height_m = height_px * self.resolution
                
                # Check if dimensions match expected panel size (±20% tolerance)
                expected_area = self.panel_size[0] * self.panel_size[1]
                actual_area = width_m * height_m
                
                if 0.8 * expected_area <= actual_area <= 1.2 * expected_area:
                    # Calculate center coordinates
                    center_x = ((x_min + x_max) / 2) * self.resolution
                    center_y = ((y_min + y_max) / 2) * self.resolution
                    
                    # Convert pixel bounds to meter bounds
                    bounds_m = (x_min * self.resolution, y_min * self.resolution,
                              x_max * self.resolution, y_max * self.resolution)
                    
                    panel = SolarPanel(
                        panel_id=len(detected_panels) + 1,
                        row=0,  # Will be assigned during grid generation
                        col=0,  # Will be assigned during grid generation
                        coordinates=(center_x, center_y),
                        bounds=bounds_m,
                        size=(width_m, height_m)
                    )
                    detected_panels.append(panel)
        
        print(f"Detected {len(detected_panels)} panels using SAR boundary detection")
        return detected_panels
    
    def generate_structured_grid(self, detected_panels: List[SolarPanel], 
                               expected_layout: Tuple[int, int] = (20, 50)) -> SolarFarm:
        """Generate structured grid from detected panels"""
        print(f"Generating structured grid: {expected_layout[0]} rows × {expected_layout[1]} columns")
        
        if not detected_panels:
            return SolarFarm(farm_id="mock_farm", layout=expected_layout)
        
        # Sort panels by position (top-to-bottom, left-to-right)
        sorted_panels = sorted(detected_panels, key=lambda p: (p.coordinates[1], p.coordinates[0]))
        
        # Assign row and column positions
        rows, cols = expected_layout
        panels_per_row = cols
        
        for i, panel in enumerate(sorted_panels):
            if i < rows * cols:  # Ensure we don't exceed expected layout
                panel.row = (i // panels_per_row) + 1
                panel.col = (i % panels_per_row) + 1
                panel.panel_id = i + 1
        
        # Create solar farm
        farm = SolarFarm(
            farm_id="mock_farm",
            panels=sorted_panels[:rows * cols],  # Take only expected number of panels
            layout=expected_layout,
            total_panels=min(len(sorted_panels), rows * cols)
        )
        
        print(f"Generated solar farm with {len(farm.panels)} panels in structured grid")
        return farm

class ThermalProcessor:
    """Thermal infrared data processor"""
    
    def __init__(self, resolution: float = 0.3, temperature_range: Tuple[float, float] = (25, 90)):
        self.resolution = resolution  # meters per pixel
        self.temperature_range = temperature_range  # (min_temp, max_temp) in Celsius
        
    def create_mock_thermal_data(self, size: Tuple[int, int] = (2000, 3000),
                               solar_farm: Optional[SolarFarm] = None) -> np.ndarray:
        """Create mock thermal infrared data with realistic temperature patterns"""
        print(f"Creating mock thermal data ({self.resolution}m resolution)")
        
        # Base temperature (ambient)
        base_temp = 25.0  # Celsius
        thermal_image = np.full(size, base_temp, dtype=np.float32)
        
        # Add gradual spatial temperature variation (environmental effects)
        y, x = np.ogrid[:size[0], :size[1]]
        thermal_image += 5 * np.sin(x / 500) * np.cos(y / 300)
        
        if solar_farm and solar_farm.panels:
            # Add thermal signatures for solar panels
            for panel in solar_farm.panels:
                # Convert panel bounds from SAR coordinates to thermal coordinates
                scale_factor = 0.5 / self.resolution  # SAR resolution / thermal resolution
                
                x_min = int(panel.bounds[0] / self.resolution)
                y_min = int(panel.bounds[1] / self.resolution)
                x_max = int(panel.bounds[2] / self.resolution)
                y_max = int(panel.bounds[3] / self.resolution)
                
                # Ensure bounds are within image
                x_min = max(0, min(x_min, size[1]-1))
                x_max = max(0, min(x_max, size[1]))
                y_min = max(0, min(y_min, size[0]-1))
                y_max = max(0, min(y_max, size[0]))
                
                if x_max > x_min and y_max > y_min:
                    # Normal panel temperature (operational)
                    panel_temp = np.random.normal(45, 3)  # 45°C ± 3°C
                    
                    # Create some anomalous panels
                    if np.random.random() < 0.05:  # 5% anomaly rate
                        panel_temp = np.random.normal(75, 5)  # Overheating
                        
                        # Add hotspot within panel (simulating cell failure)
                        if np.random.random() < 0.5:  # 50% chance of hotspot in anomalous panel
                            hotspot_x = np.random.randint(x_min, x_max)
                            hotspot_y = np.random.randint(y_min, y_max)
                            hotspot_size = 3  # 3-pixel hotspot
                            
                            hotspot_x_start = max(x_min, hotspot_x - hotspot_size//2)
                            hotspot_x_end = min(x_max, hotspot_x + hotspot_size//2)
                            hotspot_y_start = max(y_min, hotspot_y - hotspot_size//2)
                            hotspot_y_end = min(y_max, hotspot_y + hotspot_size//2)
                            
                            thermal_image[hotspot_y_start:hotspot_y_end, 
                                        hotspot_x_start:hotspot_x_end] = np.random.normal(85, 2)
                    
                    # Set panel temperature
                    thermal_image[y_min:y_max, x_min:x_max] = panel_temp
        
        # Add thermal noise
        noise = np.random.normal(0, 0.5, size)
        thermal_image += noise
        
        # Ensure temperature range is realistic
        thermal_image = np.clip(thermal_image, self.temperature_range[0], self.temperature_range[1])
        
        return thermal_image

class SARThermalFusion:
    """SAR-Thermal fusion engine for cell-level solar monitoring"""
    
    def __init__(self, sar_processor: HighResSARProcessor, thermal_processor: ThermalProcessor):
        self.sar_processor = sar_processor
        self.thermal_processor = thermal_processor
        self.normal_temp_range = (40, 50)  # Normal operating temperature range
        self.anomaly_thresholds = {
            'minor': 55,     # 55-65°C
            'moderate': 65,  # 65-75°C  
            'severe': 75,    # 75-85°C
            'critical': 85   # >85°C
        }
        
    def register_thermal_to_sar(self, thermal_image: np.ndarray, 
                              sar_reference_size: Tuple[int, int]) -> np.ndarray:
        """Register thermal image to SAR coordinate system"""
        print("Registering thermal image to SAR coordinate system...")
        
        # Simple scaling-based registration (assumes same area coverage)
        # In practice, this would use sophisticated geometric registration
        thermal_registered = cv2.resize(thermal_image, 
                                      (sar_reference_size[1], sar_reference_size[0]),
                                      interpolation=cv2.INTER_CUBIC)
        
        return thermal_registered
    
    def extract_panel_temperatures(self, thermal_image: np.ndarray, 
                                 solar_farm: SolarFarm) -> SolarFarm:
        """Extract temperature statistics for each panel"""
        print("Extracting panel-level temperature data...")
        
        thermal_scale = self.thermal_processor.resolution
        
        for panel in solar_farm.panels:
            # Convert panel bounds to thermal image coordinates
            x_min = int(panel.bounds[0] / thermal_scale)
            y_min = int(panel.bounds[1] / thermal_scale)
            x_max = int(panel.bounds[2] / thermal_scale)
            y_max = int(panel.bounds[3] / thermal_scale)
            
            # Ensure bounds are within image
            x_min = max(0, min(x_min, thermal_image.shape[1]-1))
            x_max = max(0, min(x_max, thermal_image.shape[1]))
            y_min = max(0, min(y_min, thermal_image.shape[0]-1))
            y_max = max(0, min(y_max, thermal_image.shape[0]))
            
            if x_max > x_min and y_max > y_min:
                # Extract panel region
                panel_region = thermal_image[y_min:y_max, x_min:x_max]
                
                if panel_region.size > 0:
                    # Calculate temperature statistics
                    panel.temperature = float(np.mean(panel_region))
                    panel.thermal_std = float(np.std(panel_region))
                    
                    # Determine anomaly status
                    max_temp = np.max(panel_region)
                    panel.is_anomalous = max_temp > self.normal_temp_range[1]
                    
                    # Classify severity
                    if max_temp < self.anomaly_thresholds['minor']:
                        panel.anomaly_severity = "normal"
                    elif max_temp < self.anomaly_thresholds['moderate']:
                        panel.anomaly_severity = "minor"
                    elif max_temp < self.anomaly_thresholds['severe']:
                        panel.anomaly_severity = "moderate"
                    elif max_temp < self.anomaly_thresholds['critical']:
                        panel.anomaly_severity = "severe"
                    else:
                        panel.anomaly_severity = "critical"
        
        return solar_farm
    
    def detect_cell_level_hotspots(self, thermal_image: np.ndarray, 
                                 solar_farm: SolarFarm, 
                                 cells_per_panel: Tuple[int, int] = (4, 2)) -> List[SolarCell]:
        """Detect cell-level hotspots within panels"""
        print("Detecting cell-level thermal hotspots...")
        
        cells = []
        thermal_scale = self.thermal_processor.resolution
        
        for panel in solar_farm.panels:
            if panel.is_anomalous:  # Only analyze anomalous panels for efficiency
                # Convert panel bounds to thermal image coordinates
                x_min = int(panel.bounds[0] / thermal_scale)
                y_min = int(panel.bounds[1] / thermal_scale)
                x_max = int(panel.bounds[2] / thermal_scale)
                y_max = int(panel.bounds[3] / thermal_scale)
                
                # Ensure bounds are within image
                x_min = max(0, min(x_min, thermal_image.shape[1]-1))
                x_max = max(0, min(x_max, thermal_image.shape[1]))
                y_min = max(0, min(y_min, thermal_image.shape[0]-1))
                y_max = max(0, min(y_max, thermal_image.shape[0]))
                
                if x_max > x_min and y_max > y_min:
                    panel_region = thermal_image[y_min:y_max, x_min:x_max]
                    
                    # Divide panel into cells
                    cell_rows, cell_cols = cells_per_panel
                    cell_height = (y_max - y_min) // cell_rows
                    cell_width = (x_max - x_min) // cell_cols
                    
                    for cell_row in range(cell_rows):
                        for cell_col in range(cell_cols):
                            cell_y_start = y_min + cell_row * cell_height
                            cell_y_end = min(y_max, cell_y_start + cell_height)
                            cell_x_start = x_min + cell_col * cell_width
                            cell_x_end = min(x_max, cell_x_start + cell_width)
                            
                            if cell_y_end > cell_y_start and cell_x_end > cell_x_start:
                                cell_region = thermal_image[cell_y_start:cell_y_end, 
                                                          cell_x_start:cell_x_end]
                                
                                if cell_region.size > 0:
                                    max_cell_temp = np.max(cell_region)
                                    avg_cell_temp = np.mean(cell_region)
                                    
                                    # Cell coordinates in absolute system
                                    cell_center_x = ((cell_x_start + cell_x_end) / 2) * thermal_scale
                                    cell_center_y = ((cell_y_start + cell_y_end) / 2) * thermal_scale
                                    
                                    # Determine if cell is a hotspot
                                    is_hotspot = max_cell_temp > self.anomaly_thresholds['moderate']
                                    
                                    severity = "normal"
                                    if max_cell_temp > self.anomaly_thresholds['critical']:
                                        severity = "critical"
                                    elif max_cell_temp > self.anomaly_thresholds['severe']:
                                        severity = "severe"
                                    elif max_cell_temp > self.anomaly_thresholds['moderate']:
                                        severity = "moderate"
                                    
                                    cell = SolarCell(
                                        cell_id=f"Panel_{panel.panel_id}_Cell_{chr(65+cell_row)}{cell_col+1}",
                                        panel_id=panel.panel_id,
                                        cell_position=(cell_row, cell_col),
                                        coordinates=(cell_center_x, cell_center_y),
                                        temperature=float(avg_cell_temp),
                                        is_hotspot=is_hotspot,
                                        hotspot_severity=severity
                                    )
                                    cells.append(cell)
        
        return cells
    
    def generate_precision_report(self, solar_farm: SolarFarm, 
                                cells: List[SolarCell]) -> Dict:
        """Generate precision monitoring report with exact coordinates"""
        print("Generating precision monitoring report...")
        
        # Count anomalies by severity
        anomaly_counts = {
            'normal': 0, 'minor': 0, 'moderate': 0, 'severe': 0, 'critical': 0
        }
        
        anomalous_panels = []
        hotspot_cells = []
        
        for panel in solar_farm.panels:
            anomaly_counts[panel.anomaly_severity] += 1
            
            if panel.is_anomalous:
                anomalous_panels.append({
                    'panel_id': panel.panel_id,
                    'row': panel.row,
                    'col': panel.col,
                    'coordinates': panel.coordinates,
                    'temperature': panel.temperature,
                    'severity': panel.anomaly_severity,
                    'location_description': f"Panel #{panel.panel_id} at coordinates ({panel.coordinates[0]:.1f}m, {panel.coordinates[1]:.1f}m)"
                })
        
        for cell in cells:
            if cell.is_hotspot:
                hotspot_cells.append({
                    'cell_id': cell.cell_id,
                    'panel_id': cell.panel_id,
                    'coordinates': cell.coordinates,
                    'temperature': cell.temperature,
                    'severity': cell.hotspot_severity,
                    'location_description': f"{cell.cell_id} at coordinates ({cell.coordinates[0]:.1f}m, {cell.coordinates[1]:.1f}m)"
                })
        
        # Generate summary statistics
        total_panels = len(solar_farm.panels)
        total_anomalous = len(anomalous_panels)
        total_hotspots = len(hotspot_cells)
        
        report = {
            'farm_summary': {
                'farm_id': solar_farm.farm_id,
                'total_panels': total_panels,
                'layout': f"{solar_farm.layout[0]} rows × {solar_farm.layout[1]} columns",
                'anomalous_panels': total_anomalous,
                'anomaly_rate': f"{(total_anomalous/total_panels)*100:.1f}%",
                'cell_hotspots': total_hotspots
            },
            'anomaly_breakdown': anomaly_counts,
            'anomalous_panels': anomalous_panels,
            'hotspot_cells': hotspot_cells,
            'priority_actions': []
        }
        
        # Generate priority actions
        critical_panels = [p for p in anomalous_panels if p['severity'] == 'critical']
        critical_cells = [c for c in hotspot_cells if c['severity'] == 'critical']
        
        if critical_panels or critical_cells:
            report['priority_actions'].append({
                'priority': 'IMMEDIATE',
                'action': 'Emergency inspection required',
                'items': [p['location_description'] for p in critical_panels] + 
                        [c['location_description'] for c in critical_cells]
            })
        
        severe_panels = [p for p in anomalous_panels if p['severity'] == 'severe']
        if severe_panels:
            report['priority_actions'].append({
                'priority': 'HIGH',
                'action': 'Maintenance within 24 hours',
                'items': [p['location_description'] for p in severe_panels]
            })
        
        return report

def main():
    """Main execution function for SAR-Thermal fusion system"""
    print("=== SAR-Thermal Fusion System for Cell-Level Solar Monitoring ===")
    
    # Initialize processors
    sar_processor = HighResSARProcessor(resolution=0.5)  # 0.5m resolution
    thermal_processor = ThermalProcessor(resolution=0.3)  # 0.3m resolution
    fusion_system = SARThermalFusion(sar_processor, thermal_processor)
    
    try:
        # Step 1: Generate high-resolution SAR data
        print("\n=== Step 1: High-Resolution SAR Processing ===")
        sar_image = sar_processor.create_mock_highres_sar()
        
        # Step 2: Detect panel boundaries
        print("\n=== Step 2: Panel Boundary Detection ===")
        detected_panels = sar_processor.detect_panel_boundaries(sar_image)
        
        # Step 3: Generate structured grid
        print("\n=== Step 3: Structured Grid Generation ===")
        solar_farm = sar_processor.generate_structured_grid(detected_panels)
        
        # Step 4: Generate thermal data
        print("\n=== Step 4: Thermal Data Processing ===")
        thermal_image = thermal_processor.create_mock_thermal_data(
            size=sar_image.shape, solar_farm=solar_farm)
        
        # Step 5: Register thermal to SAR
        print("\n=== Step 5: SAR-Thermal Registration ===")
        thermal_registered = fusion_system.register_thermal_to_sar(
            thermal_image, sar_image.shape)
        
        # Step 6: Extract panel temperatures
        print("\n=== Step 6: Panel Temperature Extraction ===")
        solar_farm = fusion_system.extract_panel_temperatures(thermal_registered, solar_farm)
        
        # Step 7: Detect cell-level hotspots
        print("\n=== Step 7: Cell-Level Hotspot Detection ===")
        hotspot_cells = fusion_system.detect_cell_level_hotspots(thermal_registered, solar_farm)
        
        # Step 8: Generate precision report
        print("\n=== Step 8: Precision Report Generation ===")
        report = fusion_system.generate_precision_report(solar_farm, hotspot_cells)
        
        # Print summary report
        print(f"\n=== SAR-Thermal Fusion Results ===")
        print(f"Farm: {report['farm_summary']['farm_id']}")
        print(f"Layout: {report['farm_summary']['layout']}")
        print(f"Total panels: {report['farm_summary']['total_panels']}")
        print(f"Anomalous panels: {report['farm_summary']['anomalous_panels']} ({report['farm_summary']['anomaly_rate']})")
        print(f"Cell hotspots: {report['farm_summary']['cell_hotspots']}")
        
        print(f"\nAnomaly breakdown:")
        for severity, count in report['anomaly_breakdown'].items():
            if count > 0:
                print(f"  {severity.capitalize()}: {count}")
        
        print(f"\nCritical issues requiring immediate attention:")
        for action in report['priority_actions']:
            if action['priority'] == 'IMMEDIATE':
                for item in action['items']:
                    print(f"  🚨 {item}")
        
        print(f"\nExample precision detection:")
        if report['anomalous_panels']:
            panel = report['anomalous_panels'][0]
            print(f"  Panel #{panel['panel_id']} at coordinates ({panel['coordinates'][0]:.1f}m, {panel['coordinates'][1]:.1f}m)")
            print(f"  Temperature: {panel['temperature']:.1f}°C (Severity: {panel['severity']})")
        
        if report['hotspot_cells']:
            cell = report['hotspot_cells'][0]
            print(f"  {cell['cell_id']} at coordinates ({cell['coordinates'][0]:.1f}m, {cell['coordinates'][1]:.1f}m)")
            print(f"  Temperature: {cell['temperature']:.1f}°C (Hotspot severity: {cell['severity']})")
        
        # Save report
        with open('sar_thermal_fusion_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nDetailed report saved to: sar_thermal_fusion_report.json")
        
    except Exception as e:
        print(f"Error during SAR-Thermal fusion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()