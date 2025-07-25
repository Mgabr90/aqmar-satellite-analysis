#!/usr/bin/env python3
"""
Multi-Sensor Solar Module Analysis System
Combines AQMAR optical imagery with Sentinel-1 SAR data for solar farm detection and anomaly analysis
"""

import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import json
from scipy import ndimage
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SolarInstallation:
    """Data class for detected solar installations"""
    bounds: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    center: Tuple[float, float]
    area: float
    confidence: float
    sensor_source: str  # 'optical', 'sar', or 'fusion'
    anomalies: List[Dict] = None

class MultiSensorSolarAnalyzer:
    """Multi-sensor solar farm detection and anomaly analysis system"""
    
    def __init__(self, data_dir: str, sar_dir: str = "sar_data"):
        self.data_dir = Path(data_dir)
        self.sar_dir = Path(sar_dir)
        self.optical_data = {}
        self.sar_data = {}
        self.solar_installations = []
        
    def load_aqmar_data(self) -> Dict:
        """Load AQMAR optical satellite data"""
        print("Loading AQMAR optical data...")
        
        # Find extracted directories
        dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        for data_folder in dirs:
            scene_id = data_folder.name
            
            # Load metadata
            meta_file = data_folder / f"{scene_id}.meta.xml"
            tiff_file = data_folder / f"{scene_id}.tif"
            
            if meta_file.exists() and tiff_file.exists():
                # Parse metadata
                with open(meta_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                tree = ET.fromstring(content)
                
                # Extract spatial information
                img_info = tree.find('ImageInfo')
                center_lat = float(img_info.find('CenterLocation/Latitude').text)
                center_lon = float(img_info.find('CenterLocation/Longitude').text)
                
                # Load TIFF imagery
                try:
                    with Image.open(tiff_file) as img:
                        image_array = np.array(img)
                        
                    self.optical_data[scene_id] = {
                        'image': image_array,
                        'center_lat': center_lat,
                        'center_lon': center_lon,
                        'metadata': tree,
                        'file_path': tiff_file
                    }
                    
                    print(f"Loaded optical scene: {scene_id}")
                    print(f"  Location: {center_lat:.3f}N, {center_lon:.3f}E")
                    print(f"  Image shape: {image_array.shape}")
                    
                except Exception as e:
                    print(f"Error loading TIFF {scene_id}: {e}")
        
        return self.optical_data
    
    def preprocess_optical_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess optical imagery for analysis"""
        # Convert to 8-bit if needed
        if image.dtype != np.uint8:
            # Normalize to 0-255 range
            img_normalized = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        else:
            img_normalized = image
            
        # Apply contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img_normalized)
        
        return enhanced
    
    def detect_geometric_patterns(self, image: np.ndarray, min_area: int = 500) -> List[Tuple]:
        """Detect rectangular geometric patterns typical of solar installations"""
        # Edge detection
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Morphological operations to connect edges
        kernel = np.ones((3,3), np.uint8)
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangular_regions = []
        
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's roughly rectangular (4-8 sides) and has sufficient area
            area = cv2.contourArea(contour)
            if len(approx) >= 4 and len(approx) <= 8 and area > min_area:
                # Calculate bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio (solar farms tend to be rectangular)
                aspect_ratio = w / h
                if 0.3 < aspect_ratio < 3.0:  # Reasonable aspect ratio range
                    rectangular_regions.append((x, y, w, h, area))
        
        return rectangular_regions
    
    def analyze_intensity_patterns(self, image: np.ndarray, regions: List[Tuple]) -> List[SolarInstallation]:
        """Analyze intensity patterns within detected regions"""
        installations = []
        
        for x, y, w, h, area in regions:
            # Extract region of interest
            roi = image[y:y+h, x:x+w]
            
            # Calculate statistical features
            mean_intensity = np.mean(roi)
            std_intensity = np.std(roi)
            
            # Look for periodic patterns (solar panel rows)
            # Simple approach: check for alternating dark/light bands
            row_means = np.mean(roi, axis=1)
            col_means = np.mean(roi, axis=0)
            
            # Calculate confidence based on pattern regularity
            row_variance = np.var(np.diff(row_means))
            col_variance = np.var(np.diff(col_means))
            pattern_score = 1.0 / (1.0 + row_variance + col_variance)
            
            # Additional criteria for solar installations
            # Solar panels typically have moderate reflectance
            intensity_score = 1.0 - abs(mean_intensity - 128) / 128
            
            # Combined confidence
            confidence = (pattern_score + intensity_score) / 2.0
            
            if confidence > 0.3:  # Threshold for solar installation
                installation = SolarInstallation(
                    bounds=(x, y, x+w, y+h),
                    center=(x + w/2, y + h/2),
                    area=area,
                    confidence=confidence,
                    sensor_source='optical'
                )
                installations.append(installation)
        
        return installations
    
    def detect_optical_solar_farms(self, scene_id: str) -> List[SolarInstallation]:
        """Main optical solar farm detection pipeline"""
        if scene_id not in self.optical_data:
            raise ValueError(f"Scene {scene_id} not loaded")
        
        image = self.optical_data[scene_id]['image']
        
        # Preprocess image
        processed_image = self.preprocess_optical_image(image)
        
        # Detect geometric patterns
        rectangular_regions = self.detect_geometric_patterns(processed_image)
        
        # Analyze patterns for solar characteristics
        installations = self.analyze_intensity_patterns(processed_image, rectangular_regions)
        
        print(f"Detected {len(installations)} potential solar installations in {scene_id}")
        
        return installations
    
    def detect_anomalies_optical(self, installation: SolarInstallation, image: np.ndarray) -> List[Dict]:
        """Detect anomalies within a solar installation using optical data"""
        x1, y1, x2, y2 = installation.bounds
        roi = image[y1:y2, x1:x2]
        
        anomalies = []
        
        # Statistical anomaly detection
        mean_val = np.mean(roi)
        std_val = np.std(roi)
        threshold = mean_val - 2 * std_val  # Dark regions (potential failures)
        
        # Find anomalously dark regions
        dark_mask = roi < threshold
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(dark_mask.astype(np.uint8))
        
        for label in range(1, num_labels):
            mask = (labels == label)
            area = np.sum(mask)
            
            if area > 50:  # Minimum anomaly size
                # Get bounding box of anomaly
                coords = np.where(mask)
                y_min, y_max = np.min(coords[0]), np.max(coords[0])
                x_min, x_max = np.min(coords[1]), np.max(coords[1])
                
                anomaly = {
                    'type': 'dark_region',
                    'bounds': (x1 + x_min, y1 + y_min, x1 + x_max, y1 + y_max),
                    'area': area,
                    'severity': 'medium' if area < 200 else 'high',
                    'sensor': 'optical'
                }
                anomalies.append(anomaly)
        
        return anomalies
    
    def load_sar_data(self) -> Dict:
        """Load Sentinel-1 SAR data"""
        print("Loading SAR data...")
        
        sar_regions = ['virginia', 'taiwan']
        
        for region in sar_regions:
            region_dir = self.sar_dir / region
            
            if region_dir.exists():
                try:
                    # Load SAR intensity data
                    vv_intensity = np.load(region_dir / 'vv_intensity.npy')
                    vh_intensity = np.load(region_dir / 'vh_intensity.npy')
                    vv_db = np.load(region_dir / 'vv_db.npy')
                    vh_db = np.load(region_dir / 'vh_db.npy')
                    
                    # Load metadata
                    with open(region_dir / 'metadata.json', 'r') as f:
                        metadata = json.load(f)
                    
                    self.sar_data[region] = {
                        'vv_intensity': vv_intensity,
                        'vh_intensity': vh_intensity,
                        'vv_db': vv_db,
                        'vh_db': vh_db,
                        'metadata': metadata
                    }
                    
                    print(f"Loaded SAR data for {region}: {vv_intensity.shape}")
                    
                except Exception as e:
                    print(f"Error loading SAR data for {region}: {e}")
        
        return self.sar_data
    
    def detect_sar_structures(self, vv_db: np.ndarray, vh_db: np.ndarray, 
                            min_backscatter: float = -15.0) -> List[Tuple]:
        """Detect geometric structures in SAR data using backscatter analysis"""
        
        # Create binary mask for high backscatter regions (potential metal structures)
        high_backscatter_vv = vv_db > min_backscatter
        high_backscatter_vh = vh_db > (min_backscatter - 5.0)  # VH typically lower
        
        # Combine polarizations - structures show up in both
        combined_mask = high_backscatter_vv | high_backscatter_vh
        
        # Morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask.astype(np.uint8), 
                                       cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(combined_mask)
        
        structures = []
        
        for label in range(1, num_labels):
            mask = (labels == label)
            area = np.sum(mask)
            
            if area > 100:  # Minimum structure size
                # Get bounding box
                coords = np.where(mask)
                y_min, y_max = np.min(coords[0]), np.max(coords[0])
                x_min, x_max = np.min(coords[1]), np.max(coords[1])
                
                # Calculate geometric properties
                width = x_max - x_min
                height = y_max - y_min
                aspect_ratio = width / height if height > 0 else 1
                
                # Solar installations tend to be rectangular
                if 0.2 < aspect_ratio < 5.0:
                    structures.append((x_min, y_min, width, height, area))
        
        return structures
    
    def analyze_sar_texture(self, vv_db: np.ndarray, structures: List[Tuple]) -> List[SolarInstallation]:
        """Analyze SAR texture patterns within detected structures"""
        installations = []
        
        for x, y, w, h, area in structures:
            # Extract region of interest
            roi_vv = vv_db[y:y+h, x:x+w]
            
            # Calculate texture features
            # Standard deviation as texture measure
            texture_std = np.std(roi_vv)
            
            # Local binary pattern-like analysis for regularity
            # Calculate variance of local means (indicates regular patterns)
            if roi_vv.shape[0] > 10 and roi_vv.shape[1] > 10:
                # Divide into small blocks and calculate variance of block means
                block_size = min(5, roi_vv.shape[0]//4, roi_vv.shape[1]//4)
                if block_size > 1:
                    block_means = []
                    for by in range(0, roi_vv.shape[0]-block_size, block_size):
                        for bx in range(0, roi_vv.shape[1]-block_size, block_size):
                            block = roi_vv[by:by+block_size, bx:bx+block_size]
                            block_means.append(np.mean(block))
                    
                    pattern_regularity = 1.0 / (1.0 + np.var(block_means))
                else:
                    pattern_regularity = 0.5
            else:
                pattern_regularity = 0.5
            
            # Mean backscatter level
            mean_backscatter = np.mean(roi_vv)
            
            # Confidence based on multiple factors
            # High backscatter + regular patterns = likely solar installation
            backscatter_score = (mean_backscatter + 30) / 30  # Normalize around -30 to 0 dB
            backscatter_score = max(0, min(1, backscatter_score))
            
            texture_score = min(1.0, texture_std / 10.0)  # Moderate texture expected
            
            confidence = (backscatter_score + pattern_regularity + texture_score) / 3.0
            
            if confidence > 0.4:  # Threshold for SAR detection
                installation = SolarInstallation(
                    bounds=(x, y, x+w, y+h),
                    center=(x + w/2, y + h/2),
                    area=area,
                    confidence=confidence,
                    sensor_source='sar'
                )
                installations.append(installation)
        
        return installations
    
    def detect_sar_solar_farms(self, region: str) -> List[SolarInstallation]:
        """Main SAR solar farm detection pipeline"""
        if region not in self.sar_data:
            print(f"No SAR data available for {region}")
            return []
        
        vv_db = self.sar_data[region]['vv_db']
        vh_db = self.sar_data[region]['vh_db']
        
        # Detect structures using backscatter
        structures = self.detect_sar_structures(vv_db, vh_db)
        
        # Analyze texture patterns
        installations = self.analyze_sar_texture(vv_db, structures)
        
        print(f"SAR detected {len(installations)} potential solar installations in {region}")
        
        return installations
    
    def detect_anomalies_sar(self, installation: SolarInstallation, 
                           vv_db: np.ndarray, vh_db: np.ndarray) -> List[Dict]:
        """Detect anomalies within a solar installation using SAR data"""
        x1, y1, x2, y2 = installation.bounds
        roi_vv = vv_db[y1:y2, x1:x2]
        roi_vh = vh_db[y1:y2, x1:x2]
        
        anomalies = []
        
        # Detect very low backscatter regions (potential failures)
        mean_vv = np.mean(roi_vv)
        std_vv = np.std(roi_vv)
        low_threshold = mean_vv - 2 * std_vv
        
        low_backscatter_mask = roi_vv < low_threshold
        
        # Find connected low-backscatter regions
        num_labels, labels = cv2.connectedComponents(low_backscatter_mask.astype(np.uint8))
        
        for label in range(1, num_labels):
            mask = (labels == label)
            area = np.sum(mask)
            
            if area > 20:  # Minimum anomaly size in SAR
                coords = np.where(mask)
                y_min, y_max = np.min(coords[0]), np.max(coords[0])
                x_min, x_max = np.min(coords[1]), np.max(coords[1])
                
                anomaly = {
                    'type': 'low_backscatter',
                    'bounds': (x1 + x_min, y1 + y_min, x1 + x_max, y1 + y_max),
                    'area': area,
                    'severity': 'medium' if area < 100 else 'high',
                    'sensor': 'sar'
                }
                anomalies.append(anomaly)
        
        # Detect texture anomalies (irregular patterns)
        # Calculate local texture variance
        if roi_vv.shape[0] > 20 and roi_vv.shape[1] > 20:
            texture_map = ndimage.generic_filter(roi_vv, np.std, size=5)
            texture_threshold = np.percentile(texture_map, 90)  # High texture regions
            
            high_texture_mask = texture_map > texture_threshold
            num_labels, labels = cv2.connectedComponents(high_texture_mask.astype(np.uint8))
            
            for label in range(1, num_labels):
                mask = (labels == label)
                area = np.sum(mask)
                
                if area > 30:
                    coords = np.where(mask)
                    y_min, y_max = np.min(coords[0]), np.max(coords[0])
                    x_min, x_max = np.min(coords[1]), np.max(coords[1])
                    
                    anomaly = {
                        'type': 'texture_anomaly',
                        'bounds': (x1 + x_min, y1 + y_min, x1 + x_max, y1 + y_max),
                        'area': area,
                        'severity': 'low',
                        'sensor': 'sar'
                    }
                    anomalies.append(anomaly)
        
        return anomalies
    
    def visualize_detections(self, scene_id: str, installations: List[SolarInstallation], 
                           save_path: Optional[str] = None):
        """Visualize detected solar installations and anomalies"""
        if scene_id not in self.optical_data:
            return
        
        image = self.optical_data[scene_id]['image']
        processed_image = self.preprocess_optical_image(image)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image
        ax1.imshow(processed_image, cmap='gray')
        ax1.set_title(f'Original Image - {scene_id}')
        ax1.axis('off')
        
        # Detections overlay
        ax2.imshow(processed_image, cmap='gray')
        
        for i, installation in enumerate(installations):
            x1, y1, x2, y2 = installation.bounds
            
            # Draw bounding box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='red', facecolor='none')
            ax2.add_patch(rect)
            
            # Add confidence label
            ax2.text(x1, y1-5, f'Solar {i+1}\nConf: {installation.confidence:.2f}', 
                    color='red', fontsize=8, weight='bold')
            
            # Draw anomalies if present
            if installation.anomalies:
                for anomaly in installation.anomalies:
                    ax1, ay1, ax2, ay2 = anomaly['bounds']
                    anomaly_rect = patches.Rectangle((ax1, ay1), ax2-ax1, ay2-ay1,
                                                   linewidth=1, edgecolor='yellow', facecolor='none')
                    ax2.add_patch(anomaly_rect)
        
        ax2.set_title(f'Solar Installations Detected: {len(installations)}')
        ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def fuse_detections(self, optical_installations: List[SolarInstallation], 
                       sar_installations: List[SolarInstallation], 
                       overlap_threshold: float = 0.3) -> List[SolarInstallation]:
        """Fuse optical and SAR detections using spatial overlap"""
        fused_installations = []
        used_sar_indices = set()
        
        for opt_inst in optical_installations:
            best_match = None
            best_overlap = 0
            best_sar_idx = -1
            
            # Find best matching SAR detection
            for sar_idx, sar_inst in enumerate(sar_installations):
                if sar_idx in used_sar_indices:
                    continue
                
                # Calculate overlap
                overlap = self.calculate_overlap(opt_inst.bounds, sar_inst.bounds)
                
                if overlap > best_overlap and overlap > overlap_threshold:
                    best_overlap = overlap
                    best_match = sar_inst
                    best_sar_idx = sar_idx
            
            if best_match:
                # Fuse the detections
                fused_confidence = (opt_inst.confidence + best_match.confidence) / 2.0
                
                # Use optical bounds as primary (typically more precise)
                fused_installation = SolarInstallation(
                    bounds=opt_inst.bounds,
                    center=opt_inst.center,
                    area=opt_inst.area,
                    confidence=fused_confidence,
                    sensor_source='fusion'
                )
                
                # Combine anomalies from both sensors
                all_anomalies = (opt_inst.anomalies or []) + (best_match.anomalies or [])
                fused_installation.anomalies = all_anomalies
                
                fused_installations.append(fused_installation)
                used_sar_indices.add(best_sar_idx)
            else:
                # Keep optical-only detection
                fused_installations.append(opt_inst)
        
        # Add SAR-only detections that weren't matched
        for sar_idx, sar_inst in enumerate(sar_installations):
            if sar_idx not in used_sar_indices:
                fused_installations.append(sar_inst)
        
        return fused_installations
    
    def calculate_overlap(self, bounds1: Tuple[int, int, int, int], 
                         bounds2: Tuple[int, int, int, int]) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bounds1
        x1_2, y1_2, x2_2, y2_2 = bounds2
        
        # Calculate intersection
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)
        
        if x2_int <= x1_int or y2_int <= y1_int:
            return 0.0
        
        intersection_area = (x2_int - x1_int) * (y2_int - y1_int)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def visualize_sar_detections(self, region: str, installations: List[SolarInstallation], 
                               save_path: Optional[str] = None):
        """Visualize SAR detections"""
        if region not in self.sar_data:
            return
        
        vv_db = self.sar_data[region]['vv_db']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # SAR image (VV polarization)
        im1 = ax1.imshow(vv_db, cmap='gray', vmin=-25, vmax=0)
        ax1.set_title(f'SAR VV (dB) - {region}')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Detections overlay
        im2 = ax2.imshow(vv_db, cmap='gray', vmin=-25, vmax=0)
        
        for i, installation in enumerate(installations):
            x1, y1, x2, y2 = installation.bounds
            
            # Draw bounding box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='red', facecolor='none')
            ax2.add_patch(rect)
            
            # Add confidence label
            ax2.text(x1, y1-5, f'SAR {i+1}\nConf: {installation.confidence:.2f}', 
                    color='red', fontsize=8, weight='bold')
            
            # Draw anomalies
            if installation.anomalies:
                for anomaly in installation.anomalies:
                    ax1_pos, ay1_pos, ax2_pos, ay2_pos = anomaly['bounds']
                    anomaly_rect = patches.Rectangle((ax1_pos, ay1_pos), ax2_pos-ax1_pos, ay2_pos-ay1_pos,
                                                   linewidth=1, edgecolor='yellow', facecolor='none')
                    ax2.add_patch(anomaly_rect)
        
        ax2.set_title(f'SAR Solar Installations: {len(installations)}')
        ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SAR visualization saved to: {save_path}")
        
        plt.show()
    
    def run_complete_analysis(self) -> Dict:
        """Run complete multi-sensor analysis pipeline"""
        print("=== Multi-Sensor Solar Analysis ===")
        
        # Load all data
        self.load_aqmar_data()
        self.load_sar_data()
        
        all_results = {}
        
        # Process optical data
        for scene_id in self.optical_data.keys():
            print(f"\nProcessing optical scene: {scene_id}")
            
            # Detect solar installations in optical data
            optical_installations = self.detect_optical_solar_farms(scene_id)
            
            # Detect anomalies in optical installations
            image = self.optical_data[scene_id]['image']
            processed_image = self.preprocess_optical_image(image)
            
            for installation in optical_installations:
                anomalies = self.detect_anomalies_optical(installation, processed_image)
                installation.anomalies = anomalies
            
            all_results[scene_id] = {
                'optical': optical_installations,
                'sar': [],
                'fused': optical_installations  # Default to optical if no SAR
            }
            
            # Visualize optical results
            output_path = f"optical_detection_{scene_id}.png"
            self.visualize_detections(scene_id, optical_installations, output_path)
        
        # Process SAR data
        for region in self.sar_data.keys():
            print(f"\nProcessing SAR region: {region}")
            
            # Detect solar installations in SAR data
            sar_installations = self.detect_sar_solar_farms(region)
            
            # Detect anomalies in SAR installations
            vv_db = self.sar_data[region]['vv_db']
            vh_db = self.sar_data[region]['vh_db']
            
            for installation in sar_installations:
                anomalies = self.detect_anomalies_sar(installation, vv_db, vh_db)
                installation.anomalies = anomalies
            
            # Store SAR results (create entry if doesn't exist)
            if region not in all_results:
                all_results[region] = {'optical': [], 'sar': sar_installations, 'fused': sar_installations}
            else:
                all_results[region]['sar'] = sar_installations
                
                # Fuse optical and SAR detections
                optical_insts = all_results[region]['optical']
                fused_installations = self.fuse_detections(optical_insts, sar_installations)
                all_results[region]['fused'] = fused_installations
            
            # Visualize SAR results
            output_path = f"sar_detection_{region}.png"
            self.visualize_sar_detections(region, sar_installations, output_path)
        
        return all_results
    
    def run_optical_analysis(self) -> Dict:
        """Run complete optical analysis pipeline"""
        # Load data
        self.load_aqmar_data()
        
        all_results = {}
        
        for scene_id in self.optical_data.keys():
            print(f"\nAnalyzing scene: {scene_id}")
            
            # Detect solar installations
            installations = self.detect_optical_solar_farms(scene_id)
            
            # Detect anomalies within installations
            image = self.optical_data[scene_id]['image']
            processed_image = self.preprocess_optical_image(image)
            
            for installation in installations:
                anomalies = self.detect_anomalies_optical(installation, processed_image)
                installation.anomalies = anomalies
                
            all_results[scene_id] = installations
            
            # Visualize results
            output_path = f"solar_detection_{scene_id}.png"
            self.visualize_detections(scene_id, installations, output_path)
            
        return all_results

def main():
    """Main execution function"""
    data_dir = "../Data"
    sar_dir = "sar_data"
    
    print("=== Multi-Sensor Solar Analysis System ===")
    
    analyzer = MultiSensorSolarAnalyzer(data_dir, sar_dir)
    
    try:
        # Run complete multi-sensor analysis
        results = analyzer.run_complete_analysis()
        
        # Summary statistics
        print(f"\n=== Multi-Sensor Analysis Summary ===")
        
        for region_id, region_results in results.items():
            optical_count = len(region_results.get('optical', []))
            sar_count = len(region_results.get('sar', []))
            fused_count = len(region_results.get('fused', []))
            
            # Count anomalies
            optical_anomalies = sum(len(inst.anomalies or []) for inst in region_results.get('optical', []))
            sar_anomalies = sum(len(inst.anomalies or []) for inst in region_results.get('sar', []))
            fused_anomalies = sum(len(inst.anomalies or []) for inst in region_results.get('fused', []))
            
            # Calculate average confidences
            opt_conf = np.mean([inst.confidence for inst in region_results.get('optical', [])]) if optical_count > 0 else 0
            sar_conf = np.mean([inst.confidence for inst in region_results.get('sar', [])]) if sar_count > 0 else 0
            fused_conf = np.mean([inst.confidence for inst in region_results.get('fused', [])]) if fused_count > 0 else 0
            
            print(f"\nRegion: {region_id}")
            print(f"  Optical detections: {optical_count} (conf: {opt_conf:.3f}, anomalies: {optical_anomalies})")
            print(f"  SAR detections: {sar_count} (conf: {sar_conf:.3f}, anomalies: {sar_anomalies})")
            print(f"  Fused detections: {fused_count} (conf: {fused_conf:.3f}, anomalies: {fused_anomalies})")
        
        # Overall summary
        total_optical = sum(len(r.get('optical', [])) for r in results.values())
        total_sar = sum(len(r.get('sar', [])) for r in results.values())
        total_fused = sum(len(r.get('fused', [])) for r in results.values())
        total_anomalies = sum(len(inst.anomalies or []) for r in results.values() 
                            for inst in r.get('fused', []))
        
        print(f"\n=== Overall Summary ===")
        print(f"Total optical detections: {total_optical}")
        print(f"Total SAR detections: {total_sar}")
        print(f"Total fused detections: {total_fused}")
        print(f"Total anomalies detected: {total_anomalies}")
        print(f"Regions analyzed: {len(results)}")
        
        # Detection method breakdown
        fusion_count = sum(1 for r in results.values() for inst in r.get('fused', []) 
                          if inst.sensor_source == 'fusion')
        optical_only = sum(1 for r in results.values() for inst in r.get('fused', []) 
                          if inst.sensor_source == 'optical')
        sar_only = sum(1 for r in results.values() for inst in r.get('fused', []) 
                      if inst.sensor_source == 'sar')
        
        print(f"\nDetection Source Breakdown:")
        print(f"  Multi-sensor fusion: {fusion_count}")
        print(f"  Optical only: {optical_only}")
        print(f"  SAR only: {sar_only}")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()