#!/usr/bin/env python3
"""
SAR-Thermal Fusion Visualization System
Creates comprehensive visualizations for cell-level solar monitoring results
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import seaborn as sns

class FusionVisualizer:
    """Visualization system for SAR-thermal fusion results"""
    
    def __init__(self):
        # Define color schemes
        self.thermal_cmap = plt.cm.hot
        self.sar_cmap = plt.cm.gray
        
        # Anomaly severity colors
        self.severity_colors = {
            'normal': 'green',
            'minor': 'yellow',
            'moderate': 'orange', 
            'severe': 'red',
            'critical': 'darkred'
        }
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def create_comprehensive_dashboard(self, sar_image: np.ndarray, 
                                     thermal_image: np.ndarray,
                                     solar_farm, hotspot_cells: List,
                                     report: Dict, save_path: str = "fusion_dashboard.png"):
        """Create comprehensive monitoring dashboard"""
        
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. SAR Image with Panel Detection
        ax1 = fig.add_subplot(gs[0, 0])
        sar_db = 10 * np.log10(sar_image + 1e-10)
        im1 = ax1.imshow(sar_db, cmap='gray', vmin=-30, vmax=0)
        self._overlay_panel_boundaries(ax1, solar_farm, scale=0.5)
        ax1.set_title('SAR: Panel Boundaries')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Backscatter (dB)')
        
        # 2. Thermal Image with Temperature Scale
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(thermal_image, cmap='hot', vmin=25, vmax=90)
        ax2.set_title('Thermal: Temperature Map')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Temperature (°C)')
        
        # 3. Fused Results: Anomaly Overlay
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(thermal_image, cmap='hot', vmin=25, vmax=90, alpha=0.7)
        self._overlay_anomalies(ax3, solar_farm, scale=0.3)
        ax3.set_title('Fusion: Anomaly Detection')
        ax3.axis('off')
        
        # 4. Cell-Level Hotspots
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(thermal_image, cmap='hot', vmin=25, vmax=90, alpha=0.5)
        self._overlay_hotspot_cells(ax4, hotspot_cells, scale=0.3)
        ax4.set_title('Cell-Level Hotspots')
        ax4.axis('off')
        
        # 5. Temperature Distribution Histogram
        ax5 = fig.add_subplot(gs[1, 0])
        panel_temps = [p.temperature for p in solar_farm.panels if p.temperature is not None]
        ax5.hist(panel_temps, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax5.axvline(x=50, color='orange', linestyle='--', label='Normal threshold')
        ax5.axvline(x=75, color='red', linestyle='--', label='Critical threshold')
        ax5.set_xlabel('Temperature (°C)')
        ax5.set_ylabel('Number of Panels')
        ax5.set_title('Panel Temperature Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Anomaly Severity Breakdown
        ax6 = fig.add_subplot(gs[1, 1])
        severity_counts = report['anomaly_breakdown']
        colors = [self.severity_colors[sev] for sev in severity_counts.keys()]
        wedges, texts, autotexts = ax6.pie(severity_counts.values(), 
                                          labels=severity_counts.keys(),
                                          colors=colors, autopct='%1.1f%%')
        ax6.set_title('Anomaly Severity Breakdown')
        
        # 7. Panel Grid Layout with Color Coding
        ax7 = fig.add_subplot(gs[1, 2:])
        self._create_panel_grid_view(ax7, solar_farm)
        ax7.set_title('Solar Farm Layout - Color Coded by Temperature')
        
        # 8. Priority Actions Table
        ax8 = fig.add_subplot(gs[2, :])
        self._create_priority_table(ax8, report)
        ax8.set_title('Priority Actions and Maintenance Schedule')
        
        # Add main title
        fig.suptitle('SAR-Thermal Fusion: Cell-Level Solar Monitoring Dashboard', 
                    fontsize=16, fontweight='bold')
        
        # Add timestamp and summary stats
        summary_text = f"""
        Total Panels: {report['farm_summary']['total_panels']} | 
        Anomalous: {report['farm_summary']['anomalous_panels']} ({report['farm_summary']['anomaly_rate']}) | 
        Cell Hotspots: {report['farm_summary']['cell_hotspots']}
        """
        fig.text(0.5, 0.02, summary_text, ha='center', fontsize=10)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive dashboard saved: {save_path}")
        plt.show()
        
        return fig
    
    def _overlay_panel_boundaries(self, ax, solar_farm, scale: float = 1.0):
        """Overlay panel boundaries on image"""
        for panel in solar_farm.panels:
            x_min, y_min, x_max, y_max = panel.bounds
            # Convert to pixel coordinates
            x_min_px = x_min / scale
            y_min_px = y_min / scale
            width_px = (x_max - x_min) / scale
            height_px = (y_max - y_min) / scale
            
            rect = patches.Rectangle((x_min_px, y_min_px), width_px, height_px,
                                   linewidth=0.5, edgecolor='cyan', facecolor='none')
            ax.add_patch(rect)
    
    def _overlay_anomalies(self, ax, solar_farm, scale: float = 1.0):
        """Overlay anomalous panels with color coding"""
        for panel in solar_farm.panels:
            if panel.is_anomalous:
                x_min, y_min, x_max, y_max = panel.bounds
                x_min_px = x_min / scale
                y_min_px = y_min / scale
                width_px = (x_max - x_min) / scale
                height_px = (y_max - y_min) / scale
                
                color = self.severity_colors[panel.anomaly_severity]
                rect = patches.Rectangle((x_min_px, y_min_px), width_px, height_px,
                                       linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                
                # Add panel ID
                ax.text(x_min_px + width_px/2, y_min_px + height_px/2, 
                       str(panel.panel_id), color='white', fontsize=6, 
                       ha='center', va='center', weight='bold')
    
    def _overlay_hotspot_cells(self, ax, hotspot_cells: List, scale: float = 1.0):
        """Overlay cell-level hotspots"""
        for cell in hotspot_cells:
            if cell.is_hotspot:
                x, y = cell.coordinates
                x_px = x / scale
                y_px = y / scale
                
                color = self.severity_colors[cell.hotspot_severity]
                circle = patches.Circle((x_px, y_px), radius=3, 
                                      color=color, alpha=0.8)
                ax.add_patch(circle)
    
    def _create_panel_grid_view(self, ax, solar_farm):
        """Create grid view of solar farm with temperature color coding"""
        rows, cols = solar_farm.layout
        
        # Create temperature matrix
        temp_matrix = np.full((rows, cols), np.nan)
        
        for panel in solar_farm.panels:
            if panel.temperature is not None and panel.row > 0 and panel.col > 0:
                temp_matrix[panel.row-1, panel.col-1] = panel.temperature
        
        # Create heatmap
        mask = np.isnan(temp_matrix)
        im = ax.imshow(temp_matrix, cmap='RdYlBu_r', vmin=40, vmax=80, 
                      aspect='auto', interpolation='nearest')
        
        # Add grid
        ax.set_xticks(np.arange(cols))
        ax.set_yticks(np.arange(rows))
        ax.set_xticklabels(np.arange(1, cols+1))
        ax.set_yticklabels(np.arange(1, rows+1))
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Temperature (°C)')
        
        # Mark anomalous panels
        for panel in solar_farm.panels:
            if panel.is_anomalous and panel.row > 0 and panel.col > 0:
                ax.add_patch(patches.Rectangle((panel.col-1.5, panel.row-1.5), 1, 1,
                                             fill=False, edgecolor='red', linewidth=2))
        
        ax.set_xlabel('Panel Column')
        ax.set_ylabel('Panel Row')
    
    def _create_priority_table(self, ax, report: Dict):
        """Create priority actions table"""
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        
        # Add header
        table_data.append(['Priority', 'Action', 'Location', 'Details'])
        
        # Add critical items
        for action in report['priority_actions']:
            for i, item in enumerate(action['items']):
                if i == 0:
                    table_data.append([action['priority'], action['action'], item, ''])
                else:
                    table_data.append(['', '', item, ''])
        
        # Add some anomalous panels for reference
        for i, panel in enumerate(report['anomalous_panels'][:5]):  # Top 5
            severity_emoji = {'critical': '🔥', 'severe': '⚠️', 'moderate': '⚡', 'minor': '💡'}
            emoji = severity_emoji.get(panel['severity'], '')
            location = f"Panel #{panel['panel_id']} ({panel['coordinates'][0]:.1f}m, {panel['coordinates'][1]:.1f}m)"
            details = f"{panel['temperature']:.1f}°C {emoji}"
            table_data.append(['Monitor', 'Temperature check', location, details])
        
        if table_data:
            table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                           cellLoc='left', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 2)
            
            # Color code rows by priority
            for i in range(1, len(table_data)):
                if i < len(table_data):
                    if 'IMMEDIATE' in table_data[i][0]:
                        for j in range(4):
                            table[(i, j)].set_facecolor('#ffcccc')  # Light red
                    elif 'HIGH' in table_data[i][0]:
                        for j in range(4):
                            table[(i, j)].set_facecolor('#ffffcc')  # Light yellow
    
    def create_detailed_anomaly_view(self, thermal_image: np.ndarray, 
                                   anomalous_panels: List, save_path: str = "anomaly_detail.png"):
        """Create detailed view of anomalous panels"""
        
        if not anomalous_panels:
            print("No anomalous panels to visualize")
            return
        
        n_panels = min(len(anomalous_panels), 6)  # Show up to 6 panels
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        thermal_scale = 0.3  # Thermal resolution
        
        for i in range(n_panels):
            panel_info = anomalous_panels[i]
            ax = axes[i]
            
            # Extract panel region from thermal image
            x_center, y_center = panel_info['coordinates']
            panel_size = 50  # pixels around panel center
            
            x_min = max(0, int(x_center / thermal_scale) - panel_size)
            x_max = min(thermal_image.shape[1], int(x_center / thermal_scale) + panel_size)
            y_min = max(0, int(y_center / thermal_scale) - panel_size)
            y_max = min(thermal_image.shape[0], int(y_center / thermal_scale) + panel_size)
            
            panel_region = thermal_image[y_min:y_max, x_min:x_max]
            
            im = ax.imshow(panel_region, cmap='hot', vmin=25, vmax=90)
            ax.set_title(f"Panel #{panel_info['panel_id']}\n{panel_info['temperature']:.1f}°C - {panel_info['severity']}")
            ax.axis('off')
            
            # Add temperature scale
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for i in range(n_panels, len(axes)):
            axes[i].axis('off')
        
        fig.suptitle('Detailed View: Anomalous Solar Panels', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Detailed anomaly view saved: {save_path}")
        plt.show()

def main():
    """Test the visualization system with sample data"""
    print("=== SAR-Thermal Fusion Visualization System ===")
    
    # Load report if available
    report_file = Path("sar_thermal_fusion_report.json")
    if report_file.exists():
        with open(report_file, 'r') as f:
            report = json.load(f)
        
        print("Loaded existing fusion report")
        print(f"Anomalous panels: {len(report.get('anomalous_panels', []))}")
        print(f"Hotspot cells: {len(report.get('hotspot_cells', []))}")
        
        # Create sample images for visualization
        sar_image = np.random.gamma(2, 0.01, (2000, 3000))
        thermal_image = np.random.normal(45, 10, (2000, 3000))
        thermal_image = np.clip(thermal_image, 25, 90)
        
        # Mock solar farm for visualization
        class MockPanel:
            def __init__(self, data):
                self.panel_id = data['panel_id']
                self.row = data['row']
                self.col = data['col']
                self.coordinates = data['coordinates']
                self.bounds = (data['coordinates'][0]-1, data['coordinates'][1]-0.5,
                             data['coordinates'][0]+1, data['coordinates'][1]+0.5)
                self.temperature = data['temperature']
                self.is_anomalous = data['severity'] != 'normal'
                self.anomaly_severity = data['severity']
        
        class MockFarm:
            def __init__(self):
                self.layout = (20, 50)
                self.panels = [MockPanel(p) for p in report.get('anomalous_panels', [])]
        
        class MockCell:
            def __init__(self, data):
                self.cell_id = data['cell_id']
                self.coordinates = data['coordinates']
                self.temperature = data['temperature']
                self.is_hotspot = True
                self.hotspot_severity = data['severity']
        
        solar_farm = MockFarm()
        hotspot_cells = [MockCell(c) for c in report.get('hotspot_cells', [])]
        
        # Create visualizations
        visualizer = FusionVisualizer()
        
        # Comprehensive dashboard
        visualizer.create_comprehensive_dashboard(sar_image, thermal_image, 
                                                solar_farm, hotspot_cells, report)
        
        # Detailed anomaly view
        if report.get('anomalous_panels'):
            visualizer.create_detailed_anomaly_view(thermal_image, 
                                                  report['anomalous_panels'])
        
    else:
        print("No fusion report found. Run sar_thermal_fusion.py first.")

if __name__ == "__main__":
    main()