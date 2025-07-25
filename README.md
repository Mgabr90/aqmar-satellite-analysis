# AQMAR Satellite Data Analysis

Geospatial analysis tools for AQMAR satellite imagery with pixel-to-coordinate mapping capabilities.

## Overview

This repository contains tools for analyzing AS03 satellite data from the AQMAR mission, including:
- Satellite data diagnosis and metadata extraction
- Pixel-to-coordinate mapping for geospatial analysis
- Coordinate grid generation for georeferencing

## Datasets

### Virginia Coast Dataset
- **Location**: 36.88°N to 36.99°N, -76.40°W to -76.22°W
- **Date**: April 14, 2025
- **Sensor**: IRS (Infrared Scanner)
- **Resolution**: 1530×1001 pixels, 10m spatial resolution

### Taiwan Dataset  
- **Location**: 23.94°N to 24.06°N, 121.54°E to 121.69°E
- **Date**: March 27, 2025
- **Sensor**: IRS (Infrared Scanner)
- **Resolution**: 1496×1126 pixels, 10m spatial resolution

## Files Structure

```
├── Data/                           # Raw satellite data
│   ├── AS03_IRS_GE_DL_*.zip       # Virginia dataset
│   └── AS03_IRS_GE_KS_*.zip       # Taiwan dataset
├── analysis_outputs/              # Analysis results
│   ├── pixel_coordinates.py       # Coordinate mapping functions
│   ├── satellite_data_diagnosis.py # Data analysis tools
│   ├── virginia_latitude_grid.npy  # Virginia lat coordinates
│   ├── virginia_longitude_grid.npy # Virginia lon coordinates
│   ├── taiwan_latitude_grid.npy    # Taiwan lat coordinates
│   └── taiwan_longitude_grid.npy   # Taiwan lon coordinates
└── README.md                      # This file
```

## Usage

### Coordinate Mapping

```python
from analysis_outputs.pixel_coordinates import PixelCoordinateMapper

# Load mapper with metadata
mapper = PixelCoordinateMapper("metadata.xml")

# Convert pixel to coordinates
lat, lon = mapper.pixel_to_coordinates(100, 200)

# Convert coordinates to pixel
x, y = mapper.coordinates_to_pixel(lat, lon)

# Load coordinate grids
import numpy as np
lat_grid = np.load("analysis_outputs/virginia_latitude_grid.npy")
lon_grid = np.load("analysis_outputs/virginia_longitude_grid.npy")
```

### Data Diagnosis

```python
# Run diagnosis script
python analysis_outputs/satellite_data_diagnosis.py
```

## Requirements

- Python 3.7+
- NumPy
- Pillow (PIL)
- xml.etree.ElementTree

## Applications

- Solar panel detection and monitoring
- Agricultural monitoring
- Urban planning
- Environmental monitoring
- Geospatial analysis

## License

MIT License