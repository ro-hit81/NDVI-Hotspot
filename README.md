# NDVI Hotspot Analysis

A professional Python package for analyzing vegetation indices and detecting hotspots in satellite imagery using advanced geospatial techniques and Microsoft Planetary Computer.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Remote Sensing](https://img.shields.io/badge/remote%20sensing-NDVI-orange.svg)
![Satellite Data](https://img.shields.io/badge/satellite-Landsat-red.svg)

## 🌱 Overview

This package provides a comprehensive suite of tools for vegetation analysis using satellite imagery. It specializes in:

- **NDVI (Normalized Difference Vegetation Index) Calculation**: Compute vegetation indices from multi-spectral satellite data
- **Temporal Analysis**: Analyze vegetation changes over time with statistical insights
- **Hotspot Detection**: Identify spatial clusters of high/low vegetation using Getis-Ord Gi* statistics
- **Advanced Visualization**: Create publication-ready maps and time series plots
- **Data Integration**: Seamless access to Landsat data via Microsoft Planetary Computer

## 🚀 Features

### Data Processing
- ✅ Automated Landsat data retrieval from Microsoft Planetary Computer
- ✅ Cloud masking and quality filtering
- ✅ Multi-temporal data stacking and alignment
- ✅ NDVI calculation with temporal statistics

### Spatial Analysis
- ✅ Getis-Ord Gi* hotspot analysis
- ✅ Focal statistics and neighborhood operations
- ✅ Edge detection using Sobel filters
- ✅ Hotspot persistence tracking

### Visualization
- ✅ Interactive time series plots
- ✅ Hotspot maps with custom colormaps
- ✅ True and false color composites
- ✅ Statistical distribution plots
- ✅ Publication-ready figures

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Install from Source
```bash
# Clone the repository
git clone https://github.com/ro-hit81/NDVI-Hotspot.git
cd NDVI-Hotspot

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Install in Development Mode
```bash
pip install -e .
```

## 🏃‍♂️ Quick Start

### Basic NDVI Analysis

```python
from ndvi_hotspot import LandsatDataLoader, NDVIProcessor, HotspotAnalyzer, NDVIVisualizer
from datetime import datetime
import matplotlib.pyplot as plt

# Define area of interest (bbox: [minx, miny, maxx, maxy])
bbox = [-120.5, 35.5, -120.0, 36.0]  # Central California
start_date = datetime(2023, 4, 1)
end_date = datetime(2023, 9, 30)

# Load satellite data
loader = LandsatDataLoader()
landsat_data = loader.search_and_load(
    bbox=bbox,
    start_date=start_date,
    end_date=end_date,
    max_cloud_cover=20
)

# Calculate NDVI
processor = NDVIProcessor()
ndvi_data = processor.calculate_ndvi(landsat_data)

# Analyze temporal statistics
ndvi_stats = processor.calculate_temporal_stats(ndvi_data)

# Detect hotspots
analyzer = HotspotAnalyzer()
hotspots = analyzer.calculate_hotspots(ndvi_stats['mean'])

# Visualize results
visualizer = NDVIVisualizer()
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot NDVI time series
visualizer.plot_ndvi_timeseries(ndvi_data, ax=axes[0,0])

# Plot mean NDVI
visualizer.plot_ndvi_map(ndvi_stats['mean'], ax=axes[0,1], title='Mean NDVI')

# Plot hotspots
visualizer.plot_hotspot_map(hotspots, ax=axes[1,0])

# Plot NDVI histogram
visualizer.plot_ndvi_histogram(ndvi_stats['mean'], ax=axes[1,1])

plt.tight_layout()
plt.show()
```

### Advanced Hotspot Analysis

```python
from ndvi_hotspot import HotspotAnalyzer
import numpy as np

# Initialize analyzer with custom parameters
analyzer = HotspotAnalyzer(
    neighborhood_size=5,
    significance_level=0.05
)

# Calculate hotspots with different kernel types
hotspots_circular = analyzer.calculate_hotspots(
    ndvi_stats['mean'], 
    kernel_type='circular'
)

hotspots_square = analyzer.calculate_hotspots(
    ndvi_stats['mean'], 
    kernel_type='square'
)

# Analyze hotspot persistence over time
ndvi_timeseries = processor.get_temporal_data(ndvi_data)
persistence = analyzer.analyze_hotspot_persistence(ndvi_timeseries)

# Get summary statistics
stats = analyzer.get_hotspot_stats(hotspots_circular)
print(f"Hot spots: {stats['hot_spots']} pixels")
print(f"Cold spots: {stats['cold_spots']} pixels")
print(f"Significant ratio: {stats['significant_ratio']:.2%}")
```

## 📊 Core Classes

### LandsatDataLoader
Handles data acquisition from Microsoft Planetary Computer's STAC catalog.

```python
loader = LandsatDataLoader()

# Search by coordinates and date range
data = loader.search_and_load(
    bbox=[-120.5, 35.5, -120.0, 36.0],
    start_date=datetime(2023, 4, 1),
    end_date=datetime(2023, 9, 30),
    max_cloud_cover=20
)

# Search by specific scene IDs
scene_ids = ['LC08_L2SP_043034_20230401_02_T1']
data = loader.load_by_scene_ids(scene_ids)
```

### NDVIProcessor
Calculates vegetation indices and temporal statistics.

```python
processor = NDVIProcessor()

# Calculate NDVI from Landsat data
ndvi = processor.calculate_ndvi(landsat_data)

# Calculate temporal statistics
stats = processor.calculate_temporal_stats(ndvi)
# Returns: mean, std, min, max, median, percentile_25, percentile_75

# Detect vegetation anomalies
anomalies = processor.detect_vegetation_anomalies(ndvi, threshold=2.0)
```

### HotspotAnalyzer
Performs spatial hotspot analysis using Getis-Ord Gi* statistics.

```python
analyzer = HotspotAnalyzer(neighborhood_size=3)

# Calculate hotspots
hotspots = analyzer.calculate_hotspots(ndvi_mean)

# Analyze persistence over time
persistence = analyzer.analyze_hotspot_persistence(ndvi_timeseries)

# Apply Sobel edge detection
edges = analyzer.sobel_edge_detection(ndvi_mean)
```

### NDVIVisualizer
Creates comprehensive visualizations for analysis results.

```python
visualizer = NDVIVisualizer()

# Plot NDVI time series
visualizer.plot_ndvi_timeseries(ndvi_data)

# Create hotspot maps
visualizer.plot_hotspot_map(hotspots, title='NDVI Hotspots')

# Generate true color composites
visualizer.plot_true_color(landsat_data, bands=['red', 'green', 'blue'])

# Plot statistical distributions
visualizer.plot_ndvi_histogram(ndvi_data)
```

## 🗂️ Package Structure

```
NDVI-Hotspot/
├── README.md                 # This file
├── requirements.txt          # Package dependencies
├── example_analysis.py       # Complete workflow example
├── Untitled.ipynb          # Original analysis notebook
└── ndvi_hotspot/            # Main package
    ├── __init__.py          # Package initialization
    ├── data_loader.py       # Landsat data acquisition
    ├── ndvi_processor.py    # NDVI calculation and analysis
    ├── hotspot_analyzer.py  # Spatial hotspot detection
    ├── visualizer.py        # Plotting and visualization
    ├── utils.py             # Utility functions
    └── config.py            # Configuration and constants
```

## 🔧 Configuration

The package includes configurable parameters in `ndvi_hotspot/config.py`:

```python
# Hotspot analysis parameters
HOTSPOT_SIGNIFICANCE_LEVELS = [0.01, 0.05, 0.1]
DEFAULT_NEIGHBORHOOD_SIZE = 3
EDGE_DETECTION_THRESHOLD = 0.1

# Color schemes for visualization
NDVI_COLORMAP = 'RdYlGn'
HOTSPOT_COLORS = {
    'hot': '#d7191c',
    'cold': '#2c7bb6',
    'not_significant': '#ffffbf'
}

# Memory management
MAX_MEMORY_GB = 8
CHUNK_SIZE_MB = 100
```

## 📈 Example Outputs

The package generates various types of analysis outputs:

1. **NDVI Time Series**: Track vegetation changes over time
2. **Hotspot Maps**: Spatial clusters of high/low vegetation
3. **Statistical Summaries**: Comprehensive vegetation statistics
4. **Color Composites**: True and false color satellite imagery
5. **Persistence Analysis**: Temporal stability of hotspots

## 🌍 Use Cases

### Environmental Monitoring
- Track deforestation and reforestation
- Monitor agricultural crop health
- Assess drought impacts on vegetation

### Agricultural Applications
- Precision agriculture and yield prediction
- Irrigation management
- Crop stress detection

### Research Applications
- Climate change impact studies
- Biodiversity assessments
- Land use change analysis

## 🔬 Scientific Background

### NDVI (Normalized Difference Vegetation Index)
NDVI is calculated as:
```
NDVI = (NIR - Red) / (NIR + Red)
```
Where NIR is near-infrared and Red is red band reflectance.

### Getis-Ord Gi* Statistic
The Gi* statistic identifies spatial clusters:
```
Gi* = (Σ wij * xj - X̄ * Σ wij) / (S * √[(n * Σ wij² - (Σ wij)²) / (n-1)])
```
Where wij are spatial weights and xj are attribute values.

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Format code
black ndvi_hotspot/

# Type checking
mypy ndvi_hotspot/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## 🙏 Acknowledgments

- **Microsoft Planetary Computer** for providing free access to Landsat data
- **stackstac** team for efficient satellite data processing tools
- **xrspatial** developers for spatial analysis capabilities
- **Python geospatial community** for the excellent ecosystem of tools

## 📞 Support

For questions, issues, or feature requests:

1. Check the [Issues](https://github.com/ro-hit81/NDVI-Hotspot/issues) page
2. Create a new issue with detailed description
3. Contact the maintainers

---

**Happy analyzing! 🛰️🌱**

*This package demonstrates professional Python development practices for geospatial analysis and remote sensing applications.*
