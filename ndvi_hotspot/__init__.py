"""
NDVI Hotspot Analysis Package
============================

A comprehensive Python package for analyzing vegetation patterns and detecting
hotspots using Landsat satellite imagery from Microsoft Planetary Computer.

This package provides tools for:
- Landsat data acquisition and preprocessing
- NDVI calculation and temporal analysis
- Hotspot detection using spatial statistics
- Visualization and export capabilities

Author: Rohit Khati
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Rohit Khati"
__email__ = "rhtkhati@gmail.com"

from .data_loader import LandsatDataLoader
from .ndvi_processor import NDVIProcessor
from .hotspot_analyzer import HotspotAnalyzer
from .visualizer import NDVIVisualizer
from .utils import get_sample_bbox, validate_date_range

__all__ = [
    'LandsatDataLoader',
    'NDVIProcessor', 
    'HotspotAnalyzer',
    'NDVIVisualizer',
    'get_sample_bbox',
    'validate_date_range'
]
