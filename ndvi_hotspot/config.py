"""
Configuration Module
====================

Configuration settings and constants for the NDVI Hotspot Analysis package.
"""

# Package metadata
PACKAGE_NAME = "NDVI-Hotspot"
VERSION = "1.0.0"
AUTHOR = "Rohit Khati"
EMAIL = "rhtkhati@gmail.com"

# Default analysis parameters
DEFAULT_CLOUD_COVER = 30.0
DEFAULT_RESOLUTION = 30.0  # meters
DEFAULT_KERNEL_RADIUS = 1.5
DEFAULT_CHUNK_SIZE = 256

# Landsat collections
LANDSAT_COLLECTIONS = {
    "landsat-c2-l2": "Landsat Collection 2 Level-2",
    "landsat-8-c2-l2": "Landsat 8 Collection 2 Level-2", 
    "landsat-9-c2-l2": "Landsat 9 Collection 2 Level-2"
}

# Default band assets for NDVI analysis
DEFAULT_ASSETS = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "ST_B10"]

# NDVI validation thresholds
NDVI_MIN = -1.0
NDVI_MAX = 1.0

# Hotspot significance levels
HOTSPOT_LEVELS = {
    -99: "Very Cold Spot (99% confidence)",
    -95: "Cold Spot (95% confidence)", 
    -90: "Cold Spot (90% confidence)",
    0: "Not Significant",
    90: "Hot Spot (90% confidence)",
    95: "Hot Spot (95% confidence)",
    99: "Very Hot Spot (99% confidence)"
}

# Default color schemes
HOTSPOT_COLORS = {
    -99: "#0000FF",  # Deep blue
    -95: "#4169E1",  # Blue
    -90: "#87CEEB",  # Light blue
    0: "#FFFFFF",    # White
    90: "#FFA500",   # Orange
    95: "#FF4500",   # Red orange
    99: "#FF0000"    # Red
}

# Microsoft Planetary Computer STAC endpoint
STAC_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Supported CRS codes
SUPPORTED_CRS = [4326, 3857, 32633, 32634]  # WGS84, Web Mercator, UTM zones

# Memory management
MAX_MEMORY_GB = 8.0
CHUNK_SIZE_MB = 128
