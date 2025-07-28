"""
Utility Functions Module
========================

This module provides utility functions for the NDVI Hotspot analysis package.

Features:
- Data validation and preprocessing
- Coordinate and date utilities
- Sample data generation
- Helper functions for analysis
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from datetime import datetime, date
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_bbox(bbox: List[float]) -> bool:
    """
    Validate bounding box coordinates.
    
    Args:
        bbox (List[float]): Bounding box [min_lon, min_lat, max_lon, max_lat]
        
    Returns:
        bool: True if valid, False otherwise
    """
    if len(bbox) != 4:
        logger.error("Bounding box must have exactly 4 coordinates")
        return False
    
    min_lon, min_lat, max_lon, max_lat = bbox
    
    # Check longitude range
    if not (-180 <= min_lon <= 180) or not (-180 <= max_lon <= 180):
        logger.error("Longitude values must be between -180 and 180")
        return False
    
    # Check latitude range
    if not (-90 <= min_lat <= 90) or not (-90 <= max_lat <= 90):
        logger.error("Latitude values must be between -90 and 90")
        return False
    
    # Check that min < max
    if min_lon >= max_lon:
        logger.error("Minimum longitude must be less than maximum longitude")
        return False
    
    if min_lat >= max_lat:
        logger.error("Minimum latitude must be less than maximum latitude")
        return False
    
    logger.info("Bounding box validation passed")
    return True


def validate_date_range(start_date: str, end_date: str) -> bool:
    """
    Validate date range format and logic.
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        if start_dt >= end_dt:
            logger.error("Start date must be before end date")
            return False
        
        # Check if dates are reasonable (not too far in future or past)
        current_date = datetime.now()
        min_date = datetime(1980, 1, 1)  # Approximate start of Landsat era
        
        if start_dt < min_date or end_dt > current_date:
            logger.warning(f"Date range outside typical Landsat availability ({min_date.date()} to {current_date.date()})")
        
        logger.info(f"Date range validation passed: {start_date} to {end_date}")
        return True
        
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        return False


def get_sample_bbox(region: str = "austria") -> List[float]:
    """
    Get sample bounding box coordinates for different regions.
    
    Args:
        region (str): Region name ('austria', 'nepal', 'california', 'amazon')
        
    Returns:
        List[float]: Bounding box coordinates [min_lon, min_lat, max_lon, max_lat]
    """
    sample_regions = {
        "austria": [13.324065832846372, 47.790832246642026, 13.457618384115904, 47.867354172891204],
        "nepal": [80.0, 26.0, 88.0, 30.5],
        "california": [-124.0, 32.0, -114.0, 42.0],
        "amazon": [-74.0, -18.0, -44.0, 5.0],
        "sahel": [-18.0, 10.0, 22.0, 18.0],
        "congo": [12.0, -13.0, 31.0, 5.0]
    }
    
    bbox = sample_regions.get(region.lower())
    if bbox is None:
        logger.warning(f"Unknown region '{region}', using Austria as default")
        bbox = sample_regions["austria"]
    
    logger.info(f"Retrieved sample bbox for {region}: {bbox}")
    return bbox


def format_date_for_stac(date_str: str) -> str:
    """
    Format date string for STAC API queries.
    
    Args:
        date_str (str): Date in 'YYYY-MM-DD' format
        
    Returns:
        str: Formatted date string
    """
    # Validate the date format
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return date_str
    except ValueError:
        logger.error(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")
        raise


def calculate_area_km2(bbox: List[float]) -> float:
    """
    Calculate approximate area of bounding box in km².
    
    Args:
        bbox (List[float]): Bounding box [min_lon, min_lat, max_lon, max_lat]
        
    Returns:
        float: Area in square kilometers
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    
    # Convert degrees to kilometers (approximate)
    lat_km = (max_lat - min_lat) * 111.0  # 1 degree latitude ≈ 111 km
    
    # Longitude varies with latitude
    avg_lat = (min_lat + max_lat) / 2
    lon_km = (max_lon - min_lon) * 111.0 * np.cos(np.radians(avg_lat))
    
    area_km2 = lat_km * lon_km
    
    logger.info(f"Calculated area: {area_km2:.2f} km²")
    return area_km2


def get_optimal_resolution(bbox: List[float], target_pixels: int = 1000) -> float:
    """
    Calculate optimal resolution based on area and target number of pixels.
    
    Args:
        bbox (List[float]): Bounding box coordinates
        target_pixels (int): Target number of pixels per dimension
        
    Returns:
        float: Optimal resolution in degrees
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat
    
    # Calculate resolution based on target pixels
    lon_res = lon_range / target_pixels
    lat_res = lat_range / target_pixels
    
    # Use the larger resolution to ensure we don't exceed target
    optimal_res = max(lon_res, lat_res)
    
    logger.info(f"Calculated optimal resolution: {optimal_res:.6f} degrees")
    return optimal_res


def create_landsat_scene_id(path: int, row: int, date: str, collection: str = "02", 
                          tier: str = "T1", satellite: str = "LC08") -> str:
    """
    Create Landsat scene ID from components.
    
    Args:
        path (int): WRS path number
        row (int): WRS row number
        date (str): Date in 'YYYY-MM-DD' format
        collection (str): Collection number
        tier (str): Data tier (T1, T2, RT)
        satellite (str): Satellite identifier (LC08, LC09)
        
    Returns:
        str: Formatted Landsat scene ID
    """
    # Convert date format
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    date_formatted = date_obj.strftime('%Y%m%d')
    
    # Format path and row with leading zeros
    path_str = str(path).zfill(3)
    row_str = str(row).zfill(3)
    
    scene_id = f"{satellite}_L2SP_{path_str}{row_str}_{date_formatted}_{collection}_{tier}"
    
    logger.info(f"Created scene ID: {scene_id}")
    return scene_id


def filter_scene_ids_by_date(scene_ids: List[str], 
                           start_date: str, 
                           end_date: str) -> List[str]:
    """
    Filter Landsat scene IDs by date range.
    
    Args:
        scene_ids (List[str]): List of Landsat scene IDs
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        
    Returns:
        List[str]: Filtered scene IDs
    """
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    filtered_ids = []
    
    for scene_id in scene_ids:
        try:
            # Extract date from scene ID (assumes standard format)
            # Format: LC08_L2SP_PPPRRR_YYYYMMDD_CC_TX
            parts = scene_id.split('_')
            if len(parts) >= 4:
                date_str = parts[3]
                scene_date = datetime.strptime(date_str, '%Y%m%d')
                
                if start_dt <= scene_date <= end_dt:
                    filtered_ids.append(scene_id)
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not parse date from scene ID {scene_id}: {e}")
    
    logger.info(f"Filtered {len(scene_ids)} scenes to {len(filtered_ids)} within date range")
    return filtered_ids


def get_landsat_bands_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about Landsat bands.
    
    Returns:
        Dict[str, Dict[str, Any]]: Band information dictionary
    """
    bands_info = {
        "SR_B1": {
            "name": "Coastal Aerosol",
            "wavelength": "0.43-0.45 µm",
            "resolution": 30,
            "common_name": "coastal"
        },
        "SR_B2": {
            "name": "Blue",
            "wavelength": "0.45-0.51 µm", 
            "resolution": 30,
            "common_name": "blue"
        },
        "SR_B3": {
            "name": "Green",
            "wavelength": "0.53-0.59 µm",
            "resolution": 30,
            "common_name": "green"
        },
        "SR_B4": {
            "name": "Red",
            "wavelength": "0.64-0.67 µm",
            "resolution": 30,
            "common_name": "red"
        },
        "SR_B5": {
            "name": "Near Infrared",
            "wavelength": "0.85-0.88 µm",
            "resolution": 30,
            "common_name": "nir08"
        },
        "SR_B6": {
            "name": "Shortwave Infrared 1",
            "wavelength": "1.57-1.65 µm",
            "resolution": 30,
            "common_name": "swir16"
        },
        "SR_B7": {
            "name": "Shortwave Infrared 2", 
            "wavelength": "2.11-2.29 µm",
            "resolution": 30,
            "common_name": "swir22"
        },
        "ST_B10": {
            "name": "Thermal Infrared",
            "wavelength": "10.60-11.19 µm",
            "resolution": 100,
            "common_name": "lwir11"
        }
    }
    
    return bands_info


def validate_ndvi_range(ndvi_values: np.ndarray, 
                       min_valid: float = -1.0, 
                       max_valid: float = 1.0) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate NDVI values and provide statistics.
    
    Args:
        ndvi_values (np.ndarray): NDVI values array
        min_valid (float): Minimum valid NDVI value
        max_valid (float): Maximum valid NDVI value
        
    Returns:
        Tuple[bool, Dict[str, Any]]: Validation result and statistics
    """
    # Remove NaN values for analysis
    valid_values = ndvi_values[~np.isnan(ndvi_values)]
    
    if len(valid_values) == 0:
        return False, {"error": "No valid NDVI values found"}
    
    stats = {
        "total_pixels": len(ndvi_values.flatten()),
        "valid_pixels": len(valid_values),
        "nan_pixels": np.sum(np.isnan(ndvi_values)),
        "min_value": float(np.min(valid_values)),
        "max_value": float(np.max(valid_values)),
        "mean_value": float(np.mean(valid_values)),
        "std_value": float(np.std(valid_values))
    }
    
    # Check for values outside valid range
    out_of_range = np.sum((valid_values < min_valid) | (valid_values > max_valid))
    stats["out_of_range_pixels"] = int(out_of_range)
    stats["out_of_range_percentage"] = float((out_of_range / len(valid_values)) * 100)
    
    # Validation passes if most values are in valid range
    is_valid = stats["out_of_range_percentage"] < 10.0  # Less than 10% out of range
    
    if is_valid:
        logger.info("NDVI validation passed")
    else:
        logger.warning(f"NDVI validation failed: {stats['out_of_range_percentage']:.1f}% values out of range")
    
    return is_valid, stats


def memory_usage_estimate(shape: Tuple[int, ...], dtype: str = "float32") -> Dict[str, float]:
    """
    Estimate memory usage for arrays.
    
    Args:
        shape (Tuple[int, ...]): Array shape
        dtype (str): Data type
        
    Returns:
        Dict[str, float]: Memory usage in different units
    """
    dtype_sizes = {
        "float32": 4,
        "float64": 8,
        "int16": 2,
        "int32": 4,
        "int64": 8,
        "uint8": 1,
        "uint16": 2
    }
    
    element_size = dtype_sizes.get(dtype, 4)
    total_elements = np.prod(shape)
    bytes_total = total_elements * element_size
    
    usage = {
        "bytes": bytes_total,
        "kb": bytes_total / 1024,
        "mb": bytes_total / (1024 * 1024),
        "gb": bytes_total / (1024 * 1024 * 1024)
    }
    
    logger.info(f"Estimated memory usage for shape {shape}: {usage['mb']:.2f} MB")
    return usage
