"""
Hotspot Analyzer Module
======================

This module provides functionality for detecting and analyzing hotspots in NDVI data
using spatial statistics and focal operations.

Features:
- Hotspot detection using Getis-Ord Gi* statistic
- Focal statistics calculation
- Spatial convolution operations
- Kernel generation for spatial analysis
"""

import numpy as np
import xarray as xr
# from xrspatial.convolution import calc_cellsize, circle_kernel, convolution_2d
# from xrspatial.focal import focal_stats, hotspots
from scipy import ndimage
from skimage import filters, morphology
from typing import List, Tuple, Optional, Dict, Any, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HotspotAnalyzer:
    """
    A class for detecting and analyzing hotspots in NDVI data using spatial statistics.
    
    This class provides methods to calculate hotspots, focal statistics, and 
    perform spatial convolution operations on NDVI data.
    """
    
    def __init__(self, kernel_radius: float = 1.5):
        """
        Initialize the Hotspot Analyzer.
        
        Args:
            kernel_radius (float): Radius multiplier for creating circular kernels
        """
        self.kernel_radius = kernel_radius
        logger.info(f"Initialized Hotspot Analyzer with kernel radius: {kernel_radius}")
    
    def calculate_hotspots(self, ndvi: xr.DataArray) -> xr.DataArray:
        """
        Calculate hotspots for NDVI data using Getis-Ord Gi* statistic.
        
        Args:
            ndvi (xr.DataArray): Multi-temporal NDVI data
            
        Returns:
            xr.DataArray: Hotspot analysis results
        """
        try:
            # Calculate cell size for kernel creation
            cellsize = calc_cellsize(ndvi)
            logger.info(f"Calculated cell size: {cellsize}")
            
            # Create circular kernel
            kernel = circle_kernel(*cellsize, radius=self.kernel_radius * cellsize[0])
            logger.info(f"Created circular kernel with shape: {kernel.shape}")
            
            # Calculate hotspots for each time step
            hotspot_list = []
            for ndvi_slice in ndvi:
                hotspot_slice = hotspots(ndvi_slice, kernel)
                hotspot_list.append(hotspot_slice)
            
            # Concatenate along time dimension
            hotspots_result = xr.concat(hotspot_list, dim="time")
            
            logger.info(f"Calculated hotspots for {len(hotspot_list)} time steps")
            logger.info(f"Unique hotspot values: {np.unique(hotspots_result.values)}")
            
            return hotspots_result
            
        except Exception as e:
            logger.error(f"Error calculating hotspots: {e}")
            raise
    
    def calculate_focal_statistics(self, ndvi: xr.DataArray) -> xr.DataArray:
        """
        Calculate focal statistics for NDVI data.
        
        Args:
            ndvi (xr.DataArray): Multi-temporal NDVI data
            
        Returns:
            xr.DataArray: Focal statistics with stats dimension
        """
        try:
            # Calculate cell size and create kernel
            cellsize = calc_cellsize(ndvi)
            kernel = circle_kernel(*cellsize, radius=self.kernel_radius * cellsize[0])
            
            # Calculate focal statistics for each time step
            stats_list = []
            for ndvi_slice in ndvi:
                stats_slice = focal_stats(ndvi_slice, kernel)
                stats_list.append(stats_slice)
            
            # Concatenate along time dimension
            focal_stats_result = xr.concat(stats_list, dim="time")
            
            logger.info(f"Calculated focal statistics for {len(stats_list)} time steps")
            logger.info(f"Statistics calculated: {list(focal_stats_result.coords['stats'].values)}")
            
            return focal_stats_result
            
        except Exception as e:
            logger.error(f"Error calculating focal statistics: {e}")
            raise
    
    def apply_convolution(self, 
                         ndvi: xr.DataArray, 
                         kernel: np.ndarray) -> xr.DataArray:
        """
        Apply convolution operation to NDVI data.
        
        Args:
            ndvi (xr.DataArray): Multi-temporal NDVI data
            kernel (np.ndarray): Convolution kernel
            
        Returns:
            xr.DataArray: Convolved NDVI data
        """
        try:
            # Apply convolution for each time step
            conv_list = []
            for ndvi_slice in ndvi:
                conv_slice = convolution_2d(ndvi_slice, kernel)
                conv_list.append(conv_slice)
            
            # Concatenate along time dimension
            convolved_result = xr.concat(conv_list, dim="time")
            
            logger.info(f"Applied convolution with kernel shape: {kernel.shape}")
            return convolved_result
            
        except Exception as e:
            logger.error(f"Error applying convolution: {e}")
            raise
    
    def sobel_edge_detection(self, ndvi: xr.DataArray) -> xr.DataArray:
        """
        Apply Sobel edge detection to NDVI data.
        
        Args:
            ndvi (xr.DataArray): Multi-temporal NDVI data
            
        Returns:
            xr.DataArray: Edge-detected NDVI data
        """
        # Sobel operator kernel for edge detection
        sobel_kernel = np.array([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ])
        
        logger.info("Applying Sobel edge detection")
        return self.apply_convolution(ndvi, sobel_kernel)
    
    def create_custom_kernel(self, 
                           kernel_type: str = "circle", 
                           size: int = 3, 
                           cellsize: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Create custom kernels for spatial analysis.
        
        Args:
            kernel_type (str): Type of kernel ('circle', 'square', 'cross')
            size (int): Size of the kernel
            cellsize (Optional[Tuple[float, float]]): Cell size for circular kernels
            
        Returns:
            np.ndarray: Generated kernel
        """
        if kernel_type == "circle" and cellsize is not None:
            kernel = circle_kernel(*cellsize, radius=self.kernel_radius * cellsize[0])
        elif kernel_type == "square":
            kernel = np.ones((size, size))
        elif kernel_type == "cross":
            kernel = np.zeros((size, size))
            center = size // 2
            kernel[center, :] = 1  # Horizontal line
            kernel[:, center] = 1  # Vertical line
        else:
            # Default to square kernel
            kernel = np.ones((size, size))
        
        logger.info(f"Created {kernel_type} kernel with shape: {kernel.shape}")
        return kernel
    
    def classify_hotspots(self, hotspots_data: xr.DataArray) -> Dict[str, xr.DataArray]:
        """
        Classify hotspot results into meaningful categories.
        
        Args:
            hotspots_data (xr.DataArray): Hotspot analysis results
            
        Returns:
            Dict[str, xr.DataArray]: Classified hotspot categories
        """
        classifications = {}
        
        # Define hotspot thresholds based on typical Getis-Ord values
        # Values are typically: -99, -95, -90, 0, 90, 95, 99
        classifications['cold_spots_99'] = (hotspots_data == -99)
        classifications['cold_spots_95'] = (hotspots_data == -95)
        classifications['cold_spots_90'] = (hotspots_data == -90)
        classifications['not_significant'] = (hotspots_data == 0)
        classifications['hot_spots_90'] = (hotspots_data == 90)
        classifications['hot_spots_95'] = (hotspots_data == 95)
        classifications['hot_spots_99'] = (hotspots_data == 99)
        
        # Create combined categories
        classifications['all_cold_spots'] = (hotspots_data < 0)
        classifications['all_hot_spots'] = (hotspots_data > 0)
        
        logger.info("Classified hotspots into significance categories")
        return classifications
    
    def calculate_hotspot_persistence(self, hotspots_data: xr.DataArray) -> xr.DataArray:
        """
        Calculate hotspot persistence across time.
        
        Args:
            hotspots_data (xr.DataArray): Multi-temporal hotspot data
            
        Returns:
            xr.DataArray: Hotspot persistence (percentage of time as hotspot)
        """
        # Count how many times each pixel is a significant hotspot
        hot_count = (hotspots_data > 0).sum(dim='time')
        cold_count = (hotspots_data < 0).sum(dim='time')
        total_time = len(hotspots_data.time)
        
        # Calculate persistence as percentage
        hot_persistence = (hot_count / total_time) * 100
        cold_persistence = (cold_count / total_time) * 100
        
        # Create combined persistence metric
        # Positive values for hot persistence, negative for cold persistence
        persistence = hot_persistence - cold_persistence
        
        logger.info("Calculated hotspot persistence across time")
        return persistence
    
    def identify_persistent_hotspots(self, 
                                   hotspots_data: xr.DataArray, 
                                   threshold: float = 50.0) -> xr.DataArray:
        """
        Identify pixels that are persistent hotspots.
        
        Args:
            hotspots_data (xr.DataArray): Multi-temporal hotspot data
            threshold (float): Minimum percentage of time to be considered persistent
            
        Returns:
            xr.DataArray: Binary mask of persistent hotspots
        """
        persistence = self.calculate_hotspot_persistence(hotspots_data)
        persistent_hot = persistence > threshold
        persistent_cold = persistence < -threshold
        
        # Combine into single classification
        # 1: persistent hotspot, -1: persistent cold spot, 0: variable
        persistent_classification = xr.where(
            persistent_hot, 1,
            xr.where(persistent_cold, -1, 0)
        )
        
        logger.info(f"Identified persistent hotspots with {threshold}% threshold")
        return persistent_classification
    
    def get_hotspot_statistics(self, hotspots_data: xr.DataArray) -> Dict[str, Any]:
        """
        Calculate summary statistics for hotspot analysis.
        
        Args:
            hotspots_data (xr.DataArray): Hotspot analysis results
            
        Returns:
            Dict[str, Any]: Summary statistics
        """
        stats = {}
        
        # Count pixels in each category
        unique_values, counts = np.unique(hotspots_data.values[~np.isnan(hotspots_data.values)], 
                                        return_counts=True)
        
        total_pixels = np.sum(counts)
        
        for value, count in zip(unique_values, counts):
            percentage = (count / total_pixels) * 100
            if value < 0:
                category = f"cold_spot_{abs(int(value))}"
            elif value > 0:
                category = f"hot_spot_{int(value)}"
            else:
                category = "not_significant"
            
            stats[category] = {
                'count': int(count),
                'percentage': float(percentage)
            }
        
        # Overall statistics
        hot_pixels = np.sum([count for value, count in zip(unique_values, counts) if value > 0])
        cold_pixels = np.sum([count for value, count in zip(unique_values, counts) if value < 0])
        
        stats['summary'] = {
            'total_pixels': int(total_pixels),
            'hot_spots_percentage': float((hot_pixels / total_pixels) * 100),
            'cold_spots_percentage': float((cold_pixels / total_pixels) * 100),
            'not_significant_percentage': float(((total_pixels - hot_pixels - cold_pixels) / total_pixels) * 100)
        }
        
        logger.info("Calculated hotspot summary statistics")
        return stats
