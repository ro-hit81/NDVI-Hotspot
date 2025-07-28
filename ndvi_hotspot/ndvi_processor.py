"""
NDVI Processor Module
====================

This module provides functionality for calculating and processing NDVI 
(Normalized Difference Vegetation Index) from Landsat satellite data.

Features:
- NDVI calculation from NIR and Red bands
- Temporal NDVI analysis and trend detection
- Statistical analysis of vegetation changes
- Smoothing and filtering operations
"""

import numpy as np
import xarray as xr
import xrspatial.multispectral as ms
from xrspatial.focal import mean
from typing import List, Tuple, Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NDVIProcessor:
    """
    A class for processing NDVI calculations and analysis from Landsat data.
    
    This class provides methods to calculate NDVI, perform temporal analysis,
    and apply various processing techniques for vegetation monitoring.
    """
    
    def __init__(self):
        """Initialize the NDVI Processor."""
        logger.info("Initialized NDVI Processor")
    
    def calculate_ndvi(self, data: xr.DataArray) -> xr.DataArray:
        """
        Calculate NDVI from Landsat data for all time steps.
        
        Args:
            data (xr.DataArray): Landsat data with NIR and Red bands
            
        Returns:
            xr.DataArray: NDVI values with time dimension
        """
        try:
            # Calculate NDVI for each time step
            ndvi_list = []
            for time_slice in data:
                nir = time_slice.sel(band="nir08")
                red = time_slice.sel(band="red")
                ndvi_slice = ms.ndvi(nir, red)
                ndvi_list.append(ndvi_slice)
            
            # Concatenate along time dimension
            ndvi = xr.concat(ndvi_list, dim="time")
            
            logger.info(f"Calculated NDVI for {len(ndvi_list)} time steps")
            logger.info(f"NDVI range: {float(ndvi.min().values):.3f} to {float(ndvi.max().values):.3f}")
            
            return ndvi
            
        except Exception as e:
            logger.error(f"Error calculating NDVI: {e}")
            raise
    
    def calculate_ndvi_single(self, nir: xr.DataArray, red: xr.DataArray) -> xr.DataArray:
        """
        Calculate NDVI for a single time step.
        
        Args:
            nir (xr.DataArray): Near-infrared band data
            red (xr.DataArray): Red band data
            
        Returns:
            xr.DataArray: NDVI values for single time step
        """
        ndvi = ms.ndvi(nir, red)
        return ndvi
    
    def apply_smoothing(self, ndvi: xr.DataArray) -> xr.DataArray:
        """
        Apply spatial smoothing to NDVI data using focal mean.
        
        Args:
            ndvi (xr.DataArray): Input NDVI data
            
        Returns:
            xr.DataArray: Smoothed NDVI data
        """
        try:
            # Apply mean filter to each time step
            smooth_list = []
            for ndvi_slice in ndvi:
                smooth_slice = mean(ndvi_slice)
                smooth_list.append(smooth_slice)
            
            smooth_ndvi = xr.concat(smooth_list, dim="time")
            
            logger.info("Applied spatial smoothing to NDVI data")
            return smooth_ndvi
            
        except Exception as e:
            logger.error(f"Error applying smoothing: {e}")
            raise
    
    def calculate_temporal_statistics(self, ndvi: xr.DataArray) -> Dict[str, xr.DataArray]:
        """
        Calculate temporal statistics for NDVI time series.
        
        Args:
            ndvi (xr.DataArray): Multi-temporal NDVI data
            
        Returns:
            Dict[str, xr.DataArray]: Dictionary containing temporal statistics
        """
        stats = {}
        
        # Calculate various temporal statistics
        stats['mean'] = ndvi.mean(dim='time')
        stats['std'] = ndvi.std(dim='time')
        stats['min'] = ndvi.min(dim='time')
        stats['max'] = ndvi.max(dim='time')
        stats['range'] = stats['max'] - stats['min']
        
        # Calculate coefficient of variation
        stats['cv'] = stats['std'] / stats['mean']
        
        # Calculate trend (simple linear trend)
        try:
            # Create time index for trend calculation
            time_index = np.arange(len(ndvi.time))
            trend = xr.apply_ufunc(
                lambda x: np.polyfit(time_index, x, 1)[0] if not np.all(np.isnan(x)) else np.nan,
                ndvi,
                input_core_dims=[['time']],
                dask='parallelized',
                output_dtypes=[float]
            )
            stats['trend'] = trend
        except Exception as e:
            logger.warning(f"Could not calculate trend: {e}")
            stats['trend'] = ndvi.mean(dim='time') * 0  # Zero array with same spatial dims
        
        logger.info("Calculated temporal statistics for NDVI data")
        return stats
    
    def detect_vegetation_change(self, 
                               ndvi: xr.DataArray, 
                               threshold: float = 0.1) -> xr.DataArray:
        """
        Detect significant vegetation changes between first and last time step.
        
        Args:
            ndvi (xr.DataArray): Multi-temporal NDVI data
            threshold (float): Minimum change threshold to consider significant
            
        Returns:
            xr.DataArray: Vegetation change map (-1: decrease, 0: no change, 1: increase)
        """
        first_ndvi = ndvi.isel(time=0)
        last_ndvi = ndvi.isel(time=-1)
        
        change = last_ndvi - first_ndvi
        
        # Classify changes
        change_class = xr.where(change > threshold, 1,  # Increase
                               xr.where(change < -threshold, -1, 0))  # Decrease or no change
        
        logger.info(f"Detected vegetation changes with threshold {threshold}")
        return change_class
    
    def calculate_vegetation_phenology(self, ndvi: xr.DataArray) -> Dict[str, xr.DataArray]:
        """
        Calculate basic vegetation phenology metrics.
        
        Args:
            ndvi (xr.DataArray): Multi-temporal NDVI data
            
        Returns:
            Dict[str, xr.DataArray]: Phenology metrics
        """
        phenology = {}
        
        # Peak growing season (maximum NDVI)
        phenology['peak_ndvi'] = ndvi.max(dim='time')
        phenology['peak_time'] = ndvi.argmax(dim='time')
        
        # Growing season length (approximate)
        # Count number of time steps above mean NDVI
        mean_ndvi = ndvi.mean(dim='time')
        above_mean = (ndvi > mean_ndvi).sum(dim='time')
        phenology['growing_season_length'] = above_mean
        
        # Start and end of growing season (first and last above-average NDVI)
        try:
            # This is a simplified approach - more sophisticated methods exist
            above_mean_mask = ndvi > mean_ndvi
            phenology['season_start'] = above_mean_mask.argmax(dim='time')
            
            # For season end, reverse the time dimension and find first True
            reversed_mask = above_mean_mask.isel(time=slice(None, None, -1))
            season_end_reversed = reversed_mask.argmax(dim='time')
            phenology['season_end'] = len(ndvi.time) - 1 - season_end_reversed
            
        except Exception as e:
            logger.warning(f"Could not calculate season start/end: {e}")
        
        logger.info("Calculated vegetation phenology metrics")
        return phenology
    
    def filter_valid_ndvi(self, 
                         ndvi: xr.DataArray, 
                         min_value: float = -1.0, 
                         max_value: float = 1.0) -> xr.DataArray:
        """
        Filter NDVI values to valid range and remove outliers.
        
        Args:
            ndvi (xr.DataArray): Input NDVI data
            min_value (float): Minimum valid NDVI value
            max_value (float): Maximum valid NDVI value
            
        Returns:
            xr.DataArray: Filtered NDVI data
        """
        # Remove values outside valid NDVI range
        filtered_ndvi = ndvi.where(
            (ndvi >= min_value) & (ndvi <= max_value)
        )
        
        logger.info(f"Filtered NDVI to range [{min_value}, {max_value}]")
        return filtered_ndvi
    
    def calculate_ndvi_anomaly(self, 
                             ndvi: xr.DataArray, 
                             reference_period: Optional[slice] = None) -> xr.DataArray:
        """
        Calculate NDVI anomalies relative to a reference period.
        
        Args:
            ndvi (xr.DataArray): Multi-temporal NDVI data
            reference_period (Optional[slice]): Time slice for reference period
            
        Returns:
            xr.DataArray: NDVI anomalies
        """
        if reference_period is None:
            # Use all data as reference
            reference_mean = ndvi.mean(dim='time')
        else:
            # Use specified period as reference
            reference_mean = ndvi.isel(time=reference_period).mean(dim='time')
        
        # Calculate anomalies
        anomalies = ndvi - reference_mean
        
        logger.info("Calculated NDVI anomalies")
        return anomalies
