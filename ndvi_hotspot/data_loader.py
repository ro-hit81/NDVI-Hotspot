"""
Landsat Data Loader Module
==========================

This module provides functionality to load and preprocess Landsat satellite data
from Microsoft Planetary Computer STAC catalog.

Features:
- Search and filter Landsat collections
- Load data with specified spatial and temporal parameters
- Handle cloud masking and data quality filtering
- Support for multiple Landsat missions (Landsat 8, 9)
"""

import numpy as np
import xarray as xr
import stackstac
import planetary_computer
import pystac_client
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, date
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LandsatDataLoader:
    """
    A class for loading and preprocessing Landsat satellite data from Microsoft Planetary Computer.
    
    This class provides methods to search, filter, and load Landsat imagery with
    specified parameters for spatial extent, temporal range, and data quality.
    """
    
    def __init__(self, max_cloud_cover: float = 30.0):
        """
        Initialize the Landsat Data Loader.
        
        Args:
            max_cloud_cover (float): Maximum allowable cloud cover percentage (0-100)
        """
        self.max_cloud_cover = max_cloud_cover
        self.catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        logger.info("Initialized Landsat Data Loader with Microsoft Planetary Computer")
    
    def search_landsat_data(self, 
                           bbox: List[float], 
                           start_date: str, 
                           end_date: str,
                           collection: str = "landsat-c2-l2",
                           cloud_cover: Optional[float] = None) -> pystac_client.ItemCollection:
        """
        Search for Landsat data within specified parameters.
        
        Args:
            bbox (List[float]): Bounding box [min_lon, min_lat, max_lon, max_lat]
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            collection (str): Landsat collection name
            cloud_cover (Optional[float]): Maximum cloud cover override
            
        Returns:
            pystac_client.ItemCollection: Collection of matching Landsat items
        """
        cloud_threshold = cloud_cover if cloud_cover is not None else self.max_cloud_cover
        
        search_params = {
            "collections": [collection],
            "bbox": bbox,
            "datetime": f"{start_date}/{end_date}",
        }
        
        if cloud_threshold < 100:
            search_params["query"] = {"eo:cloud_cover": {"lte": cloud_threshold}}
        
        search = self.catalog.search(**search_params)
        items = search.item_collection()
        
        logger.info(f"Found {len(items)} Landsat scenes for the specified criteria")
        return items
    
    def search_by_ids(self, 
                     bbox: List[float], 
                     scene_ids: List[str],
                     collection: str = "landsat-c2-l2") -> pystac_client.ItemCollection:
        """
        Search for specific Landsat scenes by their IDs.
        
        Args:
            bbox (List[float]): Bounding box [min_lon, min_lat, max_lon, max_lat]
            scene_ids (List[str]): List of Landsat scene IDs
            collection (str): Landsat collection name
            
        Returns:
            pystac_client.ItemCollection: Collection of matching Landsat items
        """
        search = self.catalog.search(
            collections=[collection], 
            bbox=bbox, 
            ids=scene_ids
        )
        items = search.item_collection()
        
        logger.info(f"Found {len(items)} scenes from {len(scene_ids)} requested IDs")
        return items
    
    def load_landsat_stack(self, 
                          items: pystac_client.ItemCollection,
                          bbox: List[float],
                          resolution: float = 30.0,
                          epsg: int = 4326,
                          assets: Optional[List[str]] = None,
                          chunksize: int = 256) -> xr.DataArray:
        """
        Load Landsat data as a stacked xarray DataArray.
        
        Args:
            items (pystac_client.ItemCollection): Collection of Landsat items
            bbox (List[float]): Bounding box [min_lon, min_lat, max_lon, max_lat]
            resolution (float): Spatial resolution in degrees or meters
            epsg (int): EPSG code for coordinate reference system
            assets (Optional[List[str]]): List of band assets to load
            chunksize (int): Chunk size for dask arrays
            
        Returns:
            xr.DataArray: Stacked Landsat data with time, band, y, x dimensions
        """
        if assets is None:
            # Default assets for NDVI calculation and analysis
            assets = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "ST_B10"]
        
        # Convert resolution to degrees if using geographic CRS
        if epsg == 4326 and resolution > 1:
            resolution = resolution / 111000  # Convert meters to degrees (approximate)
        
        data = (
            stackstac.stack(
                items,
                epsg=epsg,
                bounds_latlon=bbox,
                resolution=resolution,
                assets=assets,
                chunksize=chunksize,
            )
            .where(lambda x: x > 0, other=np.nan)  # Remove invalid/nodata values
            .assign_coords(
                band=lambda x: x.common_name.rename("band"),  # Use common band names
                time=lambda x: x.time.dt.round("D"),  # Round time to daily
            )
        )
        
        logger.info(f"Loaded Landsat stack with shape: {data.shape}")
        logger.info(f"Bands available: {list(data.band.values)}")
        logger.info(f"Time range: {data.time.values[0]} to {data.time.values[-1]}")
        
        return data
    
    def apply_cloud_mask(self, 
                        data: xr.DataArray, 
                        items: pystac_client.ItemCollection,
                        bbox: List[float]) -> xr.DataArray:
        """
        Apply cloud mask to Landsat data using QA_PIXEL band.
        
        Args:
            data (xr.DataArray): Landsat data array
            items (pystac_client.ItemCollection): Original items for QA band
            bbox (List[float]): Bounding box for loading QA data
            
        Returns:
            xr.DataArray: Cloud-masked Landsat data
        """
        try:
            # Load QA_PIXEL band for cloud masking
            qa_data = stackstac.stack(
                items,
                epsg=data.rio.crs.to_epsg(),
                bounds_latlon=bbox,
                resolution=data.rio.resolution()[0],
                assets=["QA_PIXEL"],
                chunksize=256,
            )
            
            # Create cloud mask (bit 3 = cloud, bit 4 = cloud shadow)
            cloud_mask = (qa_data & (1 << 3)) | (qa_data & (1 << 4))
            cloud_free_mask = cloud_mask == 0
            
            # Apply mask to data
            masked_data = data.where(cloud_free_mask)
            
            logger.info("Applied cloud mask to Landsat data")
            return masked_data
            
        except Exception as e:
            logger.warning(f"Could not apply cloud mask: {e}")
            return data
    
    def get_sample_bbox_austria(self) -> List[float]:
        """
        Get a sample bounding box for Austria (used in the original notebook).
        
        Returns:
            List[float]: Bounding box coordinates [min_lon, min_lat, max_lon, max_lat]
        """
        return [13.324065832846372, 47.790832246642026, 13.457618384115904, 47.867354172891204]
    
    def get_sample_scene_ids(self) -> List[str]:
        """
        Get sample Landsat scene IDs used in the original analysis.
        
        Returns:
            List[str]: List of Landsat scene IDs
        """
        return [
            'LC08_L2SP_191027_20170706_02_T1',
            'LC08_L2SP_191027_20180927_02_T1',
            'LC08_L2SP_192027_20190921_02_T1',
            'LC08_L2SP_191027_20200612_02_T1',
            'LC08_L2SP_191027_20210615_02_T1'
        ]
