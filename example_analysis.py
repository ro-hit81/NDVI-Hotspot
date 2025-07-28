#!/usr/bin/env python3
"""
NDVI Hotspot Analysis Example
=============================

This script demonstrates how to use the NDVI Hotspot Analysis package
to analyze vegetation patterns using Landsat satellite data.

Example workflow:
1. Load Landsat data from Microsoft Planetary Computer
2. Calculate NDVI from satellite bands
3. Perform hotspot analysis
4. Create visualizations
5. Export results
"""

import sys
import os
import logging
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from ndvi_hotspot import (
    LandsatDataLoader, 
    NDVIProcessor, 
    HotspotAnalyzer, 
    NDVIVisualizer,
    get_sample_bbox
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main function demonstrating NDVI hotspot analysis workflow.
    """
    try:
        # Create output directory
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        logger.info("Starting NDVI Hotspot Analysis Example")
        
        # 1. Initialize components
        logger.info("Initializing analysis components...")
        data_loader = LandsatDataLoader(max_cloud_cover=30.0)
        ndvi_processor = NDVIProcessor()
        hotspot_analyzer = HotspotAnalyzer(kernel_radius=1.5)
        visualizer = NDVIVisualizer(figsize=(15, 10))
        
        # 2. Define study area and parameters
        logger.info("Setting up study parameters...")
        bbox = get_sample_bbox("austria")  # Sample bounding box for Austria
        scene_ids = data_loader.get_sample_scene_ids()  # Pre-defined scene IDs
        
        logger.info(f"Study area: {bbox}")
        logger.info(f"Number of scenes: {len(scene_ids)}")
        
        # 3. Load Landsat data
        logger.info("Loading Landsat data...")
        items = data_loader.search_by_ids(bbox, scene_ids)
        
        if len(items) == 0:
            logger.error("No Landsat scenes found for the specified criteria")
            return
        
        # Load and stack the data
        data = data_loader.load_landsat_stack(
            items=items,
            bbox=bbox,
            resolution=0.000089494585235856472,  # Approximate 30m at this latitude
            assets=["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "ST_B10"]
        )
        
        # Persist data in memory for faster processing
        data = data.persist()
        logger.info("Data loaded and persisted in memory")
        
        # 4. Calculate NDVI
        logger.info("Calculating NDVI...")
        ndvi = ndvi_processor.calculate_ndvi(data)
        
        # Apply smoothing
        smooth_ndvi = ndvi_processor.apply_smoothing(ndvi)
        
        # Calculate temporal statistics
        temporal_stats = ndvi_processor.calculate_temporal_statistics(ndvi)
        
        # 5. Perform hotspot analysis
        logger.info("Performing hotspot analysis...")
        hotspots_result = hotspot_analyzer.calculate_hotspots(ndvi)
        
        # Calculate focal statistics
        focal_stats = hotspot_analyzer.calculate_focal_statistics(ndvi)
        
        # Get hotspot statistics
        hotspot_stats = hotspot_analyzer.get_hotspot_statistics(hotspots_result)
        
        # 6. Create visualizations
        logger.info("Creating visualizations...")
        
        # NDVI time series
        fig_ndvi = visualizer.plot_ndvi_timeseries(
            ndvi, 
            title="NDVI Time Series Analysis",
            save_path=output_dir / "ndvi_timeseries.png"
        )
        
        # Hotspot analysis
        fig_hotspots = visualizer.plot_hotspots(
            hotspots_result,
            title="NDVI Hotspot Analysis",
            save_path=output_dir / "hotspots_analysis.png"
        )
        
        # True color composite
        fig_true_color = visualizer.plot_true_color(
            data,
            title="True Color Composite",
            save_path=output_dir / "true_color.png"
        )
        
        # False color composite
        fig_false_color = visualizer.plot_false_color(
            data,
            title="False Color Composite (NIR-R-G)",
            save_path=output_dir / "false_color.png"
        )
        
        # Temporal statistics
        fig_stats = visualizer.plot_temporal_statistics(
            temporal_stats,
            save_path=output_dir / "temporal_statistics.png"
        )
        
        # NDVI histogram
        fig_hist = visualizer.plot_ndvi_histogram(
            ndvi,
            title="NDVI Distribution Across All Time Steps",
            save_path=output_dir / "ndvi_histogram.png"
        )
        
        # Hotspot summary
        fig_summary = visualizer.plot_hotspot_summary(
            hotspot_stats,
            save_path=output_dir / "hotspot_summary.png"
        )
        
        # 7. Save numerical results
        logger.info("Saving numerical results...")
        
        # Save NDVI data
        ndvi.to_netcdf(output_dir / "ndvi_data.nc")
        
        # Save hotspot results
        hotspots_result.to_netcdf(output_dir / "hotspots_data.nc")
        
        # Save temporal statistics
        for stat_name, stat_data in temporal_stats.items():
            stat_data.to_netcdf(output_dir / f"temporal_{stat_name}.nc")
        
        # Save hotspot statistics to text file
        with open(output_dir / "hotspot_statistics.txt", "w") as f:
            f.write("NDVI Hotspot Analysis Results\\n")
            f.write("=" * 40 + "\\n\\n")
            
            for category, stats in hotspot_stats.items():
                if category == "summary":
                    f.write("SUMMARY STATISTICS:\\n")
                    f.write("-" * 20 + "\\n")
                    for key, value in stats.items():
                        f.write(f"{key}: {value:.2f}\\n")
                    f.write("\\n")
                elif isinstance(stats, dict):
                    f.write(f"{category.replace('_', ' ').title()}:\\n")
                    f.write(f"  Count: {stats.get('count', 0)}\\n")
                    f.write(f"  Percentage: {stats.get('percentage', 0):.2f}%\\n")
        
        logger.info("Analysis completed successfully!")
        logger.info(f"Results saved to: {output_dir.absolute()}")
        
        # Print summary
        print("\\n" + "="*50)
        print("NDVI HOTSPOT ANALYSIS COMPLETED")
        print("="*50)
        print(f"Study Area: {bbox}")
        print(f"Time Steps: {len(ndvi.time)}")
        print(f"Spatial Resolution: ~30m")
        print(f"Output Directory: {output_dir.absolute()}")
        print("\\nGenerated Files:")
        for file in output_dir.glob("*"):
            print(f"  - {file.name}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Analysis failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
