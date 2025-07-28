#!/usr/bin/env python3
"""
Simple NDVI Analysis Example
============================

This script demonstrates a simplified version of NDVI analysis 
without complex spatial analysis dependencies.
"""

import sys
import os
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_ndvi_data():
    """
    Create synthetic NDVI data for demonstration purposes.
    """
    logger.info("Creating synthetic NDVI data...")
    
    # Create spatial dimensions
    lat = np.linspace(35.5, 36.0, 50)  # 50 latitude points
    lon = np.linspace(-120.5, -120.0, 50)  # 50 longitude points
    
    # Create temporal dimension (6 months of data)
    start_date = datetime(2023, 4, 1)
    dates = [start_date + timedelta(days=i*15) for i in range(12)]  # Bi-weekly
    
    # Create synthetic NDVI data with spatial and temporal patterns
    ndvi_data = np.zeros((len(dates), len(lat), len(lon)))
    
    for t, date in enumerate(dates):
        # Seasonal pattern
        seasonal_factor = 0.3 + 0.4 * np.sin(2 * np.pi * t / 12)
        
        # Spatial gradient (higher NDVI in center)
        lat_effect = np.exp(-((lat - 35.75) / 0.2) ** 2)
        lon_effect = np.exp(-((lon + 120.25) / 0.2) ** 2)
        spatial_pattern = np.outer(lat_effect, lon_effect)
        
        # Add some random noise
        noise = np.random.normal(0, 0.05, (len(lat), len(lon)))
        
        # Combine effects
        ndvi_data[t] = seasonal_factor * spatial_pattern + noise
        
        # Ensure NDVI values are in valid range [-1, 1]
        ndvi_data[t] = np.clip(ndvi_data[t], -1, 1)
    
    return ndvi_data, dates, lat, lon


def calculate_basic_statistics(ndvi_data):
    """
    Calculate basic temporal statistics for NDVI data.
    """
    logger.info("Calculating temporal statistics...")
    
    stats = {
        'mean': np.mean(ndvi_data, axis=0),
        'std': np.std(ndvi_data, axis=0),
        'min': np.min(ndvi_data, axis=0),
        'max': np.max(ndvi_data, axis=0),
        'median': np.median(ndvi_data, axis=0)
    }
    
    return stats


def simple_hotspot_detection(ndvi_mean, threshold_std=1.5):
    """
    Simple hotspot detection using statistical thresholds.
    """
    logger.info("Performing simple hotspot detection...")
    
    # Calculate mean and standard deviation
    overall_mean = np.mean(ndvi_mean)
    overall_std = np.std(ndvi_mean)
    
    # Define hotspots and coldspots
    hotspots = ndvi_mean > (overall_mean + threshold_std * overall_std)
    coldspots = ndvi_mean < (overall_mean - threshold_std * overall_std)
    
    # Create result array
    # 1 = hotspot, -1 = coldspot, 0 = not significant
    result = np.zeros_like(ndvi_mean)
    result[hotspots] = 1
    result[coldspots] = -1
    
    return result


def create_visualizations(ndvi_data, stats, hotspots, dates, lat, lon, output_dir):
    """
    Create visualizations for the analysis results.
    """
    logger.info("Creating visualizations...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('NDVI Analysis Results', fontsize=16, fontweight='bold')
    
    # 1. NDVI Time Series (mean over space)
    mean_timeseries = np.mean(ndvi_data, axis=(1, 2))
    axes[0, 0].plot(dates, mean_timeseries, 'g-', linewidth=2, marker='o')
    axes[0, 0].set_title('NDVI Time Series (Spatial Mean)')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Mean NDVI')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Mean NDVI Map
    im1 = axes[0, 1].imshow(stats['mean'], cmap='RdYlGn', vmin=-1, vmax=1)
    axes[0, 1].set_title('Mean NDVI')
    axes[0, 1].set_xlabel('Longitude Index')
    axes[0, 1].set_ylabel('Latitude Index')
    plt.colorbar(im1, ax=axes[0, 1], label='NDVI')
    
    # 3. NDVI Standard Deviation
    im2 = axes[0, 2].imshow(stats['std'], cmap='plasma')
    axes[0, 2].set_title('NDVI Standard Deviation')
    axes[0, 2].set_xlabel('Longitude Index')
    axes[0, 2].set_ylabel('Latitude Index')
    plt.colorbar(im2, ax=axes[0, 2], label='Std Dev')
    
    # 4. Hotspot Map
    hotspot_colors = ['blue', 'white', 'red']
    im3 = axes[1, 0].imshow(hotspots, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 0].set_title('Hotspot Analysis')
    axes[1, 0].set_xlabel('Longitude Index')
    axes[1, 0].set_ylabel('Latitude Index')
    cbar3 = plt.colorbar(im3, ax=axes[1, 0])
    cbar3.set_label('Hotspot Type')
    cbar3.set_ticks([-1, 0, 1])
    cbar3.set_ticklabels(['Cold Spot', 'Not Significant', 'Hot Spot'])
    
    # 5. NDVI Histogram
    axes[1, 1].hist(stats['mean'].flatten(), bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1, 1].set_title('NDVI Distribution')
    axes[1, 1].set_xlabel('NDVI Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Monthly NDVI Comparison
    monthly_means = [np.mean(ndvi_data[i]) for i in range(len(dates))]
    bars = axes[1, 2].bar(range(len(dates)), monthly_means, color='green', alpha=0.7)
    axes[1, 2].set_title('Monthly NDVI Means')
    axes[1, 2].set_xlabel('Time Period')
    axes[1, 2].set_ylabel('Mean NDVI')
    axes[1, 2].set_xticks(range(len(dates)))
    axes[1, 2].set_xticklabels([d.strftime('%m/%d') for d in dates], rotation=45)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "ndvi_analysis_results.png", dpi=300, bbox_inches='tight')
    logger.info(f"Visualization saved to {output_dir / 'ndvi_analysis_results.png'}")
    
    return fig


def main():
    """
    Main function demonstrating simplified NDVI analysis.
    """
    try:
        # Create output directory
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        logger.info("Starting Simple NDVI Analysis Example")
        
        # 1. Create synthetic data
        ndvi_data, dates, lat, lon = create_synthetic_ndvi_data()
        logger.info(f"Created NDVI data: {ndvi_data.shape} (time, lat, lon)")
        
        # 2. Calculate statistics
        stats = calculate_basic_statistics(ndvi_data)
        
        # 3. Perform hotspot detection
        hotspots = simple_hotspot_detection(stats['mean'])
        
        # 4. Create visualizations
        fig = create_visualizations(ndvi_data, stats, hotspots, dates, lat, lon, output_dir)
        
        # 5. Print summary statistics
        print("\\n" + "="*60)
        print("SIMPLE NDVI ANALYSIS SUMMARY")
        print("="*60)
        print(f"Study Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
        print(f"Spatial Extent: {len(lat)} x {len(lon)} pixels")
        print(f"Temporal Resolution: {len(dates)} time steps")
        print(f"Mean NDVI: {np.mean(stats['mean']):.3f}")
        print(f"NDVI Range: {np.min(stats['min']):.3f} to {np.max(stats['max']):.3f}")
        
        # Hotspot statistics
        hot_spots = np.sum(hotspots == 1)
        cold_spots = np.sum(hotspots == -1)
        total_pixels = hotspots.size
        
        print(f"\\nHotspot Analysis:")
        print(f"  Hot spots: {hot_spots} pixels ({100*hot_spots/total_pixels:.1f}%)")
        print(f"  Cold spots: {cold_spots} pixels ({100*cold_spots/total_pixels:.1f}%)")
        print(f"  Not significant: {total_pixels - hot_spots - cold_spots} pixels")
        
        # Save numerical results
        logger.info("Saving numerical results...")
        np.save(output_dir / "ndvi_data.npy", ndvi_data)
        np.save(output_dir / "ndvi_mean.npy", stats['mean'])
        np.save(output_dir / "hotspots.npy", hotspots)
        
        # Save summary to text file
        with open(output_dir / "analysis_summary.txt", "w") as f:
            f.write("SIMPLE NDVI ANALYSIS SUMMARY\\n")
            f.write("=" * 40 + "\\n\\n")
            f.write(f"Study Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}\\n")
            f.write(f"Spatial Extent: {len(lat)} x {len(lon)} pixels\\n")
            f.write(f"Temporal Resolution: {len(dates)} time steps\\n")
            f.write(f"Mean NDVI: {np.mean(stats['mean']):.3f}\\n")
            f.write(f"NDVI Range: {np.min(stats['min']):.3f} to {np.max(stats['max']):.3f}\\n")
            f.write(f"\\nHotspot Analysis:\\n")
            f.write(f"  Hot spots: {hot_spots} pixels ({100*hot_spots/total_pixels:.1f}%)\\n")
            f.write(f"  Cold spots: {cold_spots} pixels ({100*cold_spots/total_pixels:.1f}%)\\n")
            f.write(f"  Not significant: {total_pixels - hot_spots - cold_spots} pixels\\n")
        
        print(f"\\nResults saved to: {output_dir.absolute()}")
        print("\\nGenerated Files:")
        for file in output_dir.glob("*"):
            print(f"  - {file.name}")
        print("="*60)
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
