"""
Visualization Module
===================

This module provides comprehensive visualization capabilities for NDVI data,
hotspot analysis results, and temporal analysis.

Features:
- NDVI time series visualization
- Hotspot mapping with custom colormaps
- True and false color composite generation
- Statistical plots and temporal analysis
- Interactive and static plotting options
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import xrspatial.multispectral as ms
from typing import List, Tuple, Optional, Dict, Any, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NDVIVisualizer:
    """
    A class for visualizing NDVI data, hotspot analysis, and temporal patterns.
    
    This class provides comprehensive visualization methods for satellite data
    analysis including NDVI plots, color composites, and hotspot maps.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10), dpi: int = 100):
        """
        Initialize the NDVI Visualizer.
        
        Args:
            figsize (Tuple[int, int]): Default figure size for plots
            dpi (int): DPI for figure quality
        """
        self.figsize = figsize
        self.dpi = dpi
        
        # Set matplotlib style parameters
        self.rc_params = {
            "axes.spines.left": False,
            "axes.spines.right": False,
            "axes.spines.bottom": False,
            "axes.spines.top": False,
            "xtick.bottom": False,
            "xtick.labelbottom": False,
            "ytick.labelleft": False,
            "ytick.left": False,
        }
        
        logger.info("Initialized NDVI Visualizer")
    
    def plot_ndvi_timeseries(self, 
                           ndvi: xr.DataArray, 
                           title: str = "NDVI Time Series",
                           cmap: str = "viridis",
                           col_wrap: int = 5,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot NDVI time series as a grid of subplots.
        
        Args:
            ndvi (xr.DataArray): Multi-temporal NDVI data
            title (str): Title for the plot
            cmap (str): Colormap name
            col_wrap (int): Number of columns before wrapping
            save_path (Optional[str]): Path to save the figure
            
        Returns:
            plt.Figure: The created figure
        """
        # Apply clean style
        plt.rcParams.update(self.rc_params)
        
        # Create the plot
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        plot = ndvi.plot.imshow(
            x="x", 
            y="y", 
            col="time", 
            col_wrap=col_wrap, 
            cmap=cmap,
            cbar_kwargs={"label": "NDVI", "shrink": 0.8}
        )
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved NDVI timeseries plot to {save_path}")
        
        logger.info(f"Created NDVI timeseries plot with {len(ndvi.time)} time steps")
        return fig
    
    def plot_hotspots(self, 
                     hotspots_data: xr.DataArray,
                     title: str = "NDVI Hotspot Analysis",
                     col_wrap: int = 5,
                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot hotspot analysis results with custom colormap.
        
        Args:
            hotspots_data (xr.DataArray): Hotspot analysis results
            title (str): Title for the plot
            col_wrap (int): Number of columns before wrapping
            save_path (Optional[str]): Path to save the figure
            
        Returns:
            plt.Figure: The created figure
        """
        # Create custom hotspot colormap
        hotspot_colors = {
            -99: "#0000FF",  # Deep blue - very cold
            -95: "#4169E1",  # Blue - cold
            -90: "#87CEEB",  # Light blue - somewhat cold
            0: "#FFFFFF",    # White - not significant
            90: "#FFA500",   # Orange - somewhat hot
            95: "#FF4500",   # Red orange - hot
            99: "#FF0000"    # Red - very hot
        }
        
        # Get unique values in data to create appropriate colormap
        unique_vals = np.unique(hotspots_data.values[~np.isnan(hotspots_data.values)])
        colors = [hotspot_colors.get(val, "#FFFFFF") for val in sorted(unique_vals)]
        hotspot_cmap = ListedColormap(colors)
        
        # Apply clean style
        plt.rcParams.update(self.rc_params)
        
        # Create the plot
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        plot = hotspots_data.plot.imshow(
            x="x", 
            y="y", 
            col="time", 
            col_wrap=col_wrap, 
            cmap=hotspot_cmap,
            cbar_kwargs={"label": "Hotspot Significance", "shrink": 0.8}
        )
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved hotspot plot to {save_path}")
        
        logger.info(f"Created hotspot plot with {len(hotspots_data.time)} time steps")
        return fig
    
    def plot_true_color(self, 
                       data: xr.DataArray,
                       title: str = "True Color Composite",
                       col_wrap: int = 5,
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot true color composite (RGB) for all time steps.
        
        Args:
            data (xr.DataArray): Landsat data with RGB bands
            title (str): Title for the plot
            col_wrap (int): Number of columns before wrapping
            save_path (Optional[str]): Path to save the figure
            
        Returns:
            plt.Figure: The created figure
        """
        try:
            # Create true color composites for each time step
            true_color_list = []
            for time_slice in data:
                red = time_slice.sel(band="red")
                green = time_slice.sel(band="green")
                blue = time_slice.sel(band="blue")
                true_color = ms.true_color(red, green, blue)
                true_color_list.append(true_color)
            
            true_color_stack = xr.concat(true_color_list, dim=data.coords["time"])
            
            # Apply clean style
            plt.rcParams.update(self.rc_params)
            
            # Create the plot
            fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
            
            plot = true_color_stack.plot.imshow(
                x="x", 
                y="y", 
                col="time", 
                col_wrap=col_wrap
            )
            
            fig.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Saved true color plot to {save_path}")
            
            logger.info(f"Created true color plot with {len(data.time)} time steps")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating true color plot: {e}")
            raise
    
    def plot_false_color(self, 
                        data: xr.DataArray,
                        title: str = "False Color Composite (NIR-R-G)",
                        col_wrap: int = 5,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot false color composite (NIR-Red-Green) for all time steps.
        
        Args:
            data (xr.DataArray): Landsat data with NIR, Red, Green bands
            title (str): Title for the plot
            col_wrap (int): Number of columns before wrapping
            save_path (Optional[str]): Path to save the figure
            
        Returns:
            plt.Figure: The created figure
        """
        try:
            # Create false color composites for each time step
            false_color_list = []
            for time_slice in data:
                nir = time_slice.sel(band="nir08")
                red = time_slice.sel(band="red")
                green = time_slice.sel(band="green")
                false_color = ms.true_color(nir, red, green)  # NIR as red channel
                false_color_list.append(false_color)
            
            false_color_stack = xr.concat(false_color_list, dim=data.coords["time"])
            
            # Apply clean style
            plt.rcParams.update(self.rc_params)
            
            # Create the plot
            fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
            
            plot = false_color_stack.plot.imshow(
                x="x", 
                y="y", 
                col="time", 
                col_wrap=col_wrap
            )
            
            fig.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Saved false color plot to {save_path}")
            
            logger.info(f"Created false color plot with {len(data.time)} time steps")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating false color plot: {e}")
            raise
    
    def plot_temporal_statistics(self, 
                               stats: Dict[str, xr.DataArray],
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot temporal statistics as a grid of subplots.
        
        Args:
            stats (Dict[str, xr.DataArray]): Dictionary of temporal statistics
            save_path (Optional[str]): Path to save the figure
            
        Returns:
            plt.Figure: The created figure
        """
        n_stats = len(stats)
        cols = min(3, n_stats)
        rows = (n_stats + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=self.figsize, dpi=self.dpi)
        if n_stats == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, (stat_name, stat_data) in enumerate(stats.items()):
            if i < len(axes):
                ax = axes[i]
                im = stat_data.plot.imshow(ax=ax, cmap='viridis', add_colorbar=True)
                ax.set_title(f'{stat_name.upper()}', fontweight='bold')
                ax.set_xlabel('')
                ax.set_ylabel('')
        
        # Hide unused subplots
        for i in range(n_stats, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved temporal statistics plot to {save_path}")
        
        logger.info(f"Created temporal statistics plot with {n_stats} statistics")
        return fig
    
    def plot_ndvi_histogram(self, 
                          ndvi: xr.DataArray,
                          bins: int = 50,
                          title: str = "NDVI Distribution",
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot histogram of NDVI values across all time steps.
        
        Args:
            ndvi (xr.DataArray): Multi-temporal NDVI data
            bins (int): Number of histogram bins
            title (str): Title for the plot
            save_path (Optional[str]): Path to save the figure
            
        Returns:
            plt.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        # Flatten all NDVI values and remove NaNs
        ndvi_values = ndvi.values.flatten()
        ndvi_values = ndvi_values[~np.isnan(ndvi_values)]
        
        # Create histogram
        ax.hist(ndvi_values, bins=bins, alpha=0.7, color='green', edgecolor='black')
        ax.set_xlabel('NDVI Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_ndvi = np.mean(ndvi_values)
        std_ndvi = np.std(ndvi_values)
        stats_text = f'Mean: {mean_ndvi:.3f}\\nStd: {std_ndvi:.3f}\\nN: {len(ndvi_values):,}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved NDVI histogram to {save_path}")
        
        logger.info("Created NDVI histogram")
        return fig
    
    def plot_hotspot_summary(self, 
                           hotspot_stats: Dict[str, Any],
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Create summary visualization of hotspot statistics.
        
        Args:
            hotspot_stats (Dict[str, Any]): Hotspot statistics dictionary
            save_path (Optional[str]): Path to save the figure
            
        Returns:
            plt.Figure: The created figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=self.dpi)
        
        # Extract summary data
        summary = hotspot_stats.get('summary', {})
        
        # Pie chart of hotspot distribution
        labels = ['Hot Spots', 'Cold Spots', 'Not Significant']
        sizes = [
            summary.get('hot_spots_percentage', 0),
            summary.get('cold_spots_percentage', 0),
            summary.get('not_significant_percentage', 0)
        ]
        colors = ['#FF4500', '#4169E1', '#LIGHTGRAY']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Hotspot Distribution', fontsize=14, fontweight='bold')
        
        # Bar chart of significance levels
        categories = []
        percentages = []
        bar_colors = []
        
        for key, value in hotspot_stats.items():
            if key != 'summary' and isinstance(value, dict):
                categories.append(key.replace('_', ' ').title())
                percentages.append(value.get('percentage', 0))
                if 'hot' in key:
                    bar_colors.append('#FF4500')
                elif 'cold' in key:
                    bar_colors.append('#4169E1')
                else:
                    bar_colors.append('#LIGHTGRAY')
        
        if categories:
            bars = ax2.bar(categories, percentages, color=bar_colors, alpha=0.7, edgecolor='black')
            ax2.set_ylabel('Percentage (%)', fontsize=12)
            ax2.set_title('Significance Levels', fontsize=14, fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, pct in zip(bars, percentages):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{pct:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved hotspot summary plot to {save_path}")
        
        logger.info("Created hotspot summary visualization")
        return fig
    
    def create_comparison_plot(self, 
                             data_dict: Dict[str, xr.DataArray],
                             title: str = "Data Comparison",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comparison plot for multiple datasets.
        
        Args:
            data_dict (Dict[str, xr.DataArray]): Dictionary of datasets to compare
            title (str): Title for the plot
            save_path (Optional[str]): Path to save the figure
            
        Returns:
            plt.Figure: The created figure
        """
        n_datasets = len(data_dict)
        cols = min(3, n_datasets)
        rows = (n_datasets + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=self.figsize, dpi=self.dpi)
        if n_datasets == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, (name, data) in enumerate(data_dict.items()):
            if i < len(axes):
                ax = axes[i]
                
                # Plot mean across time if multiple time steps
                if 'time' in data.dims and len(data.time) > 1:
                    plot_data = data.mean(dim='time')
                else:
                    plot_data = data.isel(time=0) if 'time' in data.dims else data
                
                im = plot_data.plot.imshow(ax=ax, cmap='viridis', add_colorbar=True)
                ax.set_title(name.replace('_', ' ').title(), fontweight='bold')
                ax.set_xlabel('')
                ax.set_ylabel('')
        
        # Hide unused subplots
        for i in range(n_datasets, len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {save_path}")
        
        logger.info(f"Created comparison plot with {n_datasets} datasets")
        return fig
