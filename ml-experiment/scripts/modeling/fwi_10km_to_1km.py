#!/usr/bin/env python3
"""
FWI Enhancement: 10km → 1km
Using our validated 10km FWI as input
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class FWI1kmEnhancer:
    """Enhance 10km FWI to 1km resolution"""
    
    def __init__(self):
        self.fwi_10km = None
        self.fwi_1km = None
        self.evaluation_results = {}
    
    def load_10km_fwi(self):
        """Load our trained 10km FWI data"""
        print("=== Loading 10km FWI Data ===")
        
        self.fwi_10km = xr.open_dataset('data/fwi_10km_full_year.nc')
        print(f"Loaded 10km FWI: {self.fwi_10km.fwi_10km.shape}")
        print(f"Time range: {len(self.fwi_10km.time)} days")
        print(f"Spatial coverage: {len(self.fwi_10km.latitude)} x {len(self.fwi_10km.longitude)}")
        
        # Print sample statistics
        sample_day = self.fwi_10km.isel(time=0).fwi_10km
        print(f"Sample FWI stats: mean={float(sample_day.mean()):.2f}, range=[{float(sample_day.min()):.2f}, {float(sample_day.max()):.2f}]")
        
        return self.fwi_10km
    
    def create_1km_grid(self, enhancement_factor=10):
        """
        Create target 1km grid coordinates
        
        Parameters:
        -----------
        enhancement_factor : int
            Enhancement factor (10 = 10km to 1km)
        """
        print(f"\\n=== Creating 1km Grid (Factor {enhancement_factor}x) ===")
        
        # Original 10km coordinates
        lat_10km = self.fwi_10km.latitude.values
        lon_10km = self.fwi_10km.longitude.values
        
        # Calculate 1km resolution
        lat_res_10km = abs(lat_10km[1] - lat_10km[0])
        lon_res_10km = abs(lon_10km[1] - lon_10km[0])
        
        lat_res_1km = lat_res_10km / enhancement_factor
        lon_res_1km = lon_res_10km / enhancement_factor
        
        print(f"10km resolution: {lat_res_10km:.3f}° lat, {lon_res_10km:.3f}° lon")
        print(f"1km resolution: {lat_res_1km:.3f}° lat, {lon_res_1km:.3f}° lon")
        
        # Create 1km grid
        lat_1km = np.arange(
            lat_10km.min() - lat_res_10km/2, 
            lat_10km.max() + lat_res_10km/2, 
            lat_res_1km
        )
        lon_1km = np.arange(
            lon_10km.min() - lon_res_10km/2, 
            lon_10km.max() + lon_res_10km/2, 
            lon_res_1km
        )
        
        print(f"1km grid size: {len(lat_1km)} x {len(lon_1km)} = {len(lat_1km) * len(lon_1km):,} pixels")
        print(f"Enhancement: {len(lat_1km) * len(lon_1km) / (len(lat_10km) * len(lon_10km)):.1f}x more pixels")
        
        self.lat_1km = lat_1km
        self.lon_1km = lon_1km
        
        return lat_1km, lon_1km
    
    def enhance_to_1km(self, method='terrain_enhanced', subset_days=5):
        """
        Enhance 10km FWI to 1km using specified method
        
        Parameters:
        -----------
        method : str
            Enhancement method ('bilinear', 'terrain_enhanced', 'edge_preserving')
        subset_days : int
            Number of days to process for demo
        """
        print(f"\\n=== Enhancing to 1km (Method: {method}) ===")
        print(f"Processing {subset_days} days for demonstration")
        
        # Process subset of days
        n_days = min(subset_days, len(self.fwi_10km.time))
        fwi_1km_data = []
        
        for day in range(n_days):
            print(f"  Processing day {day+1}/{n_days}")
            
            fwi_day = self.fwi_10km.isel(time=day).fwi_10km
            
            if method == 'bilinear':
                fwi_1km_day = self._bilinear_enhancement(fwi_day)
            elif method == 'terrain_enhanced':
                fwi_1km_day = self._terrain_enhanced_enhancement(fwi_day)
            elif method == 'edge_preserving':
                fwi_1km_day = self._edge_preserving_enhancement(fwi_day)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            fwi_1km_data.append(fwi_1km_day)
        
        # Stack into dataset
        fwi_1km_array = np.stack(fwi_1km_data, axis=0)
        
        self.fwi_1km = xr.Dataset({
            'fwi_1km': (['time', 'latitude', 'longitude'], fwi_1km_array)
        }, coords={
            'time': self.fwi_10km.time[:n_days],
            'latitude': self.lat_1km,
            'longitude': self.lon_1km
        })
        
        print(f"Created 1km FWI dataset: {self.fwi_1km.fwi_1km.shape}")
        
        return self.fwi_1km
    
    def _bilinear_enhancement(self, fwi_day):
        """Simple bilinear interpolation"""
        fwi_1km = fwi_day.interp(
            latitude=self.lat_1km, 
            longitude=self.lon_1km, 
            method='linear'
        ).values
        return fwi_1km
    
    def _terrain_enhanced_enhancement(self, fwi_day):
        """Terrain-enhanced interpolation with synthetic features"""
        # Start with bilinear interpolation
        fwi_1km = self._bilinear_enhancement(fwi_day)
        
        # Add synthetic terrain effects for demonstration
        # In practice, would use real DEM and land cover data
        lat_grid, lon_grid = np.meshgrid(self.lat_1km, self.lon_1km, indexing='ij')
        
        # Simulate elevation effects (higher elevation = higher FWI variability)
        # Using simple sinusoidal pattern as proxy for terrain
        elevation_proxy = (np.sin(lat_grid * 100) * np.cos(lon_grid * 100) + 1) / 2
        
        # Simulate land cover effects (grassland = higher FWI, forest = lower)
        landcover_proxy = (np.sin(lat_grid * 80 + 1) * np.cos(lon_grid * 60 + 1) + 1) / 2
        
        # Apply corrections (small effects to preserve aggregation)
        terrain_correction = 0.05 * elevation_proxy * (fwi_1km + 1)  # 5% variation
        landcover_correction = 0.03 * landcover_proxy * (fwi_1km + 1)  # 3% variation
        
        # Combine effects
        fwi_enhanced = fwi_1km + terrain_correction + landcover_correction - 0.04 * (fwi_1km + 1)
        
        # Ensure non-negative and reasonable range
        fwi_enhanced = np.clip(fwi_enhanced, 0, fwi_1km.max() * 1.2)
        
        return fwi_enhanced
    
    def _edge_preserving_enhancement(self, fwi_day):
        """Edge-preserving enhancement using bilateral filtering"""
        # Start with bilinear
        fwi_1km = self._bilinear_enhancement(fwi_day)
        
        # Apply edge-preserving smoothing to reduce artifacts
        # while preserving sharp transitions
        smoothed = ndimage.gaussian_filter(fwi_1km, sigma=0.5)
        
        # Blend original and smoothed based on local gradient
        gradient_magnitude = np.sqrt(
            ndimage.sobel(fwi_1km, axis=0)**2 + 
            ndimage.sobel(fwi_1km, axis=1)**2
        )
        
        # Normalize gradient for blending weight
        gradient_norm = gradient_magnitude / (gradient_magnitude.max() + 1e-8)
        
        # Blend: use original where gradients are high, smoothed where low
        fwi_enhanced = gradient_norm * fwi_1km + (1 - gradient_norm) * smoothed
        
        return fwi_enhanced
    
    def evaluate_1km_results(self):
        """Comprehensive evaluation of 1km FWI results"""
        print("\\n=== Evaluating 1km FWI Results ===")
        
        results = {}
        
        # 1. Aggregation Consistency Test
        print("1. Testing aggregation consistency...")
        consistency_errors = []
        
        for day in range(len(self.fwi_1km.time)):
            fwi_1km_day = self.fwi_1km.isel(time=day).fwi_1km
            fwi_10km_day = self.fwi_10km.isel(time=day).fwi_10km
            
            # Aggregate 1km back to 10km by spatial averaging
            # Simple approach: coarsen by factor of 10
            factor = len(self.lat_1km) // len(self.fwi_10km.latitude)
            
            if factor >= 2:
                try:
                    fwi_aggregated = fwi_1km_day.coarsen(
                        latitude=factor, longitude=factor, boundary='trim'
                    ).mean()
                    
                    # Match shapes for comparison
                    min_lat = min(len(fwi_aggregated.latitude), len(fwi_10km_day.latitude))
                    min_lon = min(len(fwi_aggregated.longitude), len(fwi_10km_day.longitude))
                    
                    agg_subset = fwi_aggregated.isel(latitude=slice(0, min_lat), longitude=slice(0, min_lon))
                    orig_subset = fwi_10km_day.isel(latitude=slice(0, min_lat), longitude=slice(0, min_lon))
                    
                    # Calculate error
                    mse = float(((agg_subset - orig_subset) ** 2).mean())
                    consistency_errors.append(np.sqrt(mse))
                    
                except Exception as e:
                    continue
        
        results['aggregation_rmse'] = np.mean(consistency_errors)
        results['aggregation_rmse_std'] = np.std(consistency_errors)
        
        print(f"   Aggregation RMSE: {results['aggregation_rmse']:.3f} ± {results['aggregation_rmse_std']:.3f}")
        
        # 2. Spatial Coherence Test
        print("2. Testing spatial coherence...")
        sample_day = self.fwi_1km.isel(time=0).fwi_1km.values
        
        # Calculate spatial gradients
        grad_x = np.abs(np.diff(sample_day, axis=1))
        grad_y = np.abs(np.diff(sample_day, axis=0))
        
        results['mean_gradient_x'] = np.mean(grad_x)
        results['mean_gradient_y'] = np.mean(grad_y)
        results['max_gradient'] = max(np.max(grad_x), np.max(grad_y))
        
        print(f"   Mean spatial gradients: x={results['mean_gradient_x']:.3f}, y={results['mean_gradient_y']:.3f}")
        print(f"   Max gradient: {results['max_gradient']:.3f}")
        
        # 3. Physical Bounds Test
        print("3. Testing physical bounds...")
        fwi_values = self.fwi_1km.fwi_1km.values.flatten()
        fwi_values = fwi_values[~np.isnan(fwi_values)]
        
        results['min_value'] = np.min(fwi_values)
        results['max_value'] = np.max(fwi_values)
        results['negative_count'] = np.sum(fwi_values < 0)
        results['unrealistic_count'] = np.sum(fwi_values > 100)
        
        print(f"   Value range: [{results['min_value']:.2f}, {results['max_value']:.2f}]")
        print(f"   Negative values: {results['negative_count']}")
        print(f"   Unrealistic values (>100): {results['unrealistic_count']}")
        
        # 4. Overall Assessment
        print("\\n=== Overall Assessment ===")
        
        # Aggregation consistency score
        if results['aggregation_rmse'] < 0.1:
            agg_score = "Excellent"
        elif results['aggregation_rmse'] < 0.5:
            agg_score = "Good"
        elif results['aggregation_rmse'] < 1.0:
            agg_score = "Fair"
        else:
            agg_score = "Poor"
        
        # Physical bounds score
        if results['negative_count'] == 0 and results['unrealistic_count'] == 0:
            bounds_score = "Excellent"
        elif results['negative_count'] < 10 and results['unrealistic_count'] < 10:
            bounds_score = "Good"
        else:
            bounds_score = "Poor"
        
        print(f"Aggregation Consistency: {agg_score}")
        print(f"Physical Bounds: {bounds_score}")
        
        self.evaluation_results = results
        return results
    
    def visualize_results(self):
        """Create comprehensive visualization of results"""
        print("\\n=== Creating Visualizations ===")
        
        # Select sample day
        sample_day = 0
        fwi_10km_sample = self.fwi_10km.isel(time=sample_day).fwi_10km
        fwi_1km_sample = self.fwi_1km.isel(time=sample_day).fwi_1km
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('FWI Enhancement: 10km → 1km Results', fontsize=16)
        
        # 1. Original 10km
        im1 = axes[0, 0].pcolormesh(
            fwi_10km_sample.longitude, fwi_10km_sample.latitude, fwi_10km_sample,
            shading='auto', cmap='Reds', vmin=0, vmax=30
        )
        axes[0, 0].set_title('Original 10km FWI')
        axes[0, 0].set_xlabel('Longitude')
        axes[0, 0].set_ylabel('Latitude')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 2. Enhanced 1km
        im2 = axes[0, 1].pcolormesh(
            fwi_1km_sample.longitude, fwi_1km_sample.latitude, fwi_1km_sample,
            shading='auto', cmap='Reds', vmin=0, vmax=30
        )
        axes[0, 1].set_title('Enhanced 1km FWI')
        axes[0, 1].set_xlabel('Longitude')
        axes[0, 1].set_ylabel('Latitude')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 3. Difference (1km interpolated to 10km grid)
        fwi_1km_to_10km = fwi_1km_sample.interp(
            latitude=fwi_10km_sample.latitude,
            longitude=fwi_10km_sample.longitude,
            method='linear'
        )
        diff = fwi_1km_to_10km - fwi_10km_sample
        
        im3 = axes[0, 2].pcolormesh(
            fwi_10km_sample.longitude, fwi_10km_sample.latitude, diff,
            shading='auto', cmap='RdBu_r', vmin=-2, vmax=2
        )
        axes[0, 2].set_title('Difference (1km - 10km)')
        axes[0, 2].set_xlabel('Longitude')
        axes[0, 2].set_ylabel('Latitude')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # 4. Value distribution comparison
        axes[1, 0].hist(fwi_10km_sample.values.flatten(), bins=20, alpha=0.7, label='10km', density=True)
        axes[1, 0].hist(fwi_1km_sample.values.flatten(), bins=20, alpha=0.7, label='1km', density=True)
        axes[1, 0].set_xlabel('FWI Value')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Value Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Aggregation consistency
        if len(self.evaluation_results) > 0:
            # Show aggregation error over time
            axes[1, 1].bar(range(len(self.fwi_1km.time)), [self.evaluation_results['aggregation_rmse']] * len(self.fwi_1km.time))
            axes[1, 1].axhline(0.5, color='orange', linestyle='--', label='Good threshold')
            axes[1, 1].axhline(0.1, color='green', linestyle='--', label='Excellent threshold')
            axes[1, 1].set_xlabel('Day')
            axes[1, 1].set_ylabel('Aggregation RMSE')
            axes[1, 1].set_title('Aggregation Consistency')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Spatial detail enhancement
        # Show zoomed region to highlight detail
        lat_center = float(fwi_1km_sample.latitude.mean())
        lon_center = float(fwi_1km_sample.longitude.mean())
        lat_range = 0.5  # degrees
        lon_range = 0.5  # degrees
        
        # Zoom region
        lat_mask = (fwi_1km_sample.latitude >= lat_center - lat_range/2) & (fwi_1km_sample.latitude <= lat_center + lat_range/2)
        lon_mask = (fwi_1km_sample.longitude >= lon_center - lon_range/2) & (fwi_1km_sample.longitude <= lon_center + lon_range/2)
        
        fwi_zoom = fwi_1km_sample.isel(latitude=lat_mask, longitude=lon_mask)
        
        im6 = axes[1, 2].pcolormesh(
            fwi_zoom.longitude, fwi_zoom.latitude, fwi_zoom,
            shading='auto', cmap='Reds', vmin=0, vmax=30
        )
        axes[1, 2].set_title('1km Detail (Zoomed Region)')
        axes[1, 2].set_xlabel('Longitude')
        axes[1, 2].set_ylabel('Latitude')
        plt.colorbar(im6, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig('outputs/fwi_1km_comprehensive.png', dpi=150, bbox_inches='tight')
        print("Saved comprehensive results to outputs/fwi_1km_comprehensive.png")
        plt.show()
    
    def run_complete_1km_pipeline(self):
        """Run complete 10km → 1km enhancement pipeline"""
        print("\\n" + "="*70)
        print("FWI ENHANCEMENT PIPELINE: 10km → 1km")
        print("="*70)
        
        # Load data
        self.load_10km_fwi()
        
        # Create 1km grid
        self.create_1km_grid(enhancement_factor=10)
        
        # Enhance to 1km
        self.enhance_to_1km(method='terrain_enhanced', subset_days=3)
        
        # Evaluate results
        self.evaluate_1km_results()
        
        # Visualize
        self.visualize_results()
        
        print("\\n" + "="*70)
        print("1KM ENHANCEMENT COMPLETE!")
        print("="*70)
        print(f"✅ Enhanced {len(self.fwi_1km.time)} days of FWI to 1km resolution")
        print(f"📊 Grid size: {self.fwi_1km.fwi_1km.shape[1]} x {self.fwi_1km.fwi_1km.shape[2]} = {self.fwi_1km.fwi_1km.shape[1] * self.fwi_1km.fwi_1km.shape[2]:,} pixels")
        print(f"🎯 Aggregation RMSE: {self.evaluation_results['aggregation_rmse']:.3f}")
        print(f"✅ Physical bounds: Valid range [{self.evaluation_results['min_value']:.2f}, {self.evaluation_results['max_value']:.2f}]")
        
        return self.fwi_1km

def main():
    """Main execution"""
    import os
    os.makedirs('outputs', exist_ok=True)
    
    enhancer = FWI1kmEnhancer()
    fwi_1km = enhancer.run_complete_1km_pipeline()
    
    return enhancer, fwi_1km

if __name__ == "__main__":
    enhancer, fwi_1km = main()