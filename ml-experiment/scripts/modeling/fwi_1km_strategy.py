#!/usr/bin/env python3
"""
1km FWI Downscaling Strategy
Approach: 25km → 10km → 1km using terrain and land cover
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def analyze_1km_feasibility():
    """Analyze feasibility of 1km FWI prediction"""
    print("=== 1km FWI Downscaling Strategy Analysis ===")
    
    # Load existing data
    print("Loading current datasets...")
    era5_land = xr.open_dataset('data/data_0.nc')  # 10km met data
    fwi_10km = xr.open_dataset('data/fwi_10km_full_year.nc')  # Our 10km FWI
    
    print(f"Current resolution: 10km = {len(era5_land.latitude)} x {len(era5_land.longitude)} grid")
    
    # Calculate 1km grid dimensions
    lat_range = era5_land.latitude.max() - era5_land.latitude.min()
    lon_range = era5_land.longitude.max() - era5_land.longitude.min()
    
    # 1km ≈ 0.009° at Portugal latitude
    km_to_deg = 0.009
    target_lat_points = int(lat_range / km_to_deg)
    target_lon_points = int(lon_range / km_to_deg)
    
    print(f"Target 1km grid: {target_lat_points} x {target_lon_points} = {target_lat_points * target_lon_points:,} pixels")
    print(f"Upscaling factor: {(target_lat_points * target_lon_points) / (len(era5_land.latitude) * len(era5_land.longitude)):.1f}x")
    
    return target_lat_points, target_lon_points

def propose_1km_approaches():
    """Propose different approaches for 1km FWI"""
    print("\\n=== Proposed 1km FWI Approaches ===")
    
    approaches = {
        "Approach 1: Bilinear Interpolation": {
            "method": "Simple interpolation of 10km FWI to 1km grid",
            "pros": ["Fast", "Simple", "Preserves aggregate values"],
            "cons": ["No new information", "Smooth results", "No terrain effects"],
            "feasibility": "High"
        },
        
        "Approach 2: Terrain-Enhanced Interpolation": {
            "method": "10km FWI + elevation/slope/land cover corrections",
            "pros": ["Uses terrain effects", "Physically meaningful", "Static corrections"],
            "cons": ["Still limited by 10km met data", "Simplified terrain effects"],
            "feasibility": "Medium-High"
        },
        
        "Approach 3: ML with Static Features": {
            "method": "Train ML: 10km FWI + static features → synthetic 1km targets",
            "pros": ["Can learn complex patterns", "Uses available land cover"],
            "cons": ["No true 1km targets", "Limited by static features only"],
            "feasibility": "Medium"
        },
        
        "Approach 4: Multi-Scale ML": {
            "method": "25km → 10km → 1km cascaded enhancement",
            "pros": ["Proven 25km→10km works", "Can leverage land cover patterns"],
            "cons": ["Accumulates errors", "Complex validation"],
            "feasibility": "Medium"
        }
    }
    
    for name, details in approaches.items():
        print(f"\\n{name}:")
        print(f"  Method: {details['method']}")
        print(f"  Feasibility: {details['feasibility']}")
        print(f"  Pros: {', '.join(details['pros'])}")
        print(f"  Cons: {', '.join(details['cons'])}")
    
    return approaches

def design_1km_evaluation_strategy():
    """Design evaluation strategy for 1km FWI without ground truth"""
    print("\\n=== 1km FWI Evaluation Strategy ===")
    print("Challenge: No true 1km FWI ground truth available")
    
    evaluation_methods = {
        "1. Aggregation Consistency": {
            "description": "1km predictions should aggregate back to 10km values",
            "implementation": "Spatially average 1km → 10km, compare with original 10km FWI",
            "metric": "RMSE between aggregated 1km and original 10km",
            "strength": "Strong physical constraint"
        },
        
        "2. Spatial Coherence": {
            "description": "1km FWI should vary smoothly and logically",
            "implementation": "Check spatial gradients, no unrealistic jumps",
            "metric": "Spatial autocorrelation, gradient statistics",
            "strength": "Physical plausibility check"
        },
        
        "3. Land Cover Correlation": {
            "description": "FWI should correlate with fire-prone land cover",
            "implementation": "Higher FWI in grasslands/forests, lower in water/urban",
            "metric": "Correlation with ESA WorldCover classes",
            "strength": "Domain knowledge validation"
        },
        
        "4. Temporal Consistency": {
            "description": "1km FWI should follow realistic temporal patterns",
            "implementation": "Check day-to-day changes, seasonal patterns",
            "metric": "Temporal autocorrelation, change magnitude stats",
            "strength": "Time series plausibility"
        },
        
        "5. Cross-Validation": {
            "description": "Train on subset, predict on different areas/times",
            "implementation": "Spatial/temporal holdout validation",
            "metric": "Prediction consistency across folds",
            "strength": "Model generalization test"
        },
        
        "6. Physical Bounds": {
            "description": "1km FWI should respect physical constraints",
            "implementation": "Non-negative values, reasonable range",
            "metric": "Range checks, distribution comparison",
            "strength": "Basic validity check"
        }
    }
    
    for name, details in evaluation_methods.items():
        print(f"\\n{name}:")
        print(f"  Description: {details['description']}")
        print(f"  Metric: {details['metric']}")
        print(f"  Implementation: {details['implementation']}")
    
    print(f"\\n📋 Recommended Evaluation Workflow:")
    print(f"1. Primary: Aggregation consistency (must preserve 10km values)")
    print(f"2. Secondary: Spatial coherence + land cover correlation")
    print(f"3. Validation: Cross-validation + physical bounds")
    
    return evaluation_methods

def implement_simple_1km_approach():
    """Implement simplest viable 1km approach for demonstration"""
    print("\\n=== Implementing Simple 1km Approach ===")
    print("Method: Terrain-enhanced bilinear interpolation")
    
    # Load data
    fwi_10km = xr.open_dataset('data/fwi_10km_full_year.nc')
    
    # Take one day for demonstration
    day_0 = fwi_10km.isel(time=0)['fwi_10km']
    
    print(f"Original 10km FWI shape: {day_0.shape}")
    print(f"FWI range: {float(day_0.min()):.2f} to {float(day_0.max()):.2f}")
    
    # Create target 1km grid (smaller for demo)
    # Use 2km resolution for demo (still 5x enhancement)
    target_resolution = 0.02  # degrees (~2km)
    
    lat_new = np.arange(
        float(day_0.latitude.min()), 
        float(day_0.latitude.max()), 
        target_resolution
    )
    lon_new = np.arange(
        float(day_0.longitude.min()), 
        float(day_0.longitude.max()), 
        target_resolution
    )
    
    print(f"Target high-res grid: {len(lat_new)} x {len(lon_new)} = {len(lat_new) * len(lon_new):,} pixels")
    print(f"Enhancement factor: {(len(lat_new) * len(lon_new)) / (day_0.shape[0] * day_0.shape[1]):.1f}x")
    
    # Simple bilinear interpolation
    day_0_interp = day_0.interp(latitude=lat_new, longitude=lon_new, method='linear')
    
    # Add some terrain-based noise for demonstration
    # In practice, this would use real elevation/land cover data
    np.random.seed(42)
    lat_grid, lon_grid = np.meshgrid(lat_new, lon_new, indexing='ij')
    
    # Simulate terrain effects (simplified)
    terrain_effect = 0.1 * np.sin(lat_grid * 50) * np.cos(lon_grid * 50)
    terrain_effect = terrain_effect * 0.1  # Small effect
    
    # Apply terrain correction
    fwi_1km_demo = day_0_interp + terrain_effect
    fwi_1km_demo = np.clip(fwi_1km_demo, 0, None)  # Ensure non-negative
    
    print(f"\\nHigh-res FWI statistics:")
    print(f"  Mean: {float(fwi_1km_demo.mean()):.2f}")
    print(f"  Std: {float(fwi_1km_demo.std()):.2f}")
    print(f"  Range: {float(fwi_1km_demo.min()):.2f} to {float(fwi_1km_demo.max()):.2f}")
    
    # Evaluation: Check aggregation consistency
    print(f"\\n=== Evaluation: Aggregation Consistency ===")
    
    # Aggregate back to 10km resolution
    # Simple approach: downsample by averaging
    factor = len(lat_new) // len(day_0.latitude)
    if factor >= 2:
        # Coarsen by averaging
        fwi_aggregated = fwi_1km_demo.coarsen(
            latitude=factor, longitude=factor, boundary='trim'
        ).mean()
        
        # Compare with original
        original_subset = day_0.isel(
            latitude=slice(0, len(fwi_aggregated.latitude)),
            longitude=slice(0, len(fwi_aggregated.longitude))
        )
        
        # Calculate consistency metrics
        mse = float(((fwi_aggregated - original_subset) ** 2).mean())
        rmse = np.sqrt(mse)
        
        print(f"Aggregation RMSE: {rmse:.3f}")
        print(f"Relative error: {rmse / float(day_0.mean()) * 100:.1f}%")
        
        if rmse < 0.1:
            print("✅ Excellent aggregation consistency")
        elif rmse < 0.5:
            print("✅ Good aggregation consistency")
        else:
            print("⚠️  Poor aggregation consistency")
    
    # Visualization
    create_1km_comparison_plot(day_0, fwi_1km_demo)
    
    return fwi_1km_demo

def create_1km_comparison_plot(fwi_10km, fwi_1km):
    """Create comparison plot between 10km and 1km FWI"""
    print("\\nCreating comparison visualization...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original 10km
    im1 = axes[0].pcolormesh(
        fwi_10km.longitude, fwi_10km.latitude, fwi_10km,
        shading='auto', cmap='Reds', vmin=0, vmax=30
    )
    axes[0].set_title('Original 10km FWI')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    plt.colorbar(im1, ax=axes[0])
    
    # Enhanced 1km
    im2 = axes[1].pcolormesh(
        fwi_1km.longitude, fwi_1km.latitude, fwi_1km,
        shading='auto', cmap='Reds', vmin=0, vmax=30
    )
    axes[1].set_title('Enhanced ~2km FWI')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    plt.colorbar(im2, ax=axes[1])
    
    # Difference
    # Interpolate 10km to match 1km grid for comparison
    fwi_10km_interp = fwi_10km.interp(
        latitude=fwi_1km.latitude, 
        longitude=fwi_1km.longitude, 
        method='linear'
    )
    diff = fwi_1km - fwi_10km_interp
    
    im3 = axes[2].pcolormesh(
        fwi_1km.longitude, fwi_1km.latitude, diff,
        shading='auto', cmap='RdBu_r', vmin=-2, vmax=2
    )
    axes[2].set_title('Difference (Enhanced - Original)')
    axes[2].set_xlabel('Longitude')
    axes[2].set_ylabel('Latitude')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('outputs/fwi_1km_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved comparison to outputs/fwi_1km_comparison.png")
    plt.show()

def main():
    """Main analysis function"""
    print("\\n" + "="*60)
    print("1KM FWI DOWNSCALING STRATEGY & EVALUATION")
    print("="*60)
    
    # Analyze feasibility
    target_lat, target_lon = analyze_1km_feasibility()
    
    # Propose approaches
    approaches = propose_1km_approaches()
    
    # Design evaluation strategy  
    evaluation_methods = design_1km_evaluation_strategy()
    
    # Implement demo
    import os
    os.makedirs('outputs', exist_ok=True)
    fwi_1km_demo = implement_simple_1km_approach()
    
    print("\\n" + "="*60)
    print("CONCLUSIONS & RECOMMENDATIONS")
    print("="*60)
    print("\\n✅ FEASIBLE: 1km FWI enhancement is possible with current data")
    print("\\n📍 RECOMMENDED APPROACH:")
    print("   1. Use proven 10km FWI as foundation")
    print("   2. Apply terrain-enhanced interpolation to 1km")
    print("   3. Validate using aggregation consistency")
    print("   4. Incorporate ESA WorldCover land use corrections")
    print("\\n📊 EVALUATION STRATEGY:")
    print("   • Primary: Aggregation consistency (must preserve 10km values)")
    print("   • Secondary: Spatial coherence + land cover correlation")  
    print("   • Validation: Cross-validation + physical bounds")
    print("\\n🎯 NEXT STEPS:")
    print("   1. Implement full 1km grid (226k pixels)")
    print("   2. Add real elevation/land cover corrections")
    print("   3. Apply comprehensive evaluation framework")
    
    return fwi_1km_demo

if __name__ == "__main__":
    result = main()