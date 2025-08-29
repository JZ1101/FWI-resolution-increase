#!/usr/bin/env python3
"""
Check the quality and completeness of the preprocessed unified dataset
"""

import xarray as xr
import numpy as np
from pathlib import Path
import sys

def check_dataset(dataset_path):
    """Comprehensive dataset quality check"""
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return False
    
    print(f"\n{'='*70}")
    print("DATASET QUALITY CHECK")
    print('='*70)
    
    # Load dataset
    print(f"\n📂 Loading dataset: {dataset_path}")
    ds = xr.open_dataset(dataset_path)
    
    # Basic info
    print(f"\n📊 DATASET DIMENSIONS:")
    print(f"   Time points: {len(ds.time)}")
    print(f"   Latitude points: {len(ds.latitude)}")
    print(f"   Longitude points: {len(ds.longitude)}")
    print(f"   Total grid cells: {len(ds.latitude) * len(ds.longitude):,}")
    
    # Variables
    print(f"\n📋 VARIABLES ({len(ds.data_vars)} total):")
    for var in ds.data_vars:
        var_data = ds[var]
        shape = var_data.shape
        size_mb = var_data.nbytes / 1e6
        dtype = var_data.dtype
        print(f"   {var:20s} - Shape: {str(shape):20s} Size: {size_mb:8.1f} MB  Type: {dtype}")
    
    # Time coverage
    print(f"\n📅 TEMPORAL COVERAGE:")
    print(f"   Start date: {ds.time.values[0]}")
    print(f"   End date: {ds.time.values[-1]}")
    time_diff = ds.time.diff('time').values
    if len(np.unique(time_diff)) == 1:
        print(f"   ✅ Time series is regular (daily)")
    else:
        print(f"   ⚠️  Time series has gaps or irregularities")
    
    # Spatial coverage
    print(f"\n🗺️  SPATIAL COVERAGE:")
    print(f"   Latitude range: {ds.latitude.min().values:.2f}° to {ds.latitude.max().values:.2f}°")
    print(f"   Longitude range: {ds.longitude.min().values:.2f}° to {ds.longitude.max().values:.2f}°")
    lat_res = float(ds.latitude.diff('latitude').mean())
    lon_res = float(ds.longitude.diff('longitude').mean())
    print(f"   Resolution: {lat_res:.3f}° × {lon_res:.3f}° (~{lat_res*111:.1f} km)")
    
    # Data quality for each variable
    print(f"\n🔍 DATA QUALITY ANALYSIS:")
    for var in ds.data_vars:
        var_data = ds[var].values
        nan_count = np.isnan(var_data).sum()
        nan_pct = 100 * nan_count / var_data.size
        
        # Get valid data stats
        valid_data = var_data[~np.isnan(var_data)]
        if len(valid_data) > 0:
            min_val = valid_data.min()
            max_val = valid_data.max()
            mean_val = valid_data.mean()
            std_val = valid_data.std()
        else:
            min_val = max_val = mean_val = std_val = np.nan
        
        print(f"\n   {var}:")
        print(f"      NaN values: {nan_count:,} ({nan_pct:.1f}%)")
        print(f"      Range: [{min_val:.3f}, {max_val:.3f}]")
        print(f"      Mean ± Std: {mean_val:.3f} ± {std_val:.3f}")
        
        # Special checks for FWI
        if var == 'fwi':
            if min_val < 0:
                print(f"      ⚠️  Warning: Negative FWI values detected!")
            if max_val > 100:
                print(f"      ⚠️  Warning: FWI values exceed 100!")
            if nan_pct > 50:
                print(f"      ❌ Error: More than 50% NaN values!")
            else:
                print(f"      ✅ FWI data looks reasonable")
    
    # Land mask analysis
    if 'land_mask' in ds.data_vars:
        land_mask = ds['land_mask'].values
        land_pixels = (land_mask == 1).sum()
        ocean_pixels = (land_mask == 0).sum()
        total_pixels = land_mask.size
        print(f"\n🏞️  LAND MASK:")
        print(f"   Land pixels: {land_pixels:,} ({100*land_pixels/total_pixels:.1f}%)")
        print(f"   Ocean pixels: {ocean_pixels:,} ({100*ocean_pixels/total_pixels:.1f}%)")
    
    # Memory usage
    total_size_gb = ds.nbytes / 1e9
    print(f"\n💾 MEMORY USAGE:")
    print(f"   Total dataset size: {total_size_gb:.2f} GB")
    
    # Metadata
    print(f"\n📝 METADATA:")
    for key, value in ds.attrs.items():
        print(f"   {key}: {value}")
    
    # Specific date check - Pedrógão Grande fire
    print(f"\n🔥 FIRE EVENT VALIDATION (Pedrógão Grande - 2017-06-17):")
    try:
        if 'fwi' in ds.data_vars:
            fire_date = '2017-06-17'
            fire_lat = 39.95
            fire_lon = -8.13
            
            fire_fwi = ds['fwi'].sel(
                time=fire_date,
                latitude=fire_lat,
                longitude=fire_lon,
                method='nearest'
            )
            
            print(f"   FWI at fire location: {float(fire_fwi):.1f}")
            if float(fire_fwi) > 20:
                print(f"   ✅ Fire conditions detected (FWI > 20)")
            else:
                print(f"   ⚠️  Low FWI at fire location")
    except Exception as e:
        print(f"   ⚠️  Could not validate fire event: {e}")
    
    print(f"\n{'='*70}")
    print("✅ DATASET CHECK COMPLETE")
    print('='*70)
    
    return True


def main():
    # Check different dataset versions
    datasets = [
        "data/02_final_unified/unified_fwi_dataset_2010_2017.nc",
        "data/02_final_unified/unified_fwi_dataset_2016_2017.nc",
    ]
    
    for dataset_path in datasets:
        if Path(dataset_path).exists():
            check_dataset(dataset_path)
        else:
            print(f"Skipping {dataset_path} (not found)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_dataset(sys.argv[1])
    else:
        main()