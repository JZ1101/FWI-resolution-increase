#!/usr/bin/env python3
"""
Step 1: Process ERA5-Land data safely
- Load the 8 ERA5-Land NetCDF files
- Concatenate along time dimension
- Regrid to 1km master grid
- Save to processed directory
"""

import xarray as xr
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_master_grid():
    """Load the 1km master grid"""
    master_path = Path("data/01_processed/master_grid_1km_2010_2017.nc")
    if master_path.exists():
        print(f"Loading master grid from: {master_path}")
        ds = xr.open_dataset(master_path)
        return ds
    else:
        print(f"ERROR: Master grid not found at {master_path}")
        return None

def process_era5_land():
    """Process ERA5-Land files"""
    print("=== STEP 1: PROCESS ERA5-LAND DATA ===\n")
    
    # Define paths
    source_dir = Path("data/00_raw/portugal_2010_2017/ERA5_reanalysis_ 10km_atmospheric_land-surface_parameters")
    output_dir = Path("data/01_processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # List ERA5-Land files (excluding data_0.nc which seems to be a sample)
    land_files = sorted(source_dir.glob("era5_land_may_nov_*.nc"))
    
    print(f"Found {len(land_files)} ERA5-Land files:")
    for f in land_files:
        print(f"  - {f.name}")
    
    if not land_files:
        print("ERROR: No ERA5-Land files found!")
        return
    
    # Load and concatenate all files
    print("\nLoading and concatenating files...")
    datasets = []
    
    for file_path in land_files:
        print(f"  Loading: {file_path.name}")
        try:
            ds = xr.open_dataset(file_path)
            
            # Check dimensions
            print(f"    Dimensions: {dict(ds.dims)}")
            
            # Rename valid_time to time if needed
            if 'valid_time' in ds.dims and 'time' not in ds.dims:
                ds = ds.rename({'valid_time': 'time'})
            
            # Store dataset
            datasets.append(ds)
            
        except Exception as e:
            print(f"    ERROR loading {file_path.name}: {e}")
    
    if not datasets:
        print("ERROR: No datasets loaded successfully!")
        return
    
    # Concatenate along time dimension
    print("\nConcatenating datasets along time...")
    try:
        combined = xr.concat(datasets, dim='time')
        print(f"  Combined shape: {dict(combined.dims)}")
        print(f"  Variables: {list(combined.data_vars)}")
        print(f"  Time range: {combined.time.min().values} to {combined.time.max().values}")
        
        # Sort by time
        combined = combined.sortby('time')
        
        # Check for duplicates
        time_counts = combined.time.to_pandas().value_counts()
        duplicates = time_counts[time_counts > 1]
        if len(duplicates) > 0:
            print(f"  WARNING: Found {len(duplicates)} duplicate timestamps")
            print("  Removing duplicates by keeping first occurrence...")
            combined = combined.sel(time=~combined.time.to_index().duplicated())
            print(f"  Shape after deduplication: {dict(combined.dims)}")
        
    except Exception as e:
        print(f"ERROR concatenating: {e}")
        return
    
    # Load master grid for regridding
    print("\nLoading master grid...")
    master = load_master_grid()
    
    if master is None:
        print("ERROR: Cannot proceed without master grid")
        return
    
    print(f"  Master grid shape: lat={len(master.latitude)}, lon={len(master.longitude)}")
    
    # Regrid to master grid
    print("\nRegridding to 1km master grid...")
    try:
        # Ensure consistent coordinate names
        if 'lat' in combined.dims:
            combined = combined.rename({'lat': 'latitude'})
        if 'lon' in combined.dims:
            combined = combined.rename({'lon': 'longitude'})
        
        # Interpolate to master grid
        regridded = combined.interp(
            latitude=master.latitude,
            longitude=master.longitude,
            method='linear'
        )
        
        print(f"  Regridded shape: {dict(regridded.dims)}")
        
        # Add attributes
        regridded.attrs['description'] = 'ERA5-Land data regridded to 1km'
        regridded.attrs['source'] = 'ERA5-Land reanalysis'
        regridded.attrs['processing'] = 'Concatenated and regridded to 1km master grid'
        
        # Save processed file
        output_path = output_dir / "processed_era5_land_1km.nc"
        print(f"\nSaving to: {output_path}")
        
        # Encode with compression
        encoding = {}
        for var in regridded.data_vars:
            encoding[var] = {'zlib': True, 'complevel': 4}
        
        regridded.to_netcdf(output_path, encoding=encoding)
        print("  Saved successfully!")
        
        # Verify file
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  File size: {file_size_mb:.1f} MB")
        
        print("\n=== STEP 1 COMPLETE ===")
        print(f"Successfully processed {len(land_files)} ERA5-Land files")
        print(f"Output: {output_path}")
        
    except Exception as e:
        print(f"ERROR during regridding: {e}")

if __name__ == "__main__":
    process_era5_land()