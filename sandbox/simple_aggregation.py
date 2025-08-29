#!/usr/bin/env python3
"""
Simple data aggregation - processes one variable at a time to avoid memory issues
"""

import xarray as xr
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=== SIMPLE DATA AGGREGATION ===\n")
    
    # Check what variables already exist
    unified_path = Path("data/02_final_unified/unified_fwi_dataset_2010_2017.nc")
    print(f"Checking existing unified dataset: {unified_path}")
    
    with xr.open_dataset(unified_path) as ds:
        existing_vars = list(ds.data_vars)
        print(f"Existing variables: {existing_vars}")
        print(f"Dimensions: time={len(ds.time)}, lat={len(ds.latitude)}, lon={len(ds.longitude)}")
    
    # Check atmospheric data availability
    atm_path = Path("data/00_raw/portugal_2010_2017/ERA5_reanalysis_25km_atmospheric_parameters")
    print(f"\nChecking atmospheric data: {atm_path}")
    
    if atm_path.exists():
        # Count available files
        nc_files = list(atm_path.glob("*.nc"))
        print(f"Found {len(nc_files)} NetCDF files")
        
        # List first 10 files
        print("\nSample files:")
        for f in nc_files[:10]:
            print(f"  - {f.name}")
        
        # Try to load one file as example
        sample_file = nc_files[0]
        print(f"\nChecking sample file: {sample_file.name}")
        with xr.open_dataset(sample_file) as ds:
            print(f"  Variables: {list(ds.data_vars)}")
            print(f"  Dimensions: {dict(ds.dims)}")
            if 'time' in ds.dims:
                print(f"  Time steps: {len(ds.time)}")
    
    # Check ERA5-Land data
    land_path = Path("data/00_raw/portugal_2010_2017/ERA5_reanalysis_ 10km_atmospheric_land-surface_parameters/data_0.nc")
    print(f"\nChecking ERA5-Land data: {land_path}")
    
    if land_path.exists():
        with xr.open_dataset(land_path) as ds:
            print(f"  Variables: {list(ds.data_vars)}")
            print(f"  Dimensions: {dict(ds.dims)}")
    else:
        print("  ERA5-Land file not found")
    
    print("\n=== SUMMARY ===")
    print("Data aggregation assessment complete.")
    print("Next step: Process individual variables one at a time to avoid memory issues")

if __name__ == "__main__":
    main()