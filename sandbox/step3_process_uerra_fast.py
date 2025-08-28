#!/usr/bin/env python3
"""
Step 3: Process UERRA data (fast version with better memory management)
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def process_uerra_fast():
    """Fast UERRA processing using xarray interpolation"""
    print("=" * 70)
    print("STEP 3: PROCESS UERRA DATA (OPTIMIZED)")
    print("=" * 70)
    
    # Load master grid
    master = xr.open_dataset("data/01_processed/master_grid_1km_2010_2017.nc")
    print(f"\nMaster grid: {len(master.latitude)} x {len(master.longitude)}")
    
    # List UERRA files
    source_dir = Path("data/00_raw/portugal_2010_2017/UERRA_reanalysis_atmospheric parameters")
    uerra_files = sorted(source_dir.glob("uerra_mescan_*.nc"))
    print(f"\nFound {len(uerra_files)} UERRA files")
    
    # Process each file separately to manage memory
    all_data = []
    
    for file_idx, file_path in enumerate(uerra_files):
        print(f"\nProcessing file {file_idx+1}/{len(uerra_files)}: {file_path.name}")
        
        # Open file
        ds = xr.open_dataset(file_path)
        
        # Check if this is the format we expect
        if 'valid_time' in ds.dims:
            ds = ds.rename({'valid_time': 'time'})
        
        print(f"  Time steps: {len(ds.time)}")
        
        # Convert lon from 0-360 to -180-180 if needed
        lon_vals = ds.longitude.values
        if lon_vals.max() > 180:
            print("  Converting longitude coordinates...")
            lon_fixed = xr.where(ds.longitude > 180, ds.longitude - 360, ds.longitude)
            ds = ds.assign_coords(longitude=lon_fixed)
        
        # Get Portugal subset bounds with buffer
        lat_min, lat_max = 35, 44
        lon_min, lon_max = -11, -5
        
        # For curvilinear grids, we need to find the subset differently
        # UERRA has 2D lat/lon arrays
        lat_2d = ds.latitude.values
        lon_2d = ds.longitude.values
        
        # Fix longitude if needed
        if lon_2d.max() > 180:
            lon_2d = np.where(lon_2d > 180, lon_2d - 360, lon_2d)
        
        # Find bounding box in index space
        mask = (lat_2d >= lat_min) & (lat_2d <= lat_max) & \
               (lon_2d >= lon_min) & (lon_2d <= lon_max)
        
        if not mask.any():
            print("  WARNING: No data in Portugal region, skipping...")
            continue
        
        # Get indices of bounding box
        y_indices, x_indices = np.where(mask)
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        
        print(f"  Subset indices: y=[{y_min}:{y_max}], x=[{x_min}:{x_max}]")
        
        # Extract subset
        subset = ds.isel(y=slice(y_min, y_max+1), x=slice(x_min, x_max+1))
        
        # Create regular lat/lon grid from subset
        subset_lat = subset.latitude.values
        subset_lon = subset.longitude.values
        
        # Fix longitude if needed
        if subset_lon.max() > 180:
            subset_lon = np.where(subset_lon > 180, subset_lon - 360, subset_lon)
        
        # Get approximate regular grid (use mean lat/lon for each row/column)
        lat_1d = np.mean(subset_lat, axis=1)
        lon_1d = np.mean(subset_lon, axis=0)
        
        print(f"  Approximate regular grid: lat={len(lat_1d)}, lon={len(lon_1d)}")
        
        # Process each variable
        processed_vars = {}
        for var_name in ['si10', 'r2', 't2m']:
            if var_name in subset.data_vars:
                print(f"  Processing {var_name}...")
                
                # Create dataset with 1D coordinates for interpolation
                var_data = subset[var_name]
                
                # Create new dataset with regular coordinates
                regular_ds = xr.Dataset(
                    {var_name: (['time', 'lat', 'lon'], var_data.values)},
                    coords={
                        'time': subset.time,
                        'lat': lat_1d,
                        'lon': lon_1d
                    }
                )
                
                # Interpolate to master grid
                # The result will already have 'latitude' and 'longitude' as dimension names
                interp = regular_ds.interp(
                    lat=master.latitude,
                    lon=master.longitude,
                    method='linear'
                )
                
                # Extract the variable (dimensions are already correct)
                processed_vars[var_name] = interp[var_name]
        
        if processed_vars:
            # Combine variables
            file_ds = xr.Dataset(processed_vars)
            all_data.append(file_ds)
            print(f"  ✓ Processed {len(processed_vars)} variables")
    
    if not all_data:
        print("\nERROR: No data processed!")
        return None
    
    # Concatenate all files
    print(f"\nConcatenating {len(all_data)} processed files...")
    combined = xr.concat(all_data, dim='time')
    
    # Sort by time
    combined = combined.sortby('time')
    
    # Check for duplicates
    time_vals = pd.to_datetime(combined.time.values)
    unique_times = np.unique(time_vals)
    if len(unique_times) < len(time_vals):
        print(f"  Removing {len(time_vals) - len(unique_times)} duplicate time steps...")
        _, unique_idx = np.unique(time_vals, return_index=True)
        combined = combined.isel(time=sorted(unique_idx))
    
    # Add attributes
    combined.attrs['title'] = 'UERRA High-Resolution Reanalysis - Portugal'
    combined.attrs['source'] = 'UERRA-HARMONIE'
    combined.attrs['processing'] = 'Regridded from ~5.5km curvilinear to 1km regular grid'
    
    # Save
    output_path = Path("data/01_processed/processed_uerra_1km.nc")
    print(f"\nSaving to: {output_path}")
    
    encoding = {var: {'zlib': True, 'complevel': 4} for var in combined.data_vars}
    combined.to_netcdf(output_path, encoding=encoding)
    
    file_size_gb = output_path.stat().st_size / (1024**3)
    print(f"  Saved! File size: {file_size_gb:.2f} GB")
    
    # Print summary
    print(f"\nFinal dataset:")
    print(f"  Variables: {list(combined.data_vars)}")
    print(f"  Time steps: {len(combined.time)}")
    print(f"  Grid: {len(combined.latitude)} x {len(combined.longitude)}")
    
    return output_path

def main():
    output_path = process_uerra_fast()
    
    if output_path and output_path.exists():
        print("\n" + "=" * 70)
        print("DELIVERABLE VERIFICATION")
        print("=" * 70)
        print("\nRun this command:")
        print(f'ncdump -h {output_path} | grep "float\\|double"')
        print("\nExpected: si10, r2, t2m")

if __name__ == "__main__":
    main()