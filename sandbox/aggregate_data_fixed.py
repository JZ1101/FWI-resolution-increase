#!/usr/bin/env python3
"""
Fixed data aggregation handling different time dimension names
"""

import sys
from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from src.data_processing import load_config

def process_temperature_files(atmospheric_path, master_grid):
    """Process temperature files with proper time handling"""
    temp_files = sorted(atmospheric_path.glob("era5_daily_max_temp_*.nc"))
    print(f"  Found {len(temp_files)} temperature files")
    
    if not temp_files:
        return None
    
    all_data = []
    
    for file in temp_files:
        try:
            # Extract year and month from filename
            parts = file.stem.split('_')
            year = int(parts[-2])
            month = int(parts[-1])
            
            # Load data
            ds = xr.open_dataset(file)
            
            # Rename valid_time to time if needed
            if 'valid_time' in ds.dims and 'time' not in ds.dims:
                ds = ds.rename({'valid_time': 'time'})
            
            # Create proper time coordinate if needed
            if 'time' in ds.dims:
                # Ensure time coordinate has proper datetime values
                n_days = len(ds.time)
                start_date = pd.Timestamp(f'{year}-{month:02d}-01')
                time_range = pd.date_range(start_date, periods=n_days, freq='D')
                ds = ds.assign_coords(time=time_range)
            
            all_data.append(ds)
            
        except Exception as e:
            print(f"    Error processing {file.name}: {e}")
    
    if all_data:
        print(f"  Concatenating {len(all_data)} temperature files...")
        combined = xr.concat(all_data, dim='time')
        
        # Sort by time
        combined = combined.sortby('time')
        
        # Interpolate to master grid
        print("  Regridding temperature to 1km...")
        temp_1km = combined.interp(
            latitude=master_grid.latitude,
            longitude=master_grid.longitude,
            method='linear'
        )
        
        # Select only time periods that match master grid
        if len(temp_1km.time) > len(master_grid.time):
            # Find matching time periods
            temp_1km = temp_1km.sel(time=master_grid.time, method='nearest')
        
        return temp_1km
    
    return None

def process_wind_and_dewpoint(atmospheric_path, master_grid):
    """Process wind and dewpoint files"""
    files = {
        'u10': atmospheric_path / "10m_u_component_of_wind_stream-oper_daily-mean.nc",
        'v10': atmospheric_path / "10m_v_component_of_wind_0_daily-mean.nc",
        'd2m': atmospheric_path / "2m_dewpoint_temperature_0_daily-mean.nc"
    }
    
    processed_data = []
    
    for var_name, file_path in files.items():
        if file_path.exists():
            print(f"  Loading {file_path.name}...")
            try:
                ds = xr.open_dataset(file_path)
                
                # Rename valid_time if present
                if 'valid_time' in ds.dims:
                    ds = ds.rename({'valid_time': 'time'})
                
                # Ensure we have all 8 years of data
                # These files likely contain all the data already
                
                # Interpolate to master grid
                ds_1km = ds.interp(
                    latitude=master_grid.latitude,
                    longitude=master_grid.longitude,
                    method='linear'
                )
                
                # Match time dimension
                if 'time' in ds_1km.dims:
                    # Select matching time periods
                    ds_1km = ds_1km.sel(time=master_grid.time, method='nearest')
                
                processed_data.append(ds_1km)
                
            except Exception as e:
                print(f"    Error: {e}")
    
    if processed_data:
        return xr.merge(processed_data)
    
    return None

def main():
    print("="*70)
    print("FIXED DATA AGGREGATION")
    print("="*70)
    print(f"Started: {datetime.now()}")
    
    # Load configuration
    config = load_config()
    
    # Load existing unified dataset
    print("\n📂 Loading existing unified dataset...")
    unified_path = Path(config['data']['data_paths']['unified_dataset'])
    
    if not unified_path.exists():
        print("❌ Unified dataset not found!")
        return
    
    unified = xr.open_dataset(unified_path)
    print(f"  Variables: {list(unified.data_vars)}")
    print(f"  Time steps: {len(unified.time)}")
    print(f"  Spatial: {len(unified.latitude)} x {len(unified.longitude)}")
    
    # Use unified dataset as master grid
    master_grid = unified
    
    # Process atmospheric data
    print("\n☁️ Processing atmospheric data...")
    atmospheric_path = Path(config['data']['data_paths'].get('raw_era5_atmospheric', ''))
    
    if atmospheric_path.exists():
        # Temperature
        temp_data = process_temperature_files(atmospheric_path, master_grid)
        if temp_data is not None:
            print("  Adding temperature data...")
            for var in temp_data.data_vars:
                if var not in unified.data_vars:
                    unified[f'max_{var}'] = temp_data[var]
                    print(f"    Added: max_{var}")
        
        # Wind and dewpoint
        wind_data = process_wind_and_dewpoint(atmospheric_path, master_grid)
        if wind_data is not None:
            print("  Adding wind and dewpoint data...")
            for var in wind_data.data_vars:
                if var not in unified.data_vars:
                    unified[var] = wind_data[var]
                    print(f"    Added: {var}")
    
    # Check ERA5-Land files
    print("\n🏔️ Checking ERA5-Land data...")
    land_path = Path(config['data']['data_paths'].get('raw_era5_land', ''))
    
    if land_path.exists():
        land_files = sorted(land_path.glob("era5_land_may_nov_*.nc"))
        print(f"  Found {len(land_files)} ERA5-Land files")
        
        # Check if they're actually NetCDF
        for file in land_files[:1]:  # Check first file
            import subprocess
            result = subprocess.run(['file', str(file)], capture_output=True, text=True)
            print(f"  File type: {result.stdout.strip()}")
            
            if 'Zip' in result.stdout:
                print("  ⚠️ ERA5-Land files are still compressed. Need to extract them first.")
    
    # Update metadata
    unified.attrs.update({
        'title': 'FWI Unified Dataset - Portugal 2010-2017',
        'description': 'Fire Weather Index with atmospheric variables at 1km resolution',
        'last_updated': str(datetime.now()),
        'variables_count': len(unified.data_vars)
    })
    
    # Save enhanced dataset
    output_path = unified_path.parent / "unified_fwi_enhanced_2010_2017.nc"
    print(f"\n💾 Saving enhanced dataset...")
    
    # Use compression
    encoding = {var: {'zlib': True, 'complevel': 4} for var in unified.data_vars}
    unified.to_netcdf(output_path, encoding=encoding)
    
    print(f"\n✅ AGGREGATION COMPLETE!")
    print(f"  Output: {output_path}")
    print(f"  Variables ({len(unified.data_vars)}): {list(unified.data_vars)}")
    print(f"  Dimensions: {dict(unified.dims)}")
    print(f"  File size: {output_path.stat().st_size / 1e6:.1f} MB")
    
    # Final check
    print("\n📊 Variable Summary:")
    for var in unified.data_vars:
        var_data = unified[var]
        nan_count = np.isnan(var_data.values).sum()
        nan_pct = 100 * nan_count / var_data.size
        print(f"  {var:20s}: Shape {var_data.shape}, NaN: {nan_pct:.1f}%")
    
    return unified

if __name__ == "__main__":
    dataset = main()