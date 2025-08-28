#!/usr/bin/env python3
"""
Complete data aggregation with proper time alignment and chunking
"""

import sys
from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.data_processing import (
    load_config,
    save_dataset
)

def process_atmospheric_variables(atmospheric_path, master_time):
    """Process atmospheric data with proper time alignment"""
    atmospheric_path = Path(atmospheric_path)
    
    # Process temperature data
    temp_files = sorted(atmospheric_path.glob("era5_daily_max_temp_*.nc"))
    print(f"  Found {len(temp_files)} temperature files")
    
    if temp_files:
        all_temps = []
        for i, file in enumerate(temp_files, 1):
            if i % 10 == 0:
                print(f"    Processing temperature file {i}/{len(temp_files)}...")
            try:
                ds = xr.open_dataset(file)
                all_temps.append(ds)
            except Exception as e:
                print(f"    Error loading {file.name}: {e}")
        
        if all_temps:
            print("  Concatenating all temperature data...")
            temp_combined = xr.concat(all_temps, dim='time')
            
            # Align with master time
            print("  Aligning time dimension...")
            temp_aligned = temp_combined.reindex(time=master_time, method='nearest')
            return temp_aligned
    
    return None

def process_era5_land(land_path, master_time):
    """Process ERA5-Land data with chunking"""
    land_path = Path(land_path)
    land_files = sorted(land_path.glob("era5_land_may_nov_*.nc"))
    
    print(f"  Found {len(land_files)} ERA5-Land files")
    
    if land_files:
        all_land = []
        for i, file in enumerate(land_files, 1):
            print(f"    Processing ERA5-Land file {i}/{len(land_files)}...")
            try:
                # Check if it's actually a NetCDF file
                import subprocess
                result = subprocess.run(['file', str(file)], capture_output=True, text=True)
                if 'NetCDF' in result.stdout:
                    ds = xr.open_dataset(file, chunks={'time': 100})
                    all_land.append(ds)
                else:
                    print(f"      Skipping {file.name} - not a NetCDF file")
            except Exception as e:
                print(f"    Error loading {file.name}: {e}")
        
        if all_land:
            print("  Concatenating ERA5-Land data...")
            land_combined = xr.concat(all_land, dim='time')
            
            # Align with master time
            print("  Aligning time dimension...")
            land_aligned = land_combined.reindex(time=master_time, method='nearest')
            return land_aligned
    
    return None

def process_wind_variables(atmospheric_path, master_time):
    """Process wind component files"""
    atmospheric_path = Path(atmospheric_path)
    
    # Wind component files
    u_wind = atmospheric_path / "10m_u_component_of_wind_stream-oper_daily-mean.nc"
    v_wind = atmospheric_path / "10m_v_component_of_wind_0_daily-mean.nc"
    dewpoint = atmospheric_path / "2m_dewpoint_temperature_0_daily-mean.nc"
    
    wind_data = []
    
    for file in [u_wind, v_wind, dewpoint]:
        if file.exists():
            print(f"  Loading {file.name}...")
            try:
                ds = xr.open_dataset(file)
                # Align time if needed
                if 'time' in ds.dims:
                    ds = ds.reindex(time=master_time, method='nearest')
                wind_data.append(ds)
            except Exception as e:
                print(f"    Error: {e}")
    
    if wind_data:
        return xr.merge(wind_data)
    
    return None

def main():
    print("="*70)
    print("COMPLETE DATA AGGREGATION WITH TIME ALIGNMENT")
    print("="*70)
    print(f"Started: {datetime.now()}")
    
    # Load existing unified dataset with FWI
    print("\n📂 Loading existing unified dataset...")
    config = load_config()
    unified_path = Path(config['data']['data_paths']['unified_dataset'])
    
    if not unified_path.exists():
        print("❌ Unified dataset not found. Run preprocessing first!")
        return
    
    unified = xr.open_dataset(unified_path)
    print(f"  Loaded: {list(unified.data_vars)}")
    print(f"  Time steps: {len(unified.time)}")
    
    # Get master time dimension
    master_time = unified.time
    
    # Process atmospheric data
    print("\n☁️ Processing atmospheric variables...")
    atmospheric_path = Path(config['data']['data_paths'].get('raw_era5_atmospheric', ''))
    
    if atmospheric_path.exists():
        # Temperature data
        temp_data = process_atmospheric_variables(atmospheric_path, master_time)
        if temp_data is not None:
            print("  Regridding temperature to 1km...")
            temp_1km = temp_data.interp(
                latitude=unified.latitude,
                longitude=unified.longitude,
                method='linear'
            )
            
            # Add to unified dataset
            for var in temp_1km.data_vars:
                if var not in unified.data_vars:
                    print(f"    Adding {var} to unified dataset")
                    unified[f'temp_{var}'] = temp_1km[var]
        
        # Wind and dewpoint data
        wind_data = process_wind_variables(atmospheric_path, master_time)
        if wind_data is not None:
            print("  Regridding wind variables to 1km...")
            wind_1km = wind_data.interp(
                latitude=unified.latitude,
                longitude=unified.longitude,
                method='linear'
            )
            
            for var in wind_1km.data_vars:
                if var not in unified.data_vars:
                    print(f"    Adding {var} to unified dataset")
                    unified[var] = wind_1km[var]
    
    # Process ERA5-Land
    print("\n🏔️ Processing ERA5-Land data...")
    land_path = Path(config['data']['data_paths'].get('raw_era5_land', ''))
    
    if land_path.exists():
        land_data = process_era5_land(land_path, master_time)
        if land_data is not None:
            print("  Regridding ERA5-Land to 1km...")
            land_1km = land_data.interp(
                latitude=unified.latitude,
                longitude=unified.longitude,
                method='linear'
            )
            
            for var in land_1km.data_vars:
                if var not in unified.data_vars:
                    print(f"    Adding {var} to unified dataset")
                    unified[f'land_{var}'] = land_1km[var]
    
    # Add metadata
    unified.attrs.update({
        'title': 'FWI Unified Dataset with All Variables - Portugal 2010-2017',
        'last_updated': str(datetime.now()),
        'processing': 'Time-aligned and regridded to 1km'
    })
    
    # Save enhanced dataset
    output_path = unified_path.parent / "unified_fwi_all_variables_2010_2017.nc"
    print(f"\n💾 Saving enhanced dataset to {output_path}...")
    
    # Use compression to reduce file size
    encoding = {var: {'zlib': True, 'complevel': 4} for var in unified.data_vars}
    unified.to_netcdf(output_path, encoding=encoding)
    
    print(f"\n✅ AGGREGATION COMPLETE!")
    print(f"  Variables: {list(unified.data_vars)}")
    print(f"  Dimensions: {dict(unified.dims)}")
    print(f"  File size: {output_path.stat().st_size / 1e9:.2f} GB")
    print(f"  Completed: {datetime.now()}")
    
    return unified

if __name__ == "__main__":
    dataset = main()