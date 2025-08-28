#!/usr/bin/env python3
"""
Complete data aggregation for Portugal FWI project (2010-2017)
Processes all available data sources into unified dataset
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_fwi_data():
    """Load FWI data from processed dataset"""
    print("Loading FWI data...")
    fwi_path = Path("data/02_final_unified/unified_fwi_dataset_2010_2017.nc")
    if fwi_path.exists():
        ds = xr.open_dataset(fwi_path)
        print(f"  FWI data loaded: {ds.time.min().values} to {ds.time.max().values}")
        print(f"  Shape: time={len(ds.time)}, lat={len(ds.latitude)}, lon={len(ds.longitude)}")
        return ds
    else:
        print("  ERROR: FWI data not found!")
        return None

def load_atmospheric_data():
    """Load ERA5 atmospheric data"""
    print("\nLoading ERA5 atmospheric data...")
    atm_path = Path("data/00_raw/portugal_2010_2017/ERA5_reanalysis_25km_atmospheric_parameters")
    
    if not atm_path.exists():
        print(f"  ERROR: Path does not exist: {atm_path}")
        return None
    
    datasets = {}
    
    # Load wind components
    wind_files = {
        'u10': '10m_u_component_of_wind_stream-oper_daily-mean.nc',
        'v10': '10m_v_component_of_wind_0_daily-mean.nc'
    }
    
    for var_name, filename in wind_files.items():
        filepath = atm_path / filename
        if filepath.exists():
            try:
                ds = xr.open_dataset(filepath)
                print(f"  Loaded {var_name}: {filepath.name}")
                datasets[var_name] = ds
            except Exception as e:
                print(f"  ERROR loading {filepath.name}: {e}")
    
    # Load dewpoint temperature
    dewpoint_file = atm_path / '2m_dewpoint_temperature_0_daily-mean.nc'
    if dewpoint_file.exists():
        try:
            ds = xr.open_dataset(dewpoint_file)
            print(f"  Loaded dewpoint: {dewpoint_file.name}")
            datasets['d2m'] = ds
        except Exception as e:
            print(f"  ERROR loading dewpoint: {e}")
    
    # Load temperature files (monthly)
    temp_files = sorted(atm_path.glob('era5_daily_max_temp_*.nc'))
    print(f"  Found {len(temp_files)} temperature files")
    
    temp_datasets = []
    for temp_file in temp_files:
        try:
            ds = xr.open_dataset(temp_file)
            temp_datasets.append(ds)
        except Exception as e:
            print(f"  ERROR loading {temp_file.name}: {e}")
    
    if temp_datasets:
        try:
            temp_combined = xr.concat(temp_datasets, dim='time')
            print(f"  Combined temperature data: {len(temp_combined.time)} timesteps")
            datasets['t2m'] = temp_combined
        except Exception as e:
            print(f"  ERROR combining temperature data: {e}")
    
    # Load precipitation files
    precip_files = sorted(atm_path.glob('precipitation_*.nc'))
    print(f"  Found {len(precip_files)} precipitation files")
    
    precip_datasets = []
    for precip_file in precip_files:
        try:
            ds = xr.open_dataset(precip_file)
            precip_datasets.append(ds)
        except Exception as e:
            print(f"  ERROR loading {precip_file.name}: {e}")
    
    if precip_datasets:
        try:
            precip_combined = xr.concat(precip_datasets, dim='time')
            print(f"  Combined precipitation data: {len(precip_combined.time)} timesteps")
            datasets['tp'] = precip_combined
        except Exception as e:
            print(f"  ERROR combining precipitation data: {e}")
    
    return datasets

def load_land_data():
    """Load ERA5-Land data"""
    print("\nLoading ERA5-Land data...")
    land_path = Path("data/00_raw/portugal_2010_2017/ERA5_reanalysis_ 10km_atmospheric_land-surface_parameters")
    
    if not land_path.exists():
        print(f"  ERROR: Path does not exist: {land_path}")
        return None
    
    # Check for data_0.nc
    land_file = land_path / "data_0.nc"
    if land_file.exists():
        try:
            ds = xr.open_dataset(land_file)
            print(f"  Loaded ERA5-Land data: {land_file.name}")
            print(f"  Variables: {list(ds.data_vars)}")
            return ds
        except Exception as e:
            print(f"  ERROR loading ERA5-Land: {e}")
    else:
        print(f"  ERROR: ERA5-Land file not found: {land_file}")
    
    return None

def align_to_master_grid(ds, master_lat, master_lon, master_time, var_name="data"):
    """Align dataset to master grid"""
    print(f"  Aligning {var_name} to master grid...")
    
    # Handle coordinate names
    if 'latitude' not in ds.dims and 'lat' in ds.dims:
        ds = ds.rename({'lat': 'latitude'})
    if 'longitude' not in ds.dims and 'lon' in ds.dims:
        ds = ds.rename({'lon': 'longitude'})
    
    # Handle time coordinate
    if 'valid_time' in ds.dims and 'time' not in ds.dims:
        ds = ds.rename({'valid_time': 'time'})
    
    # Ensure time is datetime64
    if ds.time.dtype != np.dtype('datetime64[ns]'):
        try:
            ds['time'] = pd.to_datetime(ds.time.values)
        except:
            print(f"    WARNING: Could not convert time to datetime for {var_name}")
    
    # Find common time range
    ds_time_min = pd.to_datetime(ds.time.min().values)
    ds_time_max = pd.to_datetime(ds.time.max().values)
    master_time_min = pd.to_datetime(master_time.min().values)
    master_time_max = pd.to_datetime(master_time.max().values)
    
    common_start = max(ds_time_min, master_time_min)
    common_end = min(ds_time_max, master_time_max)
    
    print(f"    Time range: {common_start} to {common_end}")
    
    # Select common time range
    ds = ds.sel(time=slice(common_start, common_end))
    
    # Interpolate to master grid
    try:
        ds_aligned = ds.interp(
            latitude=master_lat,
            longitude=master_lon,
            method='linear'
        )
        
        # Reindex time to match master time (within common range)
        master_time_common = master_time.sel(time=slice(common_start, common_end))
        ds_aligned = ds_aligned.reindex(
            time=master_time_common,
            method='nearest',
            tolerance='1D'
        )
        
        print(f"    Successfully aligned: {len(ds_aligned.time)} timesteps")
        return ds_aligned
        
    except Exception as e:
        print(f"    ERROR during alignment: {e}")
        return None

def main():
    """Main aggregation workflow"""
    print("=== COMPLETE DATA AGGREGATION FOR PORTUGAL FWI PROJECT ===\n")
    
    # Load FWI data as reference
    fwi_ds = load_fwi_data()
    if fwi_ds is None:
        print("Cannot proceed without FWI data")
        return
    
    # Get master grid dimensions
    master_lat = fwi_ds.latitude
    master_lon = fwi_ds.longitude
    master_time = fwi_ds.time
    
    # Initialize unified dataset with FWI
    unified_vars = {
        'fwi': fwi_ds.fwi,
        'latitude': master_lat,
        'longitude': master_lon,
        'time': master_time
    }
    
    # Process atmospheric data
    atm_datasets = load_atmospheric_data()
    if atm_datasets:
        for var_name, ds in atm_datasets.items():
            if ds is not None:
                print(f"\nProcessing {var_name}...")
                aligned = align_to_master_grid(ds, master_lat, master_lon, master_time, var_name)
                if aligned is not None:
                    # Get the actual variable name from the dataset
                    actual_vars = list(aligned.data_vars)
                    if actual_vars:
                        unified_vars[var_name] = aligned[actual_vars[0]]
                        print(f"  Added {var_name} to unified dataset")
    
    # Process land data
    land_ds = load_land_data()
    if land_ds is not None:
        print("\nProcessing ERA5-Land variables...")
        aligned = align_to_master_grid(land_ds, master_lat, master_lon, master_time, "ERA5-Land")
        if aligned is not None:
            for var in aligned.data_vars:
                unified_vars[f"land_{var}"] = aligned[var]
                print(f"  Added land_{var} to unified dataset")
    
    # Create unified dataset
    print("\n=== Creating Unified Dataset ===")
    unified = xr.Dataset(unified_vars)
    
    # Add attributes
    unified.attrs['title'] = 'Unified Portugal FWI Dataset 2010-2017'
    unified.attrs['description'] = 'Combined ERA5, ERA5-Land and FWI data for Portugal'
    unified.attrs['creation_date'] = pd.Timestamp.now().isoformat()
    unified.attrs['region'] = 'Portugal (36-43°N, -10 to -6°E)'
    
    # Print summary
    print(f"\nUnified dataset created:")
    print(f"  Variables: {list(unified.data_vars)}")
    print(f"  Time range: {unified.time.min().values} to {unified.time.max().values}")
    print(f"  Dimensions: time={len(unified.time)}, lat={len(unified.latitude)}, lon={len(unified.longitude)}")
    
    # Calculate size
    size_mb = unified.nbytes / (1024 * 1024)
    print(f"  Estimated size: {size_mb:.1f} MB")
    
    # Save dataset
    output_path = Path("data/02_final_unified/unified_complete_2010_2017.nc")
    print(f"\nSaving to {output_path}...")
    
    # Save with compression
    encoding = {}
    for var in unified.data_vars:
        if var not in ['latitude', 'longitude', 'time']:
            encoding[var] = {'zlib': True, 'complevel': 4}
    
    unified.to_netcdf(output_path, encoding=encoding)
    print(f"  Saved successfully!")
    
    # Verify saved file
    saved_size = output_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {saved_size:.1f} MB")
    
    # Final summary
    print("\n=== AGGREGATION COMPLETE ===")
    print(f"Successfully created unified dataset with {len(unified.data_vars)} variables")

if __name__ == "__main__":
    main()