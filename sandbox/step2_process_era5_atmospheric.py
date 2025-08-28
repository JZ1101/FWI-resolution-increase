#!/usr/bin/env python3
"""
Step 2: Process ERA5 Atmospheric 25km dataset
- Handle all 115 files
- Check for and remove duplicate time entries
- Regrid from 22x14 grid to 1km master grid
- Process all available variables generically
"""

import xarray as xr
import numpy as np
import pandas as pd
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

def process_atmospheric_data():
    """Process all ERA5 atmospheric files"""
    print("=== STEP 2: PROCESS ERA5 ATMOSPHERIC 25KM DATA ===\n")
    
    # Define paths
    source_dir = Path("data/00_raw/portugal_2010_2017/ERA5_reanalysis_25km_atmospheric_parameters")
    output_dir = Path("data/01_processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load master grid
    print("Loading master grid...")
    master = load_master_grid()
    if master is None:
        print("ERROR: Cannot proceed without master grid")
        return
    print(f"  Master grid shape: lat={len(master.latitude)}, lon={len(master.longitude)}")
    
    # Categorize files by variable type
    file_categories = {
        'u10': [],  # 10m u-component of wind
        'v10': [],  # 10m v-component of wind
        'd2m': [],  # 2m dewpoint temperature
        't2m': [],  # 2m temperature
        'tp': []    # total precipitation
    }
    
    # List all NetCDF files
    nc_files = list(source_dir.glob("*.nc"))
    print(f"\nFound {len(nc_files)} NetCDF files in total")
    
    # Categorize files
    for file_path in nc_files:
        fname = file_path.name.lower()
        
        if '10m_u' in fname or 'u_component' in fname:
            file_categories['u10'].append(file_path)
        elif '10m_v' in fname or 'v_component' in fname:
            file_categories['v10'].append(file_path)
        elif 'dewpoint' in fname or 'd2m' in fname:
            file_categories['d2m'].append(file_path)
        elif 'temp' in fname and 'dewpoint' not in fname:
            file_categories['t2m'].append(file_path)
        elif 'precipitation' in fname or 'precip' in fname:
            file_categories['tp'].append(file_path)
    
    # Print categorization summary
    print("\nFile categorization:")
    for var_name, files in file_categories.items():
        print(f"  {var_name}: {len(files)} files")
    
    # Dictionary to store processed variables
    all_variables = {}
    
    # Process each variable category
    for var_name, file_list in file_categories.items():
        if not file_list:
            print(f"\n[WARNING] No files found for {var_name}")
            continue
            
        print(f"\n{'='*60}")
        print(f"Processing {var_name} ({len(file_list)} files)")
        print(f"{'='*60}")
        
        # Load all files for this variable
        datasets = []
        for file_path in sorted(file_list):
            try:
                print(f"  Loading: {file_path.name}")
                ds = xr.open_dataset(file_path)
                
                # Handle coordinate names
                if 'valid_time' in ds.dims and 'time' not in ds.dims:
                    ds = ds.rename({'valid_time': 'time'})
                
                # Print dataset info
                data_vars = list(ds.data_vars)
                print(f"    Variables: {data_vars}")
                print(f"    Dimensions: {dict(ds.dims)}")
                
                datasets.append(ds)
                
            except Exception as e:
                print(f"    ERROR loading {file_path.name}: {e}")
        
        if not datasets:
            print(f"  [ERROR] No datasets loaded for {var_name}")
            continue
        
        # Concatenate all datasets for this variable
        print(f"\n  Concatenating {len(datasets)} datasets...")
        try:
            # Concatenate along time dimension
            combined = xr.concat(datasets, dim='time')
            print(f"    Initial shape: {dict(combined.dims)}")
            
            # Sort by time
            combined = combined.sortby('time')
            
            # Check for and remove duplicate times
            print("  Checking for duplicate timestamps...")
            time_index = pd.to_datetime(combined.time.values)
            duplicated = time_index.duplicated(keep='first')
            n_duplicates = duplicated.sum()
            
            if n_duplicates > 0:
                print(f"    WARNING: Found {n_duplicates} duplicate timestamps")
                print(f"    Removing duplicates (keeping first occurrence)...")
                combined = combined.isel(time=~duplicated)
                print(f"    Shape after deduplication: {dict(combined.dims)}")
            else:
                print(f"    No duplicates found")
            
            # Get the actual variable name(s) from the dataset
            actual_vars = [v for v in combined.data_vars 
                          if v not in ['latitude', 'longitude', 'time', 'lat', 'lon', 
                                      'expver', 'number', 'surface']]
            
            if not actual_vars:
                print(f"  [ERROR] No data variables found in combined dataset")
                continue
            
            # Process each actual variable
            for actual_var in actual_vars:
                print(f"\n  Processing variable '{actual_var}'...")
                
                # Extract the variable
                var_data = combined[actual_var]
                
                # Ensure consistent coordinate names
                if 'lat' in var_data.dims:
                    var_data = var_data.rename({'lat': 'latitude'})
                if 'lon' in var_data.dims:
                    var_data = var_data.rename({'lon': 'longitude'})
                
                print(f"    Original grid shape: lat={len(var_data.latitude)}, lon={len(var_data.longitude)}")
                
                # Regrid to master grid
                print(f"    Regridding to 1km master grid...")
                var_regridded = var_data.interp(
                    latitude=master.latitude,
                    longitude=master.longitude,
                    method='linear'
                )
                
                # Store with standardized name
                standardized_name = var_name if var_name != actual_var else actual_var
                all_variables[standardized_name] = var_regridded
                print(f"    ✓ Successfully processed '{actual_var}' as '{standardized_name}'")
                print(f"    Final shape: {dict(var_regridded.dims)}")
            
        except Exception as e:
            print(f"  [ERROR] Failed to process {var_name}: {e}")
            continue
    
    # Create final dataset
    if not all_variables:
        print("\n[ERROR] No variables were successfully processed!")
        return
    
    print(f"\n{'='*60}")
    print(f"Creating final dataset with {len(all_variables)} variables...")
    print(f"{'='*60}")
    
    # Combine all variables
    final_dataset = xr.Dataset(all_variables)
    
    # Add attributes
    final_dataset.attrs['description'] = 'ERA5 Atmospheric 25km data regridded to 1km'
    final_dataset.attrs['source'] = 'ERA5 atmospheric reanalysis'
    final_dataset.attrs['processing'] = 'Concatenated, deduplicated, and regridded to 1km master grid'
    final_dataset.attrs['variables_processed'] = ', '.join(sorted(all_variables.keys()))
    final_dataset.attrs['original_grid'] = '22x14 (approximately)'
    final_dataset.attrs['target_grid'] = '701x401 (1km resolution)'
    
    # Print summary
    print(f"\nFinal dataset summary:")
    print(f"  Variables: {list(final_dataset.data_vars)}")
    print(f"  Dimensions: {dict(final_dataset.dims)}")
    if 'time' in final_dataset.dims:
        print(f"  Time range: {final_dataset.time.min().values} to {final_dataset.time.max().values}")
        print(f"  Time steps: {len(final_dataset.time)}")
    
    # Save processed file
    output_path = output_dir / "processed_era5_atmospheric_1km.nc"
    print(f"\nSaving to: {output_path}")
    
    # Encode with compression
    encoding = {}
    for var in final_dataset.data_vars:
        encoding[var] = {'zlib': True, 'complevel': 4}
    
    final_dataset.to_netcdf(output_path, encoding=encoding)
    print("  ✓ Saved successfully!")
    
    # Verify file
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")
    
    print(f"\n{'='*60}")
    print("=== STEP 2 COMPLETE ===")
    print(f"{'='*60}")
    print(f"Successfully processed {len(all_variables)} variables from {len(nc_files)} files")
    print(f"Output: {output_path}")
    
    return output_path

def main():
    """Main workflow for Step 2"""
    output_path = process_atmospheric_data()
    
    if output_path and output_path.exists():
        print("\n=== DELIVERABLE VERIFICATION ===")
        print("Run this command to verify the output:")
        print(f"ncdump -h {output_path}")

if __name__ == "__main__":
    main()