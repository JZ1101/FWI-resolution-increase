#!/usr/bin/env python3
"""
Step 1: Safely extract and process ALL ERA5-Land variables
- Extract ZIP archives to unique filenames to prevent overwriting
- Process ALL variables (not just one)
- Concatenate along time dimension
- Regrid to 1km master grid
- Save to processed directory
"""

import xarray as xr
import numpy as np
from pathlib import Path
import zipfile
import warnings
warnings.filterwarnings('ignore')

def safely_extract_zip_files():
    """Safely extract ERA5-Land ZIP files to unique names"""
    print("=== SAFELY EXTRACTING ERA5-LAND ZIP FILES ===\n")
    
    # Define paths
    source_dir = Path("data/00_raw/portugal_2010_2017/ERA5_reanalysis_ 10km_atmospheric_land-surface_parameters")
    intermediate_dir = Path("data/01_intermediate/era5_land_unzipped")
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    
    # List ZIP files (they have .nc extension but are actually ZIP)
    zip_files = sorted(source_dir.glob("era5_land_may_nov_*.nc"))
    
    print(f"Found {len(zip_files)} ERA5-Land ZIP files to extract:")
    extracted_files = []
    
    for zip_path in zip_files:
        # Extract year from filename
        year = zip_path.stem.split('_')[-1]  # e.g., "2010" from "era5_land_may_nov_2010"
        
        print(f"\nProcessing: {zip_path.name}")
        
        try:
            # Open ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # List contents
                contents = zf.namelist()
                print(f"  ZIP contains: {contents}")
                
                # Extract each file with unique name
                for member in contents:
                    # Create unique output filename
                    output_name = f"unzipped_{year}.nc"
                    output_path = intermediate_dir / output_name
                    
                    # Extract to unique filename
                    with zf.open(member) as source:
                        with open(output_path, 'wb') as target:
                            target.write(source.read())
                    
                    print(f"  Extracted to: {output_path}")
                    extracted_files.append(output_path)
                    
        except Exception as e:
            print(f"  ERROR extracting {zip_path.name}: {e}")
    
    print(f"\nSuccessfully extracted {len(extracted_files)} files")
    return extracted_files

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

def process_extracted_files(extracted_files):
    """Process the extracted NetCDF files - ALL variables"""
    print("\n=== PROCESSING ALL VARIABLES FROM EXTRACTED FILES ===\n")
    
    if not extracted_files:
        print("ERROR: No files to process!")
        return
    
    # Load master grid first for regridding
    print("Loading master grid...")
    master = load_master_grid()
    
    if master is None:
        print("ERROR: Cannot proceed without master grid")
        return
    
    print(f"  Master grid shape: lat={len(master.latitude)}, lon={len(master.longitude)}")
    
    # Dictionary to store all variables
    all_variables = {}
    variable_names = set()
    
    # First pass: identify all variables across all files
    print("\nIdentifying all variables in the dataset...")
    for file_path in sorted(extracted_files):
        with xr.open_dataset(file_path) as ds:
            # Get data variables (exclude coordinates)
            data_vars = [v for v in ds.data_vars 
                        if v not in ['latitude', 'longitude', 'time', 'lat', 'lon', 
                                    'valid_time', 'expver', 'number', 'surface']]
            variable_names.update(data_vars)
            print(f"  {file_path.name}: {data_vars}")
    
    print(f"\nFound {len(variable_names)} unique variables to process: {sorted(variable_names)}")
    
    # Process each variable separately to handle memory efficiently
    for var_name in sorted(variable_names):
        print(f"\n{'='*60}")
        print(f"Processing variable: {var_name}")
        print(f"{'='*60}")
        
        var_datasets = []
        
        # Load this variable from all files
        for file_path in sorted(extracted_files):
            try:
                ds = xr.open_dataset(file_path)
                
                # Rename valid_time to time if needed
                if 'valid_time' in ds.dims and 'time' not in ds.dims:
                    ds = ds.rename({'valid_time': 'time'})
                
                # Check if this file has the current variable
                if var_name in ds.data_vars:
                    # Extract just this variable
                    var_data = ds[var_name]
                    
                    # Create a dataset with just this variable and coordinates
                    var_ds = xr.Dataset({var_name: var_data})
                    var_datasets.append(var_ds)
                    print(f"  Loaded {var_name} from {file_path.name}: shape={dict(var_data.dims)}")
                
            except Exception as e:
                print(f"  ERROR loading {var_name} from {file_path.name}: {e}")
        
        if not var_datasets:
            print(f"  WARNING: No data found for {var_name}, skipping...")
            continue
        
        # Concatenate this variable across all files
        print(f"\n  Concatenating {var_name} across {len(var_datasets)} files...")
        try:
            var_combined = xr.concat(var_datasets, dim='time')
            print(f"    Combined shape: {dict(var_combined.dims)}")
            
            # Sort by time
            var_combined = var_combined.sortby('time')
            
            # Check for and remove duplicates
            time_index = var_combined.time.to_index()
            if time_index.duplicated().any():
                n_duplicates = time_index.duplicated().sum()
                print(f"    WARNING: Found {n_duplicates} duplicate timestamps")
                print(f"    Removing duplicates by keeping first occurrence...")
                var_combined = var_combined.sel(time=~time_index.duplicated())
                print(f"    Shape after deduplication: {dict(var_combined.dims)}")
            
            # Ensure consistent coordinate names for regridding
            if 'lat' in var_combined.dims:
                var_combined = var_combined.rename({'lat': 'latitude'})
            if 'lon' in var_combined.dims:
                var_combined = var_combined.rename({'lon': 'longitude'})
            
            # Regrid this variable to master grid
            print(f"  Regridding {var_name} to 1km master grid...")
            var_regridded = var_combined.interp(
                latitude=master.latitude,
                longitude=master.longitude,
                method='linear'
            )
            
            # Store the regridded variable
            all_variables[var_name] = var_regridded[var_name]
            print(f"  ✓ Successfully processed {var_name}: final shape={dict(var_regridded[var_name].dims)}")
            
        except Exception as e:
            print(f"  ERROR processing {var_name}: {e}")
            continue
    
    # Create final dataset with all variables
    if not all_variables:
        print("\nERROR: No variables were successfully processed!")
        return
    
    print(f"\n{'='*60}")
    print(f"Creating final dataset with {len(all_variables)} variables...")
    print(f"{'='*60}")
    
    # Combine all variables into one dataset
    final_dataset = xr.Dataset(all_variables)
    
    # Add attributes
    final_dataset.attrs['description'] = 'ERA5-Land data (ALL variables) regridded to 1km'
    final_dataset.attrs['source'] = 'ERA5-Land reanalysis'
    final_dataset.attrs['processing'] = 'Extracted, concatenated and regridded ALL variables to 1km master grid'
    final_dataset.attrs['variables_processed'] = ', '.join(sorted(all_variables.keys()))
    
    # Print summary
    print(f"\nFinal dataset summary:")
    print(f"  Variables: {list(final_dataset.data_vars)}")
    print(f"  Dimensions: {dict(final_dataset.dims)}")
    if 'time' in final_dataset.dims:
        print(f"  Time range: {final_dataset.time.min().values} to {final_dataset.time.max().values}")
    
    # Save processed file
    output_dir = Path("data/01_processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "processed_era5_land_1km.nc"
    
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
    print("=== STEP 1 COMPLETE ===")
    print(f"{'='*60}")
    print(f"Successfully processed ALL {len(all_variables)} variables from {len(extracted_files)} ERA5-Land files")
    print(f"Output: {output_path}")
    
    return output_path

def main():
    """Main workflow for Step 1"""
    print("=== STEP 1: SAFE EXTRACTION AND PROCESSING OF ALL ERA5-LAND VARIABLES ===\n")
    
    # Check if files are already extracted
    intermediate_dir = Path("data/01_intermediate/era5_land_unzipped")
    existing_extracted = list(intermediate_dir.glob("unzipped_*.nc")) if intermediate_dir.exists() else []
    
    if len(existing_extracted) == 8:
        print(f"Found {len(existing_extracted)} already extracted files, skipping extraction...")
        extracted_files = existing_extracted
    else:
        # Step 1a: Safely extract ZIP files
        extracted_files = safely_extract_zip_files()
    
    # Step 1b: Process ALL variables from extracted files
    if extracted_files:
        output_path = process_extracted_files(extracted_files)
        
        if output_path and output_path.exists():
            print("\n=== DELIVERABLE VERIFICATION ===")
            print("Run this command to verify ALL variables are present:")
            print(f'ncdump -h {output_path} | grep "float\\|double"')

if __name__ == "__main__":
    main()