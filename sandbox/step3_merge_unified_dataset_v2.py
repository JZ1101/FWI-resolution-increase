#!/usr/bin/env python3
"""
Step 3: Merge all processed datasets into unified dataset
- Load existing FWI dataset with FWI and land_mask
- Load processed ERA5-Land data (5 variables)
- Load processed ERA5 Atmospheric data (5 variables)
- Merge all into single unified dataset
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def merge_datasets():
    """Merge all processed datasets into unified dataset"""
    print("=== STEP 3: MERGE ALL DATASETS INTO UNIFIED DATASET ===\n")
    
    # Define input paths
    fwi_path = Path("data/02_final_unified/unified_fwi_dataset_2010_2017.nc")
    land_path = Path("data/01_processed/processed_era5_land_1km.nc")
    atmospheric_path = Path("data/01_processed/processed_era5_atmospheric_1km.nc")
    
    # Check all files exist
    print("Checking input files...")
    files_exist = True
    for path, name in [(fwi_path, "FWI"), (land_path, "ERA5-Land"), (atmospheric_path, "ERA5 Atmospheric")]:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {name}: {path} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {name}: {path} NOT FOUND")
            files_exist = False
    
    if not files_exist:
        print("\nERROR: Not all input files exist. Cannot proceed with merge.")
        return None
    
    print("\n" + "="*60)
    print("Loading datasets...")
    print("="*60)
    
    # Load FWI dataset (existing unified dataset)
    print("\n1. Loading FWI dataset...")
    fwi_ds = xr.open_dataset(fwi_path)
    print(f"   Variables: {list(fwi_ds.data_vars)}")
    print(f"   Dimensions: {dict(fwi_ds.dims)}")
    print(f"   Time range: {fwi_ds.time.min().values} to {fwi_ds.time.max().values}")
    
    # Load ERA5-Land processed data
    print("\n2. Loading ERA5-Land dataset...")
    land_ds = xr.open_dataset(land_path)
    print(f"   Variables: {list(land_ds.data_vars)}")
    print(f"   Dimensions: {dict(land_ds.dims)}")
    if 'time' in land_ds.dims:
        print(f"   Time range: {land_ds.time.min().values} to {land_ds.time.max().values}")
    
    # Load ERA5 Atmospheric processed data
    print("\n3. Loading ERA5 Atmospheric dataset...")
    atm_ds = xr.open_dataset(atmospheric_path)
    print(f"   Variables: {list(atm_ds.data_vars)}")
    print(f"   Dimensions: {dict(atm_ds.dims)}")
    if 'time' in atm_ds.dims:
        print(f"   Time range: {atm_ds.time.min().values} to {atm_ds.time.max().values}")
    
    print("\n" + "="*60)
    print("Merging datasets...")
    print("="*60)
    
    # Start with FWI dataset as base
    merged = fwi_ds.copy()
    print(f"\nBase dataset variables: {list(merged.data_vars)}")
    
    # Handle potential variable naming conflicts
    # ERA5-Land and Atmospheric both have t2m, d2m, u10, v10, tp
    # We'll prefix them to distinguish the sources
    
    print("\nAdding ERA5-Land variables...")
    for var in land_ds.data_vars:
        if var not in ['latitude', 'longitude', 'time']:
            # Prefix with 'land_' to distinguish from atmospheric
            new_var_name = f"land_{var}"
            print(f"  Adding {var} as {new_var_name}")
            
            # Ensure time alignment
            if 'time' in land_ds[var].dims:
                # Align to FWI time grid
                aligned_var = land_ds[var].reindex(
                    time=merged.time,
                    method='nearest',
                    tolerance='1D'
                )
                merged[new_var_name] = aligned_var
            else:
                merged[new_var_name] = land_ds[var]
    
    print("\nAdding ERA5 Atmospheric variables...")
    for var in atm_ds.data_vars:
        if var not in ['latitude', 'longitude', 'time', 'number']:
            # Prefix with 'atm_' to distinguish from land
            new_var_name = f"atm_{var}"
            print(f"  Adding {var} as {new_var_name}")
            
            # Ensure time alignment
            if 'time' in atm_ds[var].dims:
                # Align to FWI time grid
                aligned_var = atm_ds[var].reindex(
                    time=merged.time,
                    method='nearest',
                    tolerance='1D'
                )
                merged[new_var_name] = aligned_var
            else:
                merged[new_var_name] = atm_ds[var]
    
    # Update attributes
    merged.attrs['title'] = 'Unified Portugal FWI Dataset 2010-2017 (Complete)'
    merged.attrs['description'] = 'Merged dataset containing FWI, ERA5-Land, and ERA5 Atmospheric variables'
    merged.attrs['creation_date'] = pd.Timestamp.now().isoformat()
    merged.attrs['sources'] = 'FWI reanalysis, ERA5-Land, ERA5 Atmospheric'
    merged.attrs['processing'] = 'All variables regridded to 1km resolution and merged'
    merged.attrs['variables_included'] = ', '.join(sorted(merged.data_vars))
    
    print("\n" + "="*60)
    print("Final merged dataset summary:")
    print("="*60)
    print(f"  Total variables: {len(merged.data_vars)}")
    print(f"  Variable list:")
    for var in sorted(merged.data_vars):
        print(f"    - {var}")
    print(f"  Dimensions: {dict(merged.dims)}")
    print(f"  Grid: {len(merged.latitude)} × {len(merged.longitude)} (1km resolution)")
    print(f"  Time steps: {len(merged.time)}")
    print(f"  Time range: {merged.time.min().values} to {merged.time.max().values}")
    
    # Calculate estimated size
    size_gb = merged.nbytes / (1024**3)
    print(f"  Estimated size: {size_gb:.2f} GB")
    
    # Save merged dataset to NEW file
    output_path = Path("data/02_final_unified/unified_complete_2010_2017.nc")
    print(f"\nSaving merged dataset to: {output_path}")
    
    # Save with compression
    encoding = {}
    for var in merged.data_vars:
        if var not in ['latitude', 'longitude', 'time']:
            encoding[var] = {'zlib': True, 'complevel': 4}
    
    print("  Saving with compression...")
    merged.to_netcdf(output_path, encoding=encoding)
    
    # Verify saved file
    actual_size_gb = output_path.stat().st_size / (1024**3)
    print(f"  ✓ Saved successfully! Final file size: {actual_size_gb:.2f} GB")
    
    # Close datasets to free memory
    fwi_ds.close()
    land_ds.close()
    atm_ds.close()
    
    return output_path

def main():
    """Main workflow for Step 3"""
    print("STEP 3: MERGE AND VERIFY FINAL DATASET\n")
    
    output_path = merge_datasets()
    
    if output_path and output_path.exists():
        print("\n" + "="*60)
        print("=== DELIVERABLE VERIFICATION ===")
        print("="*60)
        print("\nRun this command to verify ALL variables are present:")
        print(f'ncdump -h {output_path} | grep "float\\|double"')
        print("\nExpected variables:")
        print("  - fwi (Fire Weather Index)")
        print("  - land_mask")
        print("  - land_* variables (5 from ERA5-Land)")
        print("  - atm_* variables (5 from ERA5 Atmospheric)")

if __name__ == "__main__":
    main()