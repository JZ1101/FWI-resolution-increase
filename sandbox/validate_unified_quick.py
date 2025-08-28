#!/usr/bin/env python3
"""
Quick validation of unified dataset - samples data to avoid memory issues
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def quick_validate():
    """Quick validation with sampling"""
    print("=" * 70)
    print("UNIFIED DATASET QUICK VALIDATION")
    print("=" * 70)
    
    # Find dataset
    file_path = Path("data/02_final_unified/unified_complete_2010_2017.nc")
    if not file_path.exists():
        file_path = Path("data/02_final_unified/unified_fwi_dataset_2010_2017.nc")
    
    if not file_path.exists():
        print("ERROR: No unified dataset found!")
        return
    
    print(f"\nDataset: {file_path}")
    file_size_gb = file_path.stat().st_size / (1024**3)
    print(f"File size: {file_size_gb:.2f} GB")
    
    # Open with chunks to avoid loading everything into memory
    print("\nOpening dataset with chunking...")
    ds = xr.open_dataset(file_path, chunks={'time': 100})
    
    # Basic info
    print(f"\nDimensions: {dict(ds.dims)}")
    print(f"Time range: {ds.time.min().values} to {ds.time.max().values}")
    print(f"\nVariables ({len(ds.data_vars)}):")
    for var in sorted(ds.data_vars):
        print(f"  - {var}")
    
    # Sample validation for FWI
    print("\n" + "=" * 70)
    print("FWI SAMPLE VALIDATION")
    print("=" * 70)
    
    if 'fwi' in ds.data_vars:
        # Check first time step
        fwi_sample = ds['fwi'].isel(time=0).compute()
        valid_pct = (~np.isnan(fwi_sample.values)).mean() * 100
        print(f"First day coverage: {valid_pct:.1f}%")
        
        # Check middle time step
        mid_idx = len(ds.time) // 2
        fwi_sample = ds['fwi'].isel(time=mid_idx).compute()
        valid_pct = (~np.isnan(fwi_sample.values)).mean() * 100
        print(f"Middle day coverage: {valid_pct:.1f}%")
        
        # Check last time step
        fwi_sample = ds['fwi'].isel(time=-1).compute()
        valid_pct = (~np.isnan(fwi_sample.values)).mean() * 100
        print(f"Last day coverage: {valid_pct:.1f}%")
        
        # Sample statistics (using every 100th time step)
        print("\nFWI statistics (sampled):")
        sample_indices = range(0, len(ds.time), 100)
        fwi_samples = ds['fwi'].isel(time=sample_indices).compute()
        valid_data = fwi_samples.values[~np.isnan(fwi_samples.values)]
        if len(valid_data) > 0:
            print(f"  Min: {np.min(valid_data):.2f}")
            print(f"  Max: {np.max(valid_data):.2f}")
            print(f"  Mean: {np.mean(valid_data):.2f}")
            print(f"  Std: {np.std(valid_data):.2f}")
    
    # Check ERA5-Land variables
    print("\n" + "=" * 70)
    print("ERA5-LAND VARIABLES CHECK")
    print("=" * 70)
    land_vars = [v for v in ds.data_vars if v.startswith('land_')]
    print(f"Found {len(land_vars)} ERA5-Land variables:")
    for var in sorted(land_vars):
        # Check if variable has time dimension
        if 'time' in ds[var].dims:
            # Sample first time step
            sample = ds[var].isel(time=0).compute()
            valid_pct = (~np.isnan(sample.values)).mean() * 100
            print(f"  {var}: {valid_pct:.1f}% coverage (first day)")
        else:
            # Static variable (like land_mask)
            sample = ds[var].compute()
            valid_pct = (~np.isnan(sample.values)).mean() * 100
            print(f"  {var}: {valid_pct:.1f}% coverage (static)")
    
    # Check ERA5 Atmospheric variables
    print("\n" + "=" * 70)
    print("ERA5 ATMOSPHERIC VARIABLES CHECK")
    print("=" * 70)
    atm_vars = [v for v in ds.data_vars if v.startswith('atm_')]
    print(f"Found {len(atm_vars)} ERA5 Atmospheric variables:")
    for var in sorted(atm_vars):
        # Check if variable has time dimension
        if 'time' in ds[var].dims:
            # Sample first time step
            sample = ds[var].isel(time=0).compute()
            valid_pct = (~np.isnan(sample.values)).mean() * 100
            print(f"  {var}: {valid_pct:.1f}% coverage (first day)")
        else:
            # Static variable (shouldn't happen for atm_ vars)
            sample = ds[var].compute()
            valid_pct = (~np.isnan(sample.values)).mean() * 100
            print(f"  {var}: {valid_pct:.1f}% coverage (static)")
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"✓ Dataset exists and is readable")
    print(f"✓ Contains {len(ds.data_vars)} variables total")
    print(f"✓ FWI + land_mask: 2 variables")
    print(f"✓ ERA5-Land variables: {len(land_vars)}")
    print(f"✓ ERA5 Atmospheric variables: {len(atm_vars)}")
    print(f"✓ Total expected: 12 variables")
    
    if len(ds.data_vars) == 12:
        print("\n✅ All expected variables are present!")
    else:
        print(f"\n⚠️ Expected 12 variables, found {len(ds.data_vars)}")
    
    print("\n" + "=" * 70)
    print("QUICK VALIDATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    quick_validate()