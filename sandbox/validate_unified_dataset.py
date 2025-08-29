#!/usr/bin/env python3
"""
Validate the unified dataset for quality and completeness
- Check for missing values
- Calculate basic statistics
- Verify temporal continuity
- Check spatial coverage
- Generate validation report
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def validate_unified_dataset(file_path):
    """Comprehensive validation of unified dataset"""
    print("=" * 70)
    print("UNIFIED DATASET VALIDATION REPORT")
    print("=" * 70)
    
    # Load dataset
    print(f"\nLoading dataset: {file_path}")
    ds = xr.open_dataset(file_path)
    file_size_gb = file_path.stat().st_size / (1024**3)
    print(f"File size: {file_size_gb:.2f} GB")
    
    # Basic information
    print("\n" + "=" * 70)
    print("DATASET STRUCTURE")
    print("=" * 70)
    print(f"Dimensions: {dict(ds.dims)}")
    print(f"Grid resolution: {len(ds.latitude)} x {len(ds.longitude)} (1km)")
    print(f"Time range: {ds.time.min().values} to {ds.time.max().values}")
    print(f"Time steps: {len(ds.time)}")
    
    # List all variables
    print(f"\nVariables ({len(ds.data_vars)} total):")
    for var in sorted(ds.data_vars):
        shape = ds[var].shape
        dtype = ds[var].dtype
        print(f"  - {var:15} shape={shape}, dtype={dtype}")
    
    # Temporal analysis
    print("\n" + "=" * 70)
    print("TEMPORAL ANALYSIS")
    print("=" * 70)
    time_diff = pd.to_datetime(ds.time.values[1:]) - pd.to_datetime(ds.time.values[:-1])
    unique_deltas = np.unique(time_diff)
    print(f"Time step frequency: {len(unique_deltas)} unique intervals")
    for delta in unique_deltas[:5]:  # Show first 5 unique intervals
        count = (time_diff == delta).sum()
        print(f"  - {delta}: {count} occurrences")
    
    # Check for gaps
    expected_days = pd.date_range(
        start=pd.to_datetime(ds.time.min().values),
        end=pd.to_datetime(ds.time.max().values),
        freq='D'
    )
    actual_days = pd.to_datetime(ds.time.values)
    missing_days = set(expected_days) - set(actual_days)
    print(f"\nTemporal gaps: {len(missing_days)} missing days")
    if missing_days and len(missing_days) <= 10:
        for day in sorted(list(missing_days))[:10]:
            print(f"  - {day.strftime('%Y-%m-%d')}")
    
    # Variable statistics
    print("\n" + "=" * 70)
    print("VARIABLE STATISTICS")
    print("=" * 70)
    
    for var in sorted(ds.data_vars):
        if var in ['latitude', 'longitude', 'time', 'surface']:
            continue
            
        print(f"\n{var}:")
        data = ds[var]
        
        # Handle different shapes
        if 'time' in data.dims:
            # Time-varying variable
            valid_mask = ~np.isnan(data.values)
            valid_count = valid_mask.sum()
            total_count = data.size
            valid_pct = (valid_count / total_count) * 100
            
            if valid_count > 0:
                valid_data = data.values[valid_mask]
                stats = {
                    'min': np.min(valid_data),
                    'max': np.max(valid_data),
                    'mean': np.mean(valid_data),
                    'std': np.std(valid_data),
                    'median': np.median(valid_data)
                }
                
                print(f"  Valid data: {valid_pct:.1f}% ({valid_count:,}/{total_count:,})")
                print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
                print(f"  Mean ± Std: {stats['mean']:.3f} ± {stats['std']:.3f}")
                print(f"  Median: {stats['median']:.3f}")
                
                # Check for physically unrealistic values
                if 'temp' in var or 't2m' in var:
                    if stats['min'] < 200 or stats['max'] > 350:
                        print(f"  ⚠️ WARNING: Temperature values outside realistic range (200-350 K)")
                elif 'tp' in var:
                    if stats['min'] < 0:
                        print(f"  ⚠️ WARNING: Negative precipitation values detected")
                elif 'fwi' in var:
                    if stats['min'] < 0 or stats['max'] > 100:
                        print(f"  ⚠️ WARNING: FWI values outside expected range (0-100)")
            else:
                print(f"  ⚠️ WARNING: Variable contains only NaN values!")
                
        else:
            # Static variable (like land_mask)
            valid_mask = ~np.isnan(data.values)
            valid_count = valid_mask.sum()
            total_count = data.size
            valid_pct = (valid_count / total_count) * 100
            
            if valid_count > 0:
                unique_values = np.unique(data.values[valid_mask])
                print(f"  Valid data: {valid_pct:.1f}% ({valid_count:,}/{total_count:,})")
                print(f"  Unique values: {len(unique_values)}")
                if len(unique_values) <= 10:
                    print(f"  Values: {unique_values}")
    
    # Spatial coverage check
    print("\n" + "=" * 70)
    print("SPATIAL COVERAGE ANALYSIS")
    print("=" * 70)
    
    # Check coverage for FWI (primary variable)
    if 'fwi' in ds.data_vars:
        fwi = ds['fwi']
        # Check first, middle, and last time steps
        time_indices = [0, len(ds.time)//2, -1]
        for idx in time_indices:
            time_val = pd.to_datetime(ds.time.values[idx]).strftime('%Y-%m-%d')
            slice_data = fwi.isel(time=idx)
            valid_pct = (~np.isnan(slice_data.values)).mean() * 100
            print(f"  FWI coverage on {time_val}: {valid_pct:.1f}%")
    
    # Check latitude/longitude bounds
    print(f"\nSpatial bounds:")
    print(f"  Latitude:  [{ds.latitude.min().values:.3f}, {ds.latitude.max().values:.3f}]")
    print(f"  Longitude: [{ds.longitude.min().values:.3f}, {ds.longitude.max().values:.3f}]")
    
    # Portugal approximate bounds check
    portugal_bounds = {
        'lat_min': 36.9, 'lat_max': 42.2,
        'lon_min': -9.5, 'lon_max': -6.2
    }
    
    if (ds.latitude.min() <= portugal_bounds['lat_min'] and 
        ds.latitude.max() >= portugal_bounds['lat_max'] and
        ds.longitude.min() <= portugal_bounds['lon_min'] and
        ds.longitude.max() >= portugal_bounds['lon_max']):
        print("  ✓ Spatial coverage includes mainland Portugal")
    else:
        print("  ⚠️ WARNING: Spatial coverage may not fully include mainland Portugal")
    
    # Memory estimate
    print("\n" + "=" * 70)
    print("MEMORY REQUIREMENTS")
    print("=" * 70)
    memory_gb = ds.nbytes / (1024**3)
    print(f"Uncompressed size in memory: {memory_gb:.2f} GB")
    compression_ratio = memory_gb / file_size_gb
    print(f"Compression ratio: {compression_ratio:.1f}:1")
    
    # Data quality summary
    print("\n" + "=" * 70)
    print("DATA QUALITY SUMMARY")
    print("=" * 70)
    
    issues = []
    warnings = []
    
    # Check each variable for completeness
    for var in ds.data_vars:
        if var in ['latitude', 'longitude', 'time', 'surface']:
            continue
        data = ds[var]
        if 'time' in data.dims:
            valid_pct = (~np.isnan(data.values)).mean() * 100
            if valid_pct < 50:
                issues.append(f"{var}: Only {valid_pct:.1f}% valid data")
            elif valid_pct < 90:
                warnings.append(f"{var}: {valid_pct:.1f}% valid data")
    
    if issues:
        print("\n❌ Critical Issues:")
        for issue in issues:
            print(f"  - {issue}")
    
    if warnings:
        print("\n⚠️ Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not issues and not warnings:
        print("\n✅ All variables have good data coverage (>90%)")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    if missing_days:
        print(f"1. Address {len(missing_days)} temporal gaps in the dataset")
    
    if issues:
        print(f"2. Investigate variables with low data coverage")
    
    if file_size_gb > 10:
        print(f"3. Consider chunking strategy for large file ({file_size_gb:.1f} GB)")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    
    return ds

def main():
    """Main validation workflow"""
    # Check both possible output files
    paths_to_check = [
        Path("data/02_final_unified/unified_complete_2010_2017.nc"),
        Path("data/02_final_unified/unified_fwi_dataset_2010_2017.nc")
    ]
    
    file_path = None
    for path in paths_to_check:
        if path.exists():
            file_path = path
            break
    
    if file_path is None:
        print("ERROR: No unified dataset found!")
        print("Expected locations:")
        for path in paths_to_check:
            print(f"  - {path}")
        return
    
    # Run validation
    ds = validate_unified_dataset(file_path)
    
    print("\n" + "=" * 70)
    print("Next steps:")
    print("1. Review validation report for any data quality issues")
    print("2. Address temporal gaps if critical for analysis")
    print("3. Verify variable ranges are physically realistic")
    print("4. Consider creating subset for specific experiments")
    print("=" * 70)

if __name__ == "__main__":
    main()