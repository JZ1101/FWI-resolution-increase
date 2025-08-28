#!/usr/bin/env python3
"""
Step 5: Final Hierarchical Merge of All Datasets
- Load existing FWI dataset (Dataset 1) 
- Load processed ERA5 Atmospheric (Dataset 2)
- Load processed ERA5-Land (Dataset 3)
- Load processed UERRA (Dataset 4)
- Load processed ESA Land Cover (Dataset 5)
- Perform hierarchical merge with priority: UERRA > ERA5-Land > ERA5 Atmospheric
- Save final complete unified dataset
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def hierarchical_merge():
    """Perform hierarchical merge of all 5 datasets"""
    print("=" * 70)
    print("STEP 5: FINAL HIERARCHICAL MERGE")
    print("=" * 70)
    
    # Define input paths
    datasets = {
        'FWI': Path("data/02_final_unified/unified_complete_2010_2017.nc"),
        'ERA5_Atmospheric': Path("data/01_processed/processed_era5_atmospheric_1km.nc"),
        'ERA5_Land': Path("data/01_processed/processed_era5_land_1km.nc"),
        'UERRA': Path("data/01_processed/processed_uerra_1km.nc"),
        'ESA_LandCover': Path("data/01_processed/processed_landcover_1km.nc")
    }
    
    # Check all files exist
    print("\nChecking input files...")
    all_exist = True
    for name, path in datasets.items():
        if path.exists():
            size_gb = path.stat().st_size / (1024**3)
            print(f"  ✓ {name}: {size_gb:.2f} GB")
        else:
            print(f"  ✗ {name}: NOT FOUND at {path}")
            all_exist = False
    
    if not all_exist:
        print("\nERROR: Not all required datasets found!")
        return None
    
    # Load datasets
    print("\n" + "=" * 60)
    print("Loading datasets...")
    print("=" * 60)
    
    loaded_data = {}
    for name, path in datasets.items():
        print(f"\nLoading {name}...")
        ds = xr.open_dataset(path)
        print(f"  Variables: {list(ds.data_vars)}")
        if 'time' in ds.dims:
            print(f"  Time steps: {len(ds.time)}")
        loaded_data[name] = ds
    
    # Start with FWI as base (contains fwi and land_mask)
    print("\n" + "=" * 60)
    print("Building final dataset...")
    print("=" * 60)
    
    fwi_ds = loaded_data['FWI']
    
    # Extract just fwi and land_mask from FWI dataset (ignore prefixed variables from earlier merge)
    print("\nExtracting core FWI variables...")
    final = xr.Dataset({
        'fwi': fwi_ds['fwi'],
        'land_mask': fwi_ds['land_mask']
    })
    print(f"  Added: fwi, land_mask")
    
    # Define variable hierarchy for overlapping variables
    # Priority: UERRA > ERA5-Land > ERA5 Atmospheric
    overlapping_vars = ['t2m', 'd2m', 'u10', 'v10', 'tp', 'si10', 'r2']
    
    print("\nProcessing overlapping variables with hierarchy...")
    print("Priority: UERRA > ERA5-Land > ERA5 Atmospheric")
    
    processed_vars = set()
    
    # Process each potential overlapping variable
    for var in overlapping_vars:
        var_added = False
        
        # Check UERRA first (highest priority)
        if 'UERRA' in loaded_data and var in loaded_data['UERRA'].data_vars:
            print(f"  {var}: Using UERRA (highest resolution)")
            final[var] = loaded_data['UERRA'][var]
            processed_vars.add(var)
            var_added = True
        
        # Check ERA5-Land next
        elif 'ERA5_Land' in loaded_data and var in loaded_data['ERA5_Land'].data_vars:
            print(f"  {var}: Using ERA5-Land")
            final[var] = loaded_data['ERA5_Land'][var]
            processed_vars.add(var)
            var_added = True
        
        # Check ERA5 Atmospheric last
        elif 'ERA5_Atmospheric' in loaded_data and var in loaded_data['ERA5_Atmospheric'].data_vars:
            print(f"  {var}: Using ERA5 Atmospheric")
            final[var] = loaded_data['ERA5_Atmospheric'][var]
            processed_vars.add(var)
            var_added = True
        
        if not var_added and var in ['t2m', 'd2m', 'u10', 'v10', 'tp']:
            # These are critical variables that should exist
            print(f"  WARNING: Variable '{var}' not found in any dataset!")
    
    # Add any unique variables from each dataset
    print("\nAdding unique variables from each dataset...")
    
    # Check for unique ERA5 Atmospheric variables
    for var in loaded_data['ERA5_Atmospheric'].data_vars:
        if var not in processed_vars and var not in final.data_vars:
            print(f"  Adding '{var}' from ERA5 Atmospheric")
            final[var] = loaded_data['ERA5_Atmospheric'][var]
    
    # Check for unique ERA5-Land variables
    for var in loaded_data['ERA5_Land'].data_vars:
        if var not in processed_vars and var not in final.data_vars:
            print(f"  Adding '{var}' from ERA5-Land")
            final[var] = loaded_data['ERA5_Land'][var]
    
    # Check for unique UERRA variables
    for var in loaded_data['UERRA'].data_vars:
        if var not in processed_vars and var not in final.data_vars:
            print(f"  Adding '{var}' from UERRA")
            final[var] = loaded_data['UERRA'][var]
    
    # Add all ESA Land Cover variables (they're all unique)
    print("\nAdding ESA Land Cover fractions...")
    lc_vars_added = 0
    for var in loaded_data['ESA_LandCover'].data_vars:
        if var not in final.data_vars:
            # Land cover is static (no time dimension)
            # Broadcast to match time dimension if needed
            lc_data = loaded_data['ESA_LandCover'][var]
            final[var] = lc_data
            lc_vars_added += 1
    print(f"  Added {lc_vars_added} land cover fraction variables")
    
    # Update attributes
    final.attrs['title'] = 'Unified Portugal FWI Dataset 2010-2017 (Complete Hierarchical Merge)'
    final.attrs['description'] = 'Complete dataset with FWI, ERA5, UERRA, and ESA Land Cover'
    final.attrs['creation_date'] = pd.Timestamp.now().isoformat()
    final.attrs['sources'] = 'FWI reanalysis, ERA5 Atmospheric, ERA5-Land, UERRA, ESA WorldCover'
    final.attrs['processing'] = 'Hierarchical merge with priority: UERRA > ERA5-Land > ERA5 Atmospheric'
    final.attrs['merge_hierarchy'] = 'For overlapping variables: UERRA (5.5km) > ERA5-Land (9km) > ERA5 Atmospheric (25km)'
    
    # Print summary
    print("\n" + "=" * 60)
    print("Final dataset summary:")
    print("=" * 60)
    print(f"  Total variables: {len(final.data_vars)}")
    print(f"\n  Variable categories:")
    
    # Categorize variables
    fwi_vars = [v for v in final.data_vars if 'fwi' in v.lower()]
    lc_vars = [v for v in final.data_vars if 'lc_frac' in v]
    meteo_vars = [v for v in final.data_vars if v not in fwi_vars + lc_vars + ['land_mask']]
    
    print(f"    - FWI variables: {len(fwi_vars)}")
    print(f"    - Meteorological variables: {len(meteo_vars)}")
    print(f"    - Land cover fractions: {len(lc_vars)}")
    print(f"    - Other: 1 (land_mask)")
    
    print(f"\n  Dimensions: {dict(final.dims)}")
    print(f"  Grid: {len(final.latitude)} × {len(final.longitude)} (1km resolution)")
    
    if 'time' in final.dims:
        print(f"  Time steps: {len(final.time)}")
        print(f"  Time range: {final.time.min().values} to {final.time.max().values}")
    
    # Save final dataset
    output_dir = Path("data/02_final_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "unified_complete_dataset.nc"
    
    print(f"\nSaving final dataset to: {output_path}")
    
    # Encode with compression
    encoding = {}
    for var in final.data_vars:
        if var not in ['latitude', 'longitude', 'time']:
            encoding[var] = {'zlib': True, 'complevel': 4}
    
    print("  Saving with compression...")
    final.to_netcdf(output_path, encoding=encoding)
    
    # Verify saved file
    actual_size_gb = output_path.stat().st_size / (1024**3)
    print(f"  ✓ Saved successfully! Final file size: {actual_size_gb:.2f} GB")
    
    # List all variables in final dataset
    print(f"\n  Complete variable list ({len(final.data_vars)} variables):")
    for i, var in enumerate(sorted(final.data_vars), 1):
        print(f"    {i:2d}. {var}")
    
    return output_path

def main():
    """Main workflow for Step 5"""
    output_path = hierarchical_merge()
    
    if output_path and output_path.exists():
        print("\n" + "=" * 70)
        print("DELIVERABLE VERIFICATION")
        print("=" * 70)
        print("\nRun this command to verify ALL variables:")
        print(f'ncdump -h {output_path} | grep "float\\|double"')
        print("\nExpected: A single, clean list of variables from all 5 datasets")
        print("with no prefixed duplicates (no atm_*, land_* prefixes)")
        print("\n" + "=" * 70)
        print("PREPROCESSING COMPLETE!")
        print("=" * 70)
        print(f"\nFinal unified dataset ready at: {output_path}")
        print("This dataset contains all 5 source datasets merged hierarchically.")

if __name__ == "__main__":
    main()