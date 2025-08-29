#!/usr/bin/env python3
"""
Run preprocessing with selected data sources (skip problematic UERRA)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_processing import (
    load_config,
    create_master_grid,
    process_landcover,
    load_era5_fwi,
    save_dataset,
    validate_fire_detection
)
import xarray as xr
import numpy as np

def load_atmospheric_simple(atmospheric_path, master_grid):
    """Load only temperature data from atmospheric"""
    atmospheric_path = Path(atmospheric_path)
    
    if not atmospheric_path.exists():
        print(f"⚠️ Atmospheric data path not found: {atmospheric_path}")
        return None
        
    # Load temperature files
    temp_files = sorted(atmospheric_path.glob("era5_daily_max_temp_*.nc"))
    print(f"  Found {len(temp_files)} temperature files")
    
    if temp_files:
        datasets = []
        for i, file in enumerate(temp_files[:10], 1):  # Test with first 10 files
            print(f"    [{i}/10] Loading {file.name}...")
            try:
                ds = xr.open_dataset(file)
                datasets.append(ds)
            except Exception as e:
                print(f"    Error: {e}")
        
        if datasets:
            print("  Concatenating temperature data...")
            temp_data = xr.concat(datasets, dim='time')
            
            # Regrid to master grid
            print("  Regridding to 1km...")
            temp_1km = temp_data.interp(
                latitude=master_grid.latitude,
                longitude=master_grid.longitude,
                method='linear'
            )
            
            return temp_1km
    
    return None

def load_land_simple(land_path, master_grid):
    """Load ERA5-Land data"""
    land_path = Path(land_path)
    
    if not land_path.exists():
        print(f"⚠️ Land data path not found: {land_path}")
        return None
    
    land_files = sorted(land_path.glob("era5_land_may_nov_*.nc"))
    print(f"  Found {len(land_files)} ERA5-Land files")
    
    if land_files and len(land_files) > 0:
        # Load first file as test
        print(f"  Loading {land_files[0].name}...")
        try:
            ds = xr.open_dataset(land_files[0])
            
            # Regrid to 1km
            print("  Regridding from 10km to 1km...")
            land_1km = ds.interp(
                latitude=master_grid.latitude,
                longitude=master_grid.longitude,
                method='linear'
            )
            
            return land_1km
        except Exception as e:
            print(f"  Error: {e}")
    
    return None

def main():
    print("="*70)
    print("PARTIAL FWI DATA AGGREGATION")
    print("="*70)
    
    # Load config
    config = load_config()
    
    # Create master grid
    print("\n📍 Creating master grid...")
    master_grid = create_master_grid(config)
    save_dataset(master_grid, Path(config['data']['data_paths']['master_grid']), compress=False)
    
    # Process land cover
    print("\n🌍 Processing land cover...")
    landcover_path = config['data']['data_paths'].get('raw_landcover', '')
    landcover_data = process_landcover(master_grid, landcover_path)
    
    # Load FWI
    print("\n🔥 Loading FWI data...")
    fwi_path = Path(config['data']['data_paths']['raw_fwi'])
    fwi_data = load_era5_fwi(fwi_path, master_grid, config)
    
    # Load atmospheric
    print("\n☁️ Loading atmospheric data...")
    atmospheric_path = Path(config['data']['data_paths'].get('raw_era5_atmospheric', ''))
    atmospheric_data = load_atmospheric_simple(atmospheric_path, master_grid)
    
    # Load land surface
    print("\n🏔️ Loading ERA5-Land data...")
    land_path = Path(config['data']['data_paths'].get('raw_era5_land', ''))
    land_data = load_land_simple(land_path, master_grid)
    
    # Combine datasets
    print("\n🔄 Combining datasets...")
    unified = xr.merge([
        fwi_data,
        landcover_data
    ])
    
    if atmospheric_data is not None:
        print("  Adding atmospheric variables...")
        for var in atmospheric_data.data_vars:
            if var not in unified.data_vars:
                unified[var] = atmospheric_data[var]
    
    if land_data is not None:
        print("  Adding land surface variables...")
        for var in land_data.data_vars:
            if var not in unified.data_vars:
                unified[var] = land_data[var]
    
    # Add metadata
    unified.attrs = {
        'title': 'FWI Unified Dataset Portugal 2010-2017',
        'description': 'Fire Weather Index and predictor variables at 1km resolution',
        'source': 'ERA5, ERA5-Land',
        'creation_date': str(np.datetime64('today')),
        'resolution': '0.01 degrees (~1km)',
        'region': 'Portugal'
    }
    
    # Save
    output_path = Path(config['data']['data_paths']['unified_dataset'])
    print(f"\n💾 Saving to {output_path}...")
    save_dataset(unified, output_path, compress=True)
    
    print(f"\n✅ COMPLETE!")
    print(f"   Variables: {list(unified.data_vars)}")
    print(f"   Dimensions: {dict(unified.dims)}")
    print(f"   Size: {unified.nbytes / 1e9:.2f} GB")
    
    return unified

if __name__ == "__main__":
    dataset = main()