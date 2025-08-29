#!/usr/bin/env python3
"""
Complete data processing pipeline for all data sources
This script processes everything while maintaining the original framework
"""

import sys
from pathlib import Path
import logging
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_processing import (
    load_config,
    create_master_grid,
    process_landcover,
    load_era5_fwi,
    load_era5_atmospheric,
    load_era5_land,
    load_uerra,
    unify_datasets,
    save_dataset,
    validate_fire_detection
)

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('processing_complete.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def process_atmospheric_complete(atmospheric_path, master_grid):
    """Process all ERA5 atmospheric data files"""
    atmospheric_path = Path(atmospheric_path)
    
    if not atmospheric_path.exists():
        print(f"⚠️ Atmospheric data path not found: {atmospheric_path}")
        return xr.Dataset(coords=master_grid.coords)
    
    all_datasets = {}
    
    # Process daily mean files
    mean_files = sorted(atmospheric_path.glob("era5_daily_mean_*.nc"))
    if mean_files:
        print(f"  Processing {len(mean_files)} daily mean files...")
        mean_datasets = []
        for i, file in enumerate(mean_files):
            try:
                print(f"    [{i+1}/{len(mean_files)}] Loading {file.name}...")
                ds = xr.open_dataset(file)
                mean_datasets.append(ds)
            except Exception as e:
                print(f"    Error loading {file.name}: {e}")
        
        if mean_datasets:
            print("    Concatenating mean data...")
            all_datasets['mean'] = xr.concat(mean_datasets, dim='time')
    
    # Process temperature files
    temp_files = sorted(atmospheric_path.glob("era5_daily_max_temp_*.nc"))
    if temp_files:
        print(f"  Processing {len(temp_files)} temperature files...")
        temp_datasets = []
        for i, file in enumerate(temp_files):
            try:
                print(f"    [{i+1}/{len(temp_files)}] Loading {file.name}...")
                ds = xr.open_dataset(file)
                temp_datasets.append(ds)
            except Exception as e:
                print(f"    Error loading {file.name}: {e}")
        
        if temp_datasets:
            print("    Concatenating temperature data...")
            all_datasets['temp'] = xr.concat(temp_datasets, dim='time')
    
    # Process precipitation files
    precip_files = sorted(atmospheric_path.glob("era5_daily_total_precipitation_*.nc"))
    if precip_files:
        print(f"  Processing {len(precip_files)} precipitation files...")
        precip_datasets = []
        for i, file in enumerate(precip_files):
            try:
                print(f"    [{i+1}/{len(precip_files)}] Loading {file.name}...")
                ds = xr.open_dataset(file)
                precip_datasets.append(ds)
            except Exception as e:
                print(f"    Error loading {file.name}: {e}")
        
        if precip_datasets:
            print("    Concatenating precipitation data...")
            all_datasets['precip'] = xr.concat(precip_datasets, dim='time')
    
    # Merge all atmospheric data
    if all_datasets:
        print("  Merging all atmospheric variables...")
        # Start with mean data as base
        if 'mean' in all_datasets:
            combined = all_datasets['mean']
        else:
            combined = xr.Dataset(coords=master_grid.coords)
        
        # Add temperature data
        if 'temp' in all_datasets:
            for var in all_datasets['temp'].data_vars:
                if var not in combined.data_vars:
                    combined[f'temp_{var}'] = all_datasets['temp'][var]
        
        # Add precipitation data
        if 'precip' in all_datasets:
            for var in all_datasets['precip'].data_vars:
                if var not in combined.data_vars:
                    combined[f'precip_{var}'] = all_datasets['precip'][var]
        
        # Regrid to 1km
        print("  Regridding atmospheric data to 1km...")
        combined_1km = combined.interp(
            latitude=master_grid.latitude,
            longitude=master_grid.longitude,
            method='linear'
        )
        
        # Select time period matching master grid
        combined_1km = combined_1km.sel(time=master_grid.time, method='nearest')
        
        print(f"✅ Processed {len(combined_1km.data_vars)} atmospheric variables")
        return combined_1km
    
    return xr.Dataset(coords=master_grid.coords)

def process_land_complete(land_path, master_grid):
    """Process all ERA5-Land data files"""
    land_path = Path(land_path)
    
    if not land_path.exists():
        print(f"⚠️ ERA5-Land data path not found: {land_path}")
        return xr.Dataset(coords=master_grid.coords)
    
    land_files = sorted(land_path.glob("era5_land_may_nov_*.nc"))
    print(f"  Found {len(land_files)} ERA5-Land files")
    
    if land_files:
        print(f"  Processing ERA5-Land data...")
        datasets = []
        for i, file in enumerate(land_files):
            try:
                print(f"    [{i+1}/{len(land_files)}] Loading {file.name}...")
                ds = xr.open_dataset(file)
                datasets.append(ds)
            except Exception as e:
                print(f"    Error loading {file.name}: {e}")
        
        if datasets:
            print("    Concatenating ERA5-Land data...")
            combined_ds = xr.concat(datasets, dim='time')
            
            # ERA5-Land is at 0.1° (~10km), regrid to 1km
            print("  Regridding ERA5-Land from 10km to 1km...")
            land_1km = combined_ds.interp(
                latitude=master_grid.latitude,
                longitude=master_grid.longitude,
                method='linear'
            )
            
            # Select time period matching master grid
            land_1km = land_1km.sel(time=master_grid.time, method='nearest')
            
            print(f"✅ Processed {len(land_1km.data_vars)} land surface variables")
            return land_1km
    
    return xr.Dataset(coords=master_grid.coords)

def process_uerra_complete(uerra_path, master_grid):
    """Process all UERRA data files"""
    uerra_path = Path(uerra_path)
    
    if not uerra_path.exists():
        print(f"⚠️ UERRA data path not found: {uerra_path}")
        return xr.Dataset(coords=master_grid.coords)
    
    uerra_files = sorted(uerra_path.glob("uerra_mescan_may_nov_*.nc"))
    print(f"  Found {len(uerra_files)} UERRA files")
    
    if uerra_files:
        print(f"  Processing UERRA high-resolution data...")
        datasets = []
        for i, file in enumerate(uerra_files):
            try:
                print(f"    [{i+1}/{len(uerra_files)}] Loading {file.name} ({file.stat().st_size/1e9:.1f} GB)...")
                ds = xr.open_dataset(file, chunks={'time': 100})  # Use chunking for large files
                datasets.append(ds)
                print(f"      Variables: {list(ds.data_vars)}")
            except Exception as e:
                print(f"    Error loading {file.name}: {e}")
        
        if datasets:
            print("    Concatenating UERRA data...")
            combined_ds = xr.concat(datasets, dim='time')
            
            # UERRA is at ~5.5km resolution, regrid to 1km
            print("  Regridding UERRA from 5.5km to 1km...")
            uerra_1km = combined_ds.interp(
                latitude=master_grid.latitude,
                longitude=master_grid.longitude,
                method='linear'
            )
            
            # Select time period matching master grid
            uerra_1km = uerra_1km.sel(time=master_grid.time, method='nearest')
            
            print(f"✅ Processed {len(uerra_1km.data_vars)} UERRA variables")
            return uerra_1km
    
    return xr.Dataset(coords=master_grid.coords)

def main():
    """Main processing pipeline for all data"""
    
    print("="*70)
    print("COMPLETE FWI DATA PROCESSING PIPELINE")
    print("="*70)
    print(f"Started at: {datetime.now()}")
    
    # Load configuration
    print("\n📝 Loading configuration...")
    config = load_config()
    
    # Step 1: Create master grid
    print("\n📍 STEP 1: Creating 1km master grid...")
    master_grid = create_master_grid(config)
    save_dataset(master_grid, Path(config['data']['data_paths']['master_grid']), compress=False)
    
    # Step 2: Process land cover
    print("\n🌍 STEP 2: Processing land cover...")
    landcover_path = config['data']['data_paths'].get('raw_landcover', '')
    landcover_data = process_landcover(master_grid, landcover_path)
    
    # Step 3: Process FWI data
    print("\n🔥 STEP 3: Processing ERA5 FWI data...")
    fwi_path = Path(config['data']['data_paths']['raw_fwi'])
    if fwi_path.exists():
        fwi_data = load_era5_fwi(fwi_path, master_grid, config)
    else:
        print(f"❌ FWI data not found: {fwi_path}")
        return
    
    # Step 4: Process atmospheric data
    print("\n☁️ STEP 4: Processing ERA5 atmospheric data...")
    atmospheric_path = Path(config['data']['data_paths'].get('raw_era5_atmospheric', ''))
    atmospheric_data = process_atmospheric_complete(atmospheric_path, master_grid)
    
    # Step 5: Process land surface data
    print("\n🏔️ STEP 5: Processing ERA5-Land data...")
    land_path = Path(config['data']['data_paths'].get('raw_era5_land', ''))
    land_data = process_land_complete(land_path, master_grid)
    
    # Step 6: Process UERRA data
    print("\n🎯 STEP 6: Processing UERRA high-resolution data...")
    uerra_path = Path(config['data']['data_paths'].get('raw_uerra', ''))
    uerra_data = process_uerra_complete(uerra_path, master_grid)
    
    # Step 7: Unify all datasets
    print("\n🔄 STEP 7: Unifying all datasets...")
    unified_dataset = unify_datasets(
        master_grid,
        fwi_data,
        landcover_data,
        atmospheric_data,
        land_data,
        uerra_data
    )
    
    # Step 8: Validate fire detection
    print("\n🔥 STEP 8: Validating fire detection...")
    fire_detected = validate_fire_detection(unified_dataset, config)
    
    # Step 9: Save final dataset
    print("\n💾 STEP 9: Saving unified dataset...")
    output_path = Path(config['data']['data_paths']['unified_dataset'])
    save_dataset(unified_dataset, output_path, compress=True)
    
    # Final summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE!")
    print("="*70)
    print(f"✅ Unified dataset saved: {output_path}")
    print(f"✅ Dataset dimensions: {dict(unified_dataset.dims)}")
    print(f"✅ Variables ({len(unified_dataset.data_vars)}): {list(unified_dataset.data_vars)}")
    print(f"✅ Fire detection: {'PASSED' if fire_detected else 'FAILED'}")
    print(f"✅ Total size: {unified_dataset.nbytes / 1e9:.2f} GB")
    print(f"✅ Completed at: {datetime.now()}")
    
    # Generate quality report
    print("\n📊 Generating quality report...")
    generate_quality_report(unified_dataset, output_path.parent / "processing_report.txt")
    
    return unified_dataset

def generate_quality_report(dataset, report_path):
    """Generate comprehensive quality report"""
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("FWI DATASET PROCESSING REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Generated: {datetime.now()}\n\n")
        
        f.write("DATASET SUMMARY\n")
        f.write("-"*50 + "\n")
        f.write(f"Dimensions: {dict(dataset.dims)}\n")
        f.write(f"Total size: {dataset.nbytes / 1e9:.2f} GB\n")
        f.write(f"Number of variables: {len(dataset.data_vars)}\n\n")
        
        f.write("VARIABLES\n")
        f.write("-"*50 + "\n")
        for var in dataset.data_vars:
            var_data = dataset[var]
            nan_count = np.isnan(var_data.values).sum()
            nan_pct = 100 * nan_count / var_data.size
            f.write(f"{var:30s} - Shape: {str(var_data.shape):20s} NaN: {nan_pct:.1f}%\n")
        
        f.write("\nTEMPORAL COVERAGE\n")
        f.write("-"*50 + "\n")
        f.write(f"Start: {dataset.time.values[0]}\n")
        f.write(f"End: {dataset.time.values[-1]}\n")
        f.write(f"Total days: {len(dataset.time)}\n")
        
        f.write("\nSPATIAL COVERAGE\n")
        f.write("-"*50 + "\n")
        f.write(f"Latitude: {dataset.latitude.min().values:.2f}° to {dataset.latitude.max().values:.2f}°\n")
        f.write(f"Longitude: {dataset.longitude.min().values:.2f}° to {dataset.longitude.max().values:.2f}°\n")
        f.write(f"Resolution: 0.01° (~1.1 km)\n")
        
        f.write("\nMETADATA\n")
        f.write("-"*50 + "\n")
        for key, value in dataset.attrs.items():
            f.write(f"{key}: {value}\n")
    
    print(f"✅ Report saved: {report_path}")

if __name__ == "__main__":
    dataset = main()