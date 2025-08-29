#!/usr/bin/env python3
"""
Main preprocessing script for FWI super-resolution project

This script orchestrates the preprocessing pipeline by calling functions
from the src.data_processing module in the correct sequence.
"""

import sys
from pathlib import Path
import logging
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_processing import (
    load_config,
    create_master_grid,
    process_landcover,
    load_era5_fwi,
    load_era5_atmospheric,
    load_era5_land,
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
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def main():
    """Main preprocessing pipeline"""
    
    # Setup
    parser = argparse.ArgumentParser(description='FWI Preprocessing Pipeline')
    parser.add_argument('--config', type=str, default='configs/params.yaml',
                       help='Path to configuration file')
    parser.add_argument('--skip-weather', action='store_true',
                       help='Skip weather data processing (for testing)')
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("=" * 70)
    logger.info("FWI SUPER-RESOLUTION PREPROCESSING PIPELINE")
    logger.info("=" * 70)
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Step 1: Create master grid
    logger.info("\n📍 STEP 1: Creating master grid...")
    master_grid = create_master_grid(config)
    
    # Save master grid
    master_grid_path = Path(config['data']['data_paths']['master_grid'])
    save_dataset(master_grid, master_grid_path, compress=False)
    
    # Step 2: Process land cover
    logger.info("\n🌍 STEP 2: Processing land cover...")
    landcover_path = config['data']['data_paths'].get('raw_landcover', '')
    landcover_data = process_landcover(master_grid, landcover_path)
    
    # Step 3: Load and process FWI data
    logger.info("\n🔥 STEP 3: Loading ERA5 FWI data...")
    
    # Use the actual FWI data path from config
    fwi_path = Path(config['data']['data_paths']['raw_fwi'])
    
    if not fwi_path.exists():
        logger.warning("FWI data file not found. Using dummy data for testing.")
        # Create dummy FWI data for testing
        import xarray as xr
        import numpy as np
        
        # Get dimensions in the correct order
        n_time = len(master_grid.time)
        n_lat = len(master_grid.latitude)
        n_lon = len(master_grid.longitude)
        
        fwi_data = xr.Dataset(coords=master_grid.coords)
        fwi_data['fwi'] = xr.DataArray(
            np.random.rand(n_time, n_lat, n_lon) * 50,  # Random FWI values 0-50
            dims=['time', 'latitude', 'longitude'],
            coords={'time': master_grid.time, 'latitude': master_grid.latitude, 'longitude': master_grid.longitude}
        )
    else:
        fwi_data = load_era5_fwi(fwi_path, master_grid, config)
    
    # Step 4: Load weather data (optional)
    atmospheric_data = None
    land_data = None
    uerra_data = None
    
    if not args.skip_weather:
        logger.info("\n☁️ STEP 4: Loading atmospheric data...")
        atmospheric_path = Path(config['data']['data_paths'].get('raw_era5_atmospheric', ''))
        if atmospheric_path.exists():
            from src.data_processing import load_era5_atmospheric
            atmospheric_data = load_era5_atmospheric(atmospheric_path, master_grid)
        else:
            logger.warning(f"Atmospheric data path not found: {atmospheric_path}")
        
        logger.info("\n🏔️ STEP 5: Loading land surface data...")
        land_path = Path(config['data']['data_paths'].get('raw_era5_land', ''))
        if land_path.exists():
            from src.data_processing import load_era5_land
            land_data = load_era5_land(land_path, master_grid)
        else:
            logger.warning(f"Land data path not found: {land_path}")
            
        logger.info("\n🎯 STEP 6: Loading UERRA high-resolution data...")
        uerra_path = Path(config['data']['data_paths'].get('raw_uerra', ''))
        if uerra_path.exists():
            from src.data_processing import load_uerra
            uerra_data = load_uerra(uerra_path, master_grid)
        else:
            logger.warning(f"UERRA data path not found: {uerra_path}")
    else:
        logger.info("\n⏭️ Skipping weather data processing (--skip-weather flag)")
    
    # Step 7: Unify all datasets
    logger.info("\n🔄 STEP 7: Unifying all datasets...")
    unified_dataset = unify_datasets(
        master_grid,
        fwi_data,
        landcover_data,
        atmospheric_data,
        land_data,
        uerra_data
    )
    
    # Step 6: Validate fire detection
    logger.info("\n🔥 STEP 7: Validating fire detection...")
    fire_detected = validate_fire_detection(unified_dataset, config)
    
    # Step 7: Save final dataset
    logger.info("\n💾 STEP 8: Saving unified dataset...")
    output_path = Path(config['data']['data_paths']['unified_dataset'])
    save_dataset(unified_dataset, output_path, compress=True)
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("PREPROCESSING COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"✅ Master grid created: {master_grid_path}")
    logger.info(f"✅ Unified dataset saved: {output_path}")
    logger.info(f"✅ Dataset dimensions: {dict(unified_dataset.dims)}")
    logger.info(f"✅ Variables: {list(unified_dataset.data_vars)}")
    logger.info(f"✅ Fire detection: {'PASSED' if fire_detected else 'FAILED'}")
    logger.info(f"✅ Total size: {unified_dataset.nbytes / 1e9:.2f} GB")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())