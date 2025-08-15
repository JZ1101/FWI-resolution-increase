#!/usr/bin/env python3
"""Prepare data for FWI experiments"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import FWIDataLoader
from src.utils import setup_logging, Timer
import numpy as np
import logging

logger = logging.getLogger(__name__)


def main():
    setup_logging(level="INFO")
    
    data_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    loader = FWIDataLoader(data_dir)
    
    # Process data for years 2015-2017 (excluding 2018)
    years = [2015, 2016, 2017]
    all_fwi_data = []
    all_atmospheric_data = []
    all_land_data = []
    
    for year in years:
        logger.info(f"Loading {year} data...")
        with Timer(f"Data loading for {year}"):
            fwi = loader.load_era5_fwi(year)
            atmospheric = loader.load_era5_atmospheric(year)
            land = loader.load_era5_land(year)
            
            all_fwi_data.append(fwi)
            all_atmospheric_data.append(atmospheric)
            all_land_data.append(land)
    
    logger.info("Combining data from all years...")
    # Combine data from all years
    combined_fwi = np.concatenate(all_fwi_data, axis=0)
    combined_atmospheric = np.concatenate(all_atmospheric_data, axis=0)
    combined_land = np.concatenate(all_land_data, axis=0)
    
    logger.info("Preparing training data...")
    X, _, features = loader.prepare_training_data(combined_fwi, features=combined_atmospheric)
    
    logger.info("Creating train/val/test splits...")
    splits = loader.split_data(X, train_ratio=0.7, val_ratio=0.15)
    
    for name, data in splits.items():
        save_path = processed_dir / f"{name}_2015_2017.npy"
        np.save(save_path, data)
        logger.info(f"Saved {name}: shape {data.shape}")
    
    # Also save the combined datasets
    np.save(processed_dir / "fwi_2015_2017.npy", combined_fwi)
    np.save(processed_dir / "atmospheric_2015_2017.npy", combined_atmospheric)
    np.save(processed_dir / "land_2015_2017.npy", combined_land)
    
    logger.info(f"Data preparation complete for years {years}")
    logger.info(f"Total samples: {X.shape[0]}")
    logger.info(f"Features: {X.shape[1]}")


if __name__ == "__main__":
    main()