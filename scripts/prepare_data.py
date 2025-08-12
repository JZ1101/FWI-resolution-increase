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
    
    logger.info("Loading 2017 data...")
    with Timer("Data loading"):
        fwi = loader.load_era5_fwi(2017)
        atmospheric = loader.load_era5_atmospheric(2017)
        land = loader.load_era5_land(2017)
    
    logger.info("Preparing training data...")
    X, _, features = loader.prepare_training_data(fwi, features=atmospheric)
    
    logger.info("Creating train/val/test splits...")
    splits = loader.split_data(X, train_ratio=0.7, val_ratio=0.15)
    
    for name, data in splits.items():
        save_path = processed_dir / f"{name}.npy"
        np.save(save_path, data)
        logger.info(f"Saved {name}: shape {data.shape}")
    
    logger.info("Data preparation complete")


if __name__ == "__main__":
    main()