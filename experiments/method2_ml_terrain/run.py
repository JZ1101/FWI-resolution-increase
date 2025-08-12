#!/usr/bin/env python3
"""Run Method 2: ML with Terrain Features experiment"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from src.data_loader import FWIDataLoader
from src.models import MLTerrainDownscaler
from src.train import Trainer
from src.evaluate import Evaluator, BackAggregationValidator
from src.utils import load_config, save_config, setup_logging, Timer, save_netcdf
import logging

logger = logging.getLogger(__name__)


def main():
    setup_logging(level="INFO")
    
    config = load_config("config.yaml")
    base_config = load_config("../../configs/base.yaml")
    config = {**base_config, **config}
    
    logger.info("Starting Method 2: ML with Terrain Features")
    
    data_loader = FWIDataLoader(Path("../../data/raw"))
    
    logger.info("Loading data...")
    fwi_data = data_loader.load_era5_fwi(year=2017)
    fwi_values = fwi_data['fwinx'].values
    logger.info(f"FWI data shape: {fwi_values.shape}")
    
    # Split data
    n_time = fwi_values.shape[0]
    train_end = int(0.7 * n_time)
    test_start = int(0.85 * n_time)
    
    X_train = fwi_values[:train_end]
    X_test = fwi_values[test_start:]
    
    # Create target data (using interpolation as pseudo-target for training)
    from src.models import BilinearInterpolation
    interpolator = BilinearInterpolation(upscale_factor=4)
    y_train = interpolator.predict(X_train)
    
    # Initialize ML model
    model = MLTerrainDownscaler(
        upscale_factor=4,
        model_type='random_forest',
        n_estimators=100,
        max_depth=10
    )
    
    # Train the model
    logger.info("Training ML model...")
    trainer = Trainer(model, output_dir="results")
    
    with Timer("Training"):
        trainer.train(X_train, y_train)
    
    # Generate predictions
    logger.info("Generating predictions...")
    with Timer("Prediction"):
        predictions = model.predict(X_test)
    
    logger.info(f"Prediction shape: {predictions.shape}")
    
    # Validate with back-aggregation
    validator = BackAggregationValidator(upscale_factor=4)
    metrics = validator.validate(X_test, predictions)
    
    # Save results
    save_config(metrics, "results/metrics.yaml")
    save_netcdf(predictions, "results/predictions.nc", var_name='fwinx_highres')
    
    validator.plot_comparison(
        X_test,
        predictions,
        time_step=len(X_test)//2,
        save_path="results/comparison.png"
    )
    
    trainer.save_model("ml_terrain_2017")
    
    logger.info("Method 2 experiment completed")
    logger.info(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()