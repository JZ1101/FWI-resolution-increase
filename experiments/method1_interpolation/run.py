#!/usr/bin/env python3
"""Run Method 1: Bilinear Interpolation experiment"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from src.data_loader import FWIDataLoader
from src.models import BilinearInterpolation
from src.evaluate import Evaluator, BackAggregationValidator
from src.utils import load_config, save_config, setup_logging, Timer, save_netcdf
import logging

logger = logging.getLogger(__name__)


def main():
    setup_logging(level="INFO")
    
    config = load_config("config.yaml")
    base_config = load_config("../../configs/base.yaml")
    config = {**base_config, **config}
    
    logger.info("Starting Method 1: Bilinear Interpolation")
    
    data_loader = FWIDataLoader(Path("../../data/raw"))
    
    logger.info("Loading ERA5 FWI data...")
    fwi_data = data_loader.load_era5_fwi(year=2017)
    
    fwi_values = fwi_data['fwinx'].values
    logger.info(f"FWI data shape: {fwi_values.shape}")
    
    model = BilinearInterpolation(upscale_factor=4)
    
    with Timer("Interpolation"):
        high_res_pred = model.predict(fwi_values)
    
    logger.info(f"High-res prediction shape: {high_res_pred.shape}")
    
    validator = BackAggregationValidator(upscale_factor=4)
    metrics = validator.validate(fwi_values, high_res_pred)
    
    save_config(metrics, "results/metrics.yaml")
    
    save_netcdf(
        high_res_pred,
        "results/predictions.nc",
        var_name='fwinx_highres'
    )
    
    validator.plot_comparison(
        fwi_values,
        high_res_pred,
        time_step=180,
        save_path="results/comparison.png"
    )
    
    logger.info("Method 1 experiment completed")
    logger.info(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()