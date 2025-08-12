#!/usr/bin/env python3
"""
Run 2017 Portugal FWI Resolution Enhancement Experiment
Migrated from ml-experiment/database_portugal_2017/
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import xarray as xr
from src.data_loader import FWIDataLoader
from src.models import get_model
from src.train import Trainer
from src.evaluate import Evaluator, BackAggregationValidator
from src.utils import load_config, save_config, setup_logging, Timer, save_netcdf
import logging
import json

logger = logging.getLogger(__name__)


class Portugal2017Experiment:
    """Complete experiment for Portugal 2017 FWI data"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.results = {}
        
    def run_method(self, method_name: str, model_config: dict) -> dict:
        """Run a single downscaling method"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Running Method: {method_name}")
        logger.info(f"{'='*60}")
        
        # Load data
        data_path = Path(self.config.get('data_path', '../../data/raw'))
        loader = FWIDataLoader(data_path)
        
        # Load 2017 FWI data
        fwi_data = loader.load_era5_fwi(year=2017)
        fwi_values = fwi_data['fwinx'].values
        
        logger.info(f"FWI data shape: {fwi_values.shape}")
        
        # Load auxiliary data if needed
        auxiliary_features = None
        if method_name in ['ml_terrain', 'physics_ml', 'ensemble']:
            atmospheric = loader.load_era5_atmospheric(2017)
            land = loader.load_era5_land(2017)
            if atmospheric is not None or land is not None:
                auxiliary_features = atmospheric
        
        # Initialize model
        model = get_model(method_name, model_config)
        
        # Prepare data splits
        n_time = fwi_values.shape[0]
        train_end = int(0.7 * n_time)
        val_end = int(0.85 * n_time)
        
        X_train = fwi_values[:train_end]
        X_val = fwi_values[train_end:val_end]
        X_test = fwi_values[val_end:]
        
        # Training (if required)
        if hasattr(model, 'fit') and method_name != 'interpolation':
            logger.info("Training model...")
            
            # For supervised methods, we need high-res targets
            # Here we use interpolated as pseudo-targets for demonstration
            from src.models import BilinearInterpolation
            interpolator = BilinearInterpolation(upscale_factor=4)
            y_train = interpolator.predict(X_train)
            
            trainer = Trainer(model, output_dir=f"results/{method_name}")
            
            with Timer(f"Training {method_name}"):
                history = trainer.train(
                    X_train, y_train,
                    X_val=X_val,
                    auxiliary_features=auxiliary_features
                )
            
            # Save model
            model_path = trainer.save_model(f"{method_name}_2017")
        
        # Prediction
        logger.info("Generating predictions...")
        with Timer(f"Prediction {method_name}"):
            predictions = model.predict(X_test)
        
        logger.info(f"Predictions shape: {predictions.shape}")
        
        # Validation
        validator = BackAggregationValidator(upscale_factor=4)
        metrics = validator.validate(X_test, predictions)
        
        # Save results
        results = {
            'method': method_name,
            'metrics': metrics,
            'input_shape': X_test.shape,
            'output_shape': predictions.shape,
            'upscale_factor': 4
        }
        
        # Save predictions
        save_path = Path(f"results/{method_name}")
        save_path.mkdir(parents=True, exist_ok=True)
        
        save_netcdf(
            predictions,
            save_path / "predictions.nc",
            var_name='fwinx_highres'
        )
        
        # Save metrics
        with open(save_path / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Generate comparison plot
        validator.plot_comparison(
            X_test,
            predictions,
            time_step=len(X_test)//2,  # Middle of test period
            save_path=str(save_path / "comparison.png")
        )
        
        return results
    
    def run_all_methods(self):
        """Run all configured methods"""
        
        methods = {
            'interpolation': {'upscale_factor': 4},
            'ml_terrain': {
                'upscale_factor': 4,
                'model_type': 'random_forest',
                'n_estimators': 100,
                'max_depth': 10
            },
            'physics_ml': {
                'upscale_factor': 4,
                'physics_weight': 0.6,
                'model_type': 'gradient_boosting'
            },
            'ensemble': {
                'upscale_factor': 4
            }
        }
        
        # Add XGBoost if available
        try:
            import xgboost
            methods['xgboost'] = {
                'upscale_factor': 4,
                'n_estimators': 100,
                'max_depth': 6
            }
        except ImportError:
            logger.warning("XGBoost not available")
        
        # Add CNN if torch available
        try:
            import torch
            methods['cnn'] = {
                'input_channels': 1,
                'upscale_factor': 4
            }
        except ImportError:
            logger.warning("PyTorch not available for CNN model")
        
        for method_name, config in methods.items():
            try:
                result = self.run_method(method_name, config)
                self.results[method_name] = result
            except Exception as e:
                logger.error(f"Error running {method_name}: {e}")
                self.results[method_name] = {'error': str(e)}
        
        # Save summary
        self.save_summary()
        
    def save_summary(self):
        """Save experiment summary"""
        
        summary = {
            'experiment': 'Portugal 2017 FWI Resolution Enhancement',
            'methods': {}
        }
        
        for method, result in self.results.items():
            if 'error' not in result:
                summary['methods'][method] = {
                    'correlation': result['metrics'].get('correlation', 0),
                    'rmse': result['metrics'].get('rmse', 0),
                    'conservation_error': result['metrics'].get('conservation_error', 0),
                    'r2': result['metrics'].get('r2', 0)
                }
            else:
                summary['methods'][method] = {'error': result['error']}
        
        # Save as JSON
        with open('results/experiment_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create comparison table
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("="*80)
        
        for method, metrics in summary['methods'].items():
            if 'error' not in metrics:
                logger.info(f"\n{method.upper()}:")
                logger.info(f"  Correlation: {metrics.get('correlation', 0):.4f}")
                logger.info(f"  RMSE: {metrics.get('rmse', 0):.4f}")
                logger.info(f"  Conservation Error: {metrics.get('conservation_error', 0):.2f}%")
                logger.info(f"  R²: {metrics.get('r2', 0):.4f}")
            else:
                logger.info(f"\n{method.upper()}: FAILED - {metrics['error']}")


def main():
    """Main entry point"""
    
    setup_logging(level="INFO", log_file="results/experiment.log")
    
    # Create config if it doesn't exist
    config_path = Path("config.yaml")
    if not config_path.exists():
        config = {
            'data_path': '../../data/raw',
            'year': 2017,
            'upscale_factor': 4,
            'validation': {
                'back_aggregation': True,
                'metrics': ['rmse', 'mae', 'correlation', 'r2', 'conservation_error']
            }
        }
        save_config(config, config_path)
    
    # Run experiment
    experiment = Portugal2017Experiment("config.yaml")
    experiment.run_all_methods()
    
    logger.info("\nExperiment completed successfully!")
    logger.info("Results saved in results/ directory")


if __name__ == "__main__":
    main()