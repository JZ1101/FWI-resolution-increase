#!/usr/bin/env python3
"""
Main training script for FWI super-resolution models

This script handles model training and evaluation by orchestrating
functions from the src modules.
"""

import sys
from pathlib import Path
import argparse
import logging
import torch
import numpy as np
import xarray as xr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_processing import (
    load_config,
    prepare_ml_data
)
from src.model import create_model, count_parameters
from src.train import (
    setup_logging,
    create_data_loaders,
    train_model,
    save_checkpoint,
    plot_training_history
)
from src.evaluate import (
    evaluate_model,
    create_results_table,
    plot_comparison_maps,
    plot_residuals
)


def main():
    """Main training pipeline"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='FWI Model Training Pipeline')
    parser.add_argument('--experiment', type=str, default='primary_unet',
                       help='Experiment name (primary_unet, simple_cnn, bilinear, random_forest)')
    parser.add_argument('--config', type=str, default='configs/params.yaml',
                       help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, auto)')
    parser.add_argument('--test-only', action='store_true',
                       help='Only run evaluation on existing model')
    args = parser.parse_args()
    
    # Setup
    reports_dir = Path('reports')
    models_dir = Path('models')
    reports_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    
    logger = setup_logging(reports_dir)
    logger.info("=" * 70)
    logger.info("FWI SUPER-RESOLUTION TRAINING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Experiment: {args.experiment}")
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Override config if arguments provided
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    
    # Set experiment name
    config['training']['experiment_name'] = args.experiment
    
    # Determine model type from experiment name
    if 'unet' in args.experiment.lower():
        config['model']['architecture'] = 'unet'
    elif 'cnn' in args.experiment.lower():
        config['model']['architecture'] = 'simple_cnn'
    elif 'bilinear' in args.experiment.lower():
        config['model']['architecture'] = 'bilinear'
    elif 'random_forest' in args.experiment.lower() or 'rf' in args.experiment.lower():
        config['model']['architecture'] = 'random_forest'
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("\n📂 Loading unified dataset...")
    dataset_path = Path(config['data']['data_paths']['unified_dataset'])
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        logger.error("Please run preprocessing first: python run_preprocessing.py")
        return 1
    
    dataset = xr.open_dataset(dataset_path)
    logger.info(f"Loaded dataset: {dict(dataset.dims)}")
    
    # Prepare data for ML
    logger.info("\n🔄 Preparing data for machine learning...")
    X, y, split_indices = prepare_ml_data(dataset, config)
    
    # Handle Random Forest separately (sklearn model)
    if config['model']['architecture'] == 'random_forest':
        logger.info("\n🌲 Training Random Forest baseline...")
        from src.model import RandomForestBaseline
        
        model = RandomForestBaseline(config)
        
        # Train on training data
        X_train = X[split_indices['train']]
        y_train = y[split_indices['train']]
        
        logger.info(f"Training on {len(X_train)} samples...")
        model.fit(X_train, y_train)
        
        # Evaluate
        X_test = X[split_indices['test']]
        y_test = y[split_indices['test']]
        
        predictions = model.predict(X_test)
        
        # Calculate metrics
        from src.evaluate import calculate_rmse, calculate_mae, calculate_correlation
        metrics = {
            'rmse': calculate_rmse(predictions, y_test),
            'mae': calculate_mae(predictions, y_test),
            'correlation': calculate_correlation(predictions, y_test)
        }
        
        logger.info(f"\n✅ Random Forest Results:")
        logger.info(f"   RMSE: {metrics['rmse']:.4f}")
        logger.info(f"   MAE: {metrics['mae']:.4f}")
        logger.info(f"   Correlation: {metrics['correlation']:.4f}")
        
        # Save results
        results_table = create_results_table(metrics, "Random Forest")
        results_table.to_csv(reports_dir / f'{args.experiment}_results.csv', index=False)
        logger.info(f"Results saved to: {reports_dir / f'{args.experiment}_results.csv'}")
        
        return 0
    
    # Create PyTorch data loaders
    logger.info("\n📊 Creating data loaders...")
    dataloaders = create_data_loaders(
        X, y, split_indices, 
        config['training']['batch_size'], 
        device
    )
    
    for split_name, loader in dataloaders.items():
        logger.info(f"   {split_name}: {len(loader.dataset)} samples, {len(loader)} batches")
    
    # Create model
    logger.info(f"\n🏗️ Creating {config['model']['architecture']} model...")
    model = create_model(config)
    model.to(device)
    
    num_params = count_parameters(model)
    logger.info(f"Model parameters: {num_params:,}")
    
    # Training or evaluation mode
    if args.test_only:
        logger.info("\n🧪 Running evaluation only...")
        
        # Load best model
        best_model_path = models_dir / f'{args.experiment}_best.pth'
        if not best_model_path.exists():
            logger.error(f"Model not found: {best_model_path}")
            return 1
        
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from: {best_model_path}")
        
        history = checkpoint.get('history', {})
        
    else:
        logger.info("\n🚀 Starting training...")
        
        # Train model
        history = train_model(
            model, dataloaders, config, device, logger
        )
        
        # Save best model
        best_model_path = models_dir / f'{args.experiment}_best.pth'
        save_checkpoint(
            model, None, config['training']['epochs'],
            history, config, best_model_path
        )
    
    # Evaluation
    logger.info("\n📈 Evaluating model...")
    
    X_test = X[split_indices['test']]
    y_test = y[split_indices['test']]
    
    metrics = evaluate_model(
        model, (X_test, y_test), config, device
    )
    
    # Create and save results
    results_table = create_results_table(metrics, config['model']['architecture'].upper())
    print("\n" + "=" * 70)
    print("FINAL RESULTS:")
    print("=" * 70)
    print(results_table.to_string(index=False))
    
    results_table.to_csv(reports_dir / f'{args.experiment}_results.csv', index=False)
    
    # Plot training history
    if history and not args.test_only:
        logger.info("\n📊 Plotting training history...")
        plot_training_history(history, reports_dir / 'figures')
    
    # Create comparison plots
    logger.info("\n🎨 Creating comparison plots...")
    
    # Get sample predictions for visualization
    model.eval()
    with torch.no_grad():
        # Take first test sample
        X_sample = torch.FloatTensor(X_test[0:1])
        if len(X_sample.shape) == 3:
            X_sample = X_sample.unsqueeze(1)
        
        X_sample = X_sample.to(device)
        pred_sample = model(X_sample).cpu().numpy()
    
    # Plot comparison
    if len(pred_sample.shape) == 4:
        pred_sample = pred_sample[0, 0]
    
    plot_comparison_maps(
        X_test[0] if len(X_test.shape) == 3 else X_test[0, 0],
        pred_sample,
        y_test[0] if len(y_test.shape) == 3 else y_test[0, 0],
        title=f"{config['model']['architecture'].upper()} Super-Resolution",
        output_path=reports_dir / 'figures' / f'{args.experiment}_comparison.png'
    )
    
    # Plot residuals
    logger.info("\n📉 Plotting residual analysis...")
    plot_residuals(
        pred_sample,
        y_test[0] if len(y_test.shape) == 3 else y_test[0, 0],
        output_path=reports_dir / 'figures' / f'{args.experiment}_residuals.png'
    )
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"✅ Model: {config['model']['architecture']}")
    logger.info(f"✅ Best model saved: {best_model_path}")
    logger.info(f"✅ Results saved: {reports_dir / f'{args.experiment}_results.csv'}")
    logger.info(f"✅ Test RMSE: {metrics['rmse']:.4f}")
    logger.info(f"✅ Test Correlation: {metrics['correlation']:.4f}")
    logger.info(f"✅ Back-aggregation: {'PASSED' if metrics.get('back_aggregation_passed', False) else 'FAILED'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())