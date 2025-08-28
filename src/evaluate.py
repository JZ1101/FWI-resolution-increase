#!/usr/bin/env python3
"""
Evaluation module for FWI super-resolution models

Contains metrics, visualization, and validation functions.
"""

import torch
import torch.nn as nn
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd


def calculate_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate Root Mean Square Error"""
    return np.sqrt(mean_squared_error(targets.flatten(), predictions.flatten()))


def calculate_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate Mean Absolute Error"""
    return mean_absolute_error(targets.flatten(), predictions.flatten())


def calculate_correlation(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate spatial correlation"""
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # Remove NaN values
    mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
    pred_flat = pred_flat[mask]
    target_flat = target_flat[mask]
    
    if len(pred_flat) > 0:
        correlation, _ = stats.pearsonr(pred_flat, target_flat)
        return correlation
    return 0.0


def calculate_conservation_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculate mass conservation error.
    
    The total FWI should be conserved when aggregating from high to low resolution.
    """
    total_pred = np.sum(predictions)
    total_target = np.sum(targets)
    
    if total_target != 0:
        conservation_error = abs(total_pred - total_target) / total_target
        return conservation_error
    return 0.0


def back_aggregation_test(
    high_res_predictions: np.ndarray,
    low_res_targets: np.ndarray,
    scale_factor: int = 4
) -> Dict[str, float]:
    """
    Perform back-aggregation validation.
    
    High-resolution predictions should aggregate back to match low-resolution inputs.
    
    Args:
        high_res_predictions: High-resolution model predictions
        low_res_targets: Original low-resolution targets
        scale_factor: Downscaling factor
        
    Returns:
        Dictionary with validation metrics
    """
    # Aggregate high-res predictions back to low-res
    aggregated = aggregate_to_low_res(high_res_predictions, scale_factor)
    
    # Calculate metrics
    correlation = calculate_correlation(aggregated, low_res_targets)
    conservation_error = calculate_conservation_error(aggregated, low_res_targets)
    rmse = calculate_rmse(aggregated, low_res_targets)
    
    results = {
        'correlation': correlation,
        'conservation_error': conservation_error,
        'rmse': rmse,
        'passed': correlation > 0.6 and conservation_error < 0.3
    }
    
    return results


def aggregate_to_low_res(high_res: np.ndarray, scale_factor: int) -> np.ndarray:
    """
    Aggregate high-resolution data to low-resolution by averaging.
    
    Args:
        high_res: High-resolution array
        scale_factor: Downscaling factor
        
    Returns:
        Low-resolution aggregated array
    """
    if len(high_res.shape) == 4:  # (batch, channels, height, width)
        batch, channels, height, width = high_res.shape
        new_height = height // scale_factor
        new_width = width // scale_factor
        
        # Reshape and average
        aggregated = high_res.reshape(
            batch, channels, 
            new_height, scale_factor,
            new_width, scale_factor
        ).mean(axis=(3, 5))
        
    elif len(high_res.shape) == 3:  # (time, height, width)
        time, height, width = high_res.shape
        new_height = height // scale_factor
        new_width = width // scale_factor
        
        aggregated = high_res.reshape(
            time,
            new_height, scale_factor,
            new_width, scale_factor
        ).mean(axis=(2, 4))
        
    else:
        # Simple 2D case
        height, width = high_res.shape
        new_height = height // scale_factor
        new_width = width // scale_factor
        
        aggregated = high_res.reshape(
            new_height, scale_factor,
            new_width, scale_factor
        ).mean(axis=(1, 3))
    
    return aggregated


def evaluate_fire_event(
    predictions: xr.Dataset,
    config: Dict,
    fire_date: str = "2017-06-17",
    fire_coords: Dict[str, float] = {'lat': 39.95, 'lon': -8.13}
) -> Dict:
    """
    Evaluate model performance for the Pedrógão Grande fire event.
    
    Args:
        predictions: Model predictions as xarray dataset
        config: Configuration dictionary
        fire_date: Date of fire event
        fire_coords: Coordinates of fire location
        
    Returns:
        Dictionary with fire detection results
    """
    try:
        # Extract FWI at fire location and date
        fire_fwi = predictions['fwi'].sel(
            time=fire_date,
            latitude=fire_coords['lat'],
            longitude=fire_coords['lon'],
            method='nearest'
        )
        
        fwi_value = float(fire_fwi)
        threshold = config['evaluation']['pedrogao_fire']['min_fwi_threshold']
        
        # Get surrounding area (5x5 grid)
        surrounding_fwi = predictions['fwi'].sel(
            time=fire_date,
            latitude=slice(fire_coords['lat'] - 0.05, fire_coords['lat'] + 0.05),
            longitude=slice(fire_coords['lon'] - 0.05, fire_coords['lon'] + 0.05)
        )
        
        results = {
            'fwi_at_location': fwi_value,
            'threshold': threshold,
            'detected': fwi_value >= threshold,
            'max_fwi_nearby': float(surrounding_fwi.max()),
            'mean_fwi_nearby': float(surrounding_fwi.mean())
        }
        
        return results
        
    except Exception as e:
        return {
            'error': str(e),
            'detected': False
        }


def plot_comparison_maps(
    low_res: np.ndarray,
    high_res_pred: np.ndarray,
    high_res_target: Optional[np.ndarray],
    title: str = "FWI Super-Resolution Comparison",
    output_path: Optional[Path] = None
):
    """
    Create comparison plots of low-res input, prediction, and target.
    
    Args:
        low_res: Low-resolution input
        high_res_pred: High-resolution prediction
        high_res_target: High-resolution ground truth (optional)
        title: Plot title
        output_path: Path to save figure
    """
    # Setup figure
    n_plots = 3 if high_res_target is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    
    if n_plots == 2:
        axes = [axes[0], axes[1], None]
    
    # Common colormap and normalization
    vmin = min(np.min(low_res), np.min(high_res_pred))
    vmax = max(np.max(low_res), np.max(high_res_pred))
    if high_res_target is not None:
        vmin = min(vmin, np.min(high_res_target))
        vmax = max(vmax, np.max(high_res_target))
    
    cmap = 'YlOrRd'  # Fire-appropriate colormap
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    # Plot low-res input
    im1 = axes[0].imshow(low_res, cmap=cmap, norm=norm, interpolation='nearest')
    axes[0].set_title('Low-Res Input (25km)')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Plot high-res prediction
    im2 = axes[1].imshow(high_res_pred, cmap=cmap, norm=norm, interpolation='nearest')
    axes[1].set_title('High-Res Prediction (1km)')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot high-res target if available
    if high_res_target is not None and axes[2] is not None:
        im3 = axes[2].imshow(high_res_target, cmap=cmap, norm=norm, interpolation='nearest')
        axes[2].set_title('High-Res Target (1km)')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved comparison plot: {output_path}")
    
    plt.show()


def create_results_table(
    metrics: Dict[str, float],
    model_name: str = "U-Net"
) -> pd.DataFrame:
    """
    Create a formatted results table.
    
    Args:
        metrics: Dictionary of metric values
        model_name: Name of the model
        
    Returns:
        Pandas DataFrame with results
    """
    # Format metrics for presentation
    results_data = {
        'Model': [model_name],
        'RMSE': [f"{metrics.get('rmse', 0):.4f}"],
        'MAE': [f"{metrics.get('mae', 0):.4f}"],
        'Correlation': [f"{metrics.get('correlation', 0):.4f}"],
        'Conservation Error': [f"{metrics.get('conservation_error', 0):.2%}"],
        'Back-Aggregation': ['✅' if metrics.get('back_aggregation_passed', False) else '❌'],
        'Fire Detection': ['✅' if metrics.get('fire_detected', False) else '❌']
    }
    
    df = pd.DataFrame(results_data)
    return df


def evaluate_model(
    model: nn.Module,
    test_data: Tuple[np.ndarray, np.ndarray],
    config: Dict,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained PyTorch model
        test_data: Tuple of (X_test, y_test)
        config: Configuration dictionary
        device: PyTorch device
        
    Returns:
        Dictionary with all evaluation metrics
    """
    X_test, y_test = test_data
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_test)
    y_tensor = torch.FloatTensor(y_test)
    
    # Add batch dimension if needed
    if len(X_tensor.shape) == 3:
        X_tensor = X_tensor.unsqueeze(0)
        y_tensor = y_tensor.unsqueeze(0)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        X_tensor = X_tensor.to(device)
        predictions = model(X_tensor)
        predictions = predictions.cpu().numpy()
    
    y_test = y_tensor.numpy()
    
    # Calculate metrics
    metrics = {
        'rmse': calculate_rmse(predictions, y_test),
        'mae': calculate_mae(predictions, y_test),
        'correlation': calculate_correlation(predictions, y_test),
        'conservation_error': calculate_conservation_error(predictions, y_test)
    }
    
    # Back-aggregation test
    scale_factor = config['data'].get('upscale_factor', 4)
    back_agg_results = back_aggregation_test(
        predictions[0, 0] if len(predictions.shape) == 4 else predictions,
        X_test[0] if len(X_test.shape) == 3 else X_test,
        scale_factor
    )
    
    metrics['back_aggregation_correlation'] = back_agg_results['correlation']
    metrics['back_aggregation_conservation'] = back_agg_results['conservation_error']
    metrics['back_aggregation_passed'] = back_agg_results['passed']
    
    return metrics


def plot_residuals(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: Optional[Path] = None
):
    """Plot residual analysis"""
    residuals = predictions.flatten() - targets.flatten()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Histogram of residuals
    axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Residual (Pred - Target)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Residual Distribution')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot')
    
    # Residuals vs predictions
    axes[2].scatter(predictions.flatten(), residuals, alpha=0.5, s=1)
    axes[2].set_xlabel('Predicted Values')
    axes[2].set_ylabel('Residuals')
    axes[2].set_title('Residuals vs Predictions')
    axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved residual plot: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test evaluation functions
    print("Evaluation module loaded successfully")
    print("Functions available:")
    print("  - calculate_rmse()")
    print("  - calculate_correlation()")
    print("  - calculate_conservation_error()")
    print("  - back_aggregation_test()")
    print("  - evaluate_fire_event()")
    print("  - plot_comparison_maps()")
    print("  - create_results_table()")
    print("  - evaluate_model()")