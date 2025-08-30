#!/usr/bin/env python3
"""
Evaluation metrics for FWI super-resolution model.
Primary metric: Back-aggregation correlation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union


def calculate_back_aggregation_correlation(
    y_hat_high_res: torch.Tensor,
    x_low_res: torch.Tensor,
    aggregation_factor: Optional[int] = None
) -> float:
    """
    Calculate the back-aggregation correlation coefficient.
    
    This metric measures how well the model preserves large-scale patterns
    by aggregating the high-resolution output back to low resolution
    and comparing with the original low-resolution input.
    
    Args:
        y_hat_high_res: Model's high-resolution output (B, C, H_high, W_high)
        x_low_res: Original low-resolution input (B, C, H_low, W_low)
        aggregation_factor: Factor to downsample high-res to low-res.
                          If None, automatically calculated from tensor shapes.
    
    Returns:
        float: Pearson correlation coefficient between aggregated output 
               and original low-res input (range: -1 to 1)
    
    Example:
        >>> high_res = torch.ones(1, 1, 100, 100) * 5  # 100x100 all 5s
        >>> low_res = torch.ones(1, 1, 20, 20) * 5     # 20x20 all 5s
        >>> corr = calculate_back_aggregation_correlation(high_res, low_res)
        >>> print(f"Correlation: {corr:.4f}")  # Should be 1.0
    """
    # Ensure tensors are on the same device
    if y_hat_high_res.device != x_low_res.device:
        x_low_res = x_low_res.to(y_hat_high_res.device)
    
    # Get dimensions
    b, c, h_high, w_high = y_hat_high_res.shape
    _, _, h_low, w_low = x_low_res.shape
    
    # Calculate aggregation factor if not provided
    if aggregation_factor is None:
        aggregation_factor_h = h_high // h_low
        aggregation_factor_w = w_high // w_low
        
        # Ensure uniform downsampling
        if aggregation_factor_h != aggregation_factor_w:
            raise ValueError(
                f"Non-uniform aggregation factors: "
                f"height {aggregation_factor_h} != width {aggregation_factor_w}"
            )
        
        aggregation_factor = aggregation_factor_h
    
    # Validate that dimensions are compatible
    if h_high % aggregation_factor != 0 or w_high % aggregation_factor != 0:
        raise ValueError(
            f"High-res dimensions ({h_high}, {w_high}) not divisible by "
            f"aggregation factor {aggregation_factor}"
        )
    
    # Expected low-res dimensions after aggregation
    expected_h_low = h_high // aggregation_factor
    expected_w_low = w_high // aggregation_factor
    
    if expected_h_low != h_low or expected_w_low != w_low:
        raise ValueError(
            f"Dimension mismatch: aggregated shape ({expected_h_low}, {expected_w_low}) "
            f"!= low-res shape ({h_low}, {w_low})"
        )
    
    # Perform 2D average pooling to aggregate high-res to low-res
    y_hat_aggregated = F.avg_pool2d(
        y_hat_high_res,
        kernel_size=aggregation_factor,
        stride=aggregation_factor
    )
    
    # Flatten tensors for correlation calculation
    y_hat_flat = y_hat_aggregated.flatten()
    x_low_flat = x_low_res.flatten()
    
    # Calculate Pearson correlation coefficient
    # Using torch operations for GPU compatibility
    
    # Remove mean (centering)
    y_hat_centered = y_hat_flat - y_hat_flat.mean()
    x_low_centered = x_low_flat - x_low_flat.mean()
    
    # Calculate correlation
    numerator = (y_hat_centered * x_low_centered).sum()
    
    # Standard deviations
    y_hat_std = torch.sqrt((y_hat_centered ** 2).sum())
    x_low_std = torch.sqrt((x_low_centered ** 2).sum())
    
    # Handle edge case where std is zero
    if y_hat_std == 0 or x_low_std == 0:
        # If either tensor has no variance, correlation is undefined
        # Return 0 as a reasonable default
        return 0.0
    
    correlation = numerator / (y_hat_std * x_low_std)
    
    # Convert to Python float and ensure it's in valid range
    correlation_value = float(correlation.cpu().item())
    
    # Numerical stability: clamp to valid correlation range
    correlation_value = max(-1.0, min(1.0, correlation_value))
    
    return correlation_value


def calculate_mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculate Mean Squared Error between predictions and ground truth.
    
    Args:
        y_pred: Predicted values
        y_true: Ground truth values
    
    Returns:
        float: Mean squared error
    """
    mse = F.mse_loss(y_pred, y_true)
    return float(mse.cpu().item())


def calculate_rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculate Root Mean Squared Error between predictions and ground truth.
    
    Args:
        y_pred: Predicted values
        y_true: Ground truth values
    
    Returns:
        float: Root mean squared error
    """
    mse = calculate_mse(y_pred, y_true)
    return np.sqrt(mse)


def calculate_mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculate Mean Absolute Error between predictions and ground truth.
    
    Args:
        y_pred: Predicted values
        y_true: Ground truth values
    
    Returns:
        float: Mean absolute error
    """
    mae = F.l1_loss(y_pred, y_true)
    return float(mae.cpu().item())


def calculate_pearson_correlation(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculate Pearson correlation coefficient between predictions and ground truth.
    
    Args:
        y_pred: Predicted values
        y_true: Ground truth values
    
    Returns:
        float: Pearson correlation coefficient
    """
    # Flatten tensors
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()
    
    # Center the data
    y_pred_centered = y_pred_flat - y_pred_flat.mean()
    y_true_centered = y_true_flat - y_true_flat.mean()
    
    # Calculate correlation
    numerator = (y_pred_centered * y_true_centered).sum()
    denominator = torch.sqrt((y_pred_centered ** 2).sum() * (y_true_centered ** 2).sum())
    
    if denominator == 0:
        return 0.0
    
    correlation = numerator / denominator
    return float(correlation.cpu().item())


class EvaluationMetrics:
    """
    Container class for all evaluation metrics.
    """
    
    def __init__(self, aggregation_factor: int = 25):
        """
        Initialize evaluation metrics.
        
        Args:
            aggregation_factor: Factor for back-aggregation (default: 25 for 25km->1km)
        """
        self.aggregation_factor = aggregation_factor
    
    def compute_all_metrics(
        self,
        y_pred_high_res: torch.Tensor,
        y_true_high_res: torch.Tensor,
        x_low_res: torch.Tensor
    ) -> dict:
        """
        Compute all evaluation metrics.
        
        Args:
            y_pred_high_res: Model's high-resolution predictions
            y_true_high_res: Ground truth high-resolution data
            x_low_res: Original low-resolution input
        
        Returns:
            dict: Dictionary containing all metric values
        """
        metrics = {
            'mse': calculate_mse(y_pred_high_res, y_true_high_res),
            'rmse': calculate_rmse(y_pred_high_res, y_true_high_res),
            'mae': calculate_mae(y_pred_high_res, y_true_high_res),
            'pearson_correlation': calculate_pearson_correlation(y_pred_high_res, y_true_high_res),
            'back_aggregation_correlation': calculate_back_aggregation_correlation(
                y_pred_high_res, x_low_res, self.aggregation_factor
            )
        }
        
        return metrics