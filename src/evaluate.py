"""Evaluation metrics and validation for FWI downscaling"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluation metrics for downscaling validation"""
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Pearson correlation coefficient"""
        return np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    
    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared score"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def conservation_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Conservation error - how well the total FWI is preserved
        Important for physical consistency
        """
        true_sum = np.sum(y_true)
        pred_sum = np.sum(y_pred)
        return abs(true_sum - pred_sum) / true_sum * 100
    
    @staticmethod
    def spatial_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Spatial pattern correlation"""
        if y_true.ndim == 3:
            correlations = []
            for t in range(y_true.shape[0]):
                corr = np.corrcoef(y_true[t].flatten(), y_pred[t].flatten())[0, 1]
                correlations.append(corr)
            return np.mean(correlations)
        else:
            return np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    
    def evaluate(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute all metrics"""
        
        metrics = {
            'rmse': self.rmse(y_true, y_pred),
            'mae': self.mae(y_true, y_pred),
            'correlation': self.correlation(y_true, y_pred),
            'r2': self.r2_score(y_true, y_pred),
            'conservation_error': self.conservation_error(y_true, y_pred),
            'spatial_correlation': self.spatial_correlation(y_true, y_pred)
        }
        
        return metrics


class BackAggregationValidator:
    """
    Validation through back-aggregation
    Key validation method: downscale then aggregate back to original resolution
    """
    
    def __init__(self, upscale_factor: int = 4):
        self.upscale_factor = upscale_factor
        self.evaluator = Evaluator()
        
    def aggregate(self, high_res: np.ndarray) -> np.ndarray:
        """Aggregate high-res back to low-res"""
        
        factor = self.upscale_factor
        
        if high_res.ndim == 3:
            t, h, w = high_res.shape
            new_h = h // factor
            new_w = w // factor
            
            aggregated = np.zeros((t, new_h, new_w))
            
            for i in range(new_h):
                for j in range(new_w):
                    patch = high_res[:, 
                                   i*factor:(i+1)*factor,
                                   j*factor:(j+1)*factor]
                    aggregated[:, i, j] = np.mean(patch, axis=(1, 2))
                    
        elif high_res.ndim == 2:
            h, w = high_res.shape
            new_h = h // factor
            new_w = w // factor
            
            aggregated = np.zeros((new_h, new_w))
            
            for i in range(new_h):
                for j in range(new_w):
                    patch = high_res[i*factor:(i+1)*factor,
                                   j*factor:(j+1)*factor]
                    aggregated[i, j] = np.mean(patch)
        else:
            raise ValueError(f"Expected 2D or 3D array, got {high_res.ndim}D")
            
        return aggregated
    
    def validate(
        self, 
        original_low_res: np.ndarray,
        predicted_high_res: np.ndarray
    ) -> Dict[str, float]:
        """
        Validate by comparing original with back-aggregated predictions
        
        This is the key validation: if the model is good, aggregating the
        high-res predictions should match the original low-res data
        """
        
        aggregated = self.aggregate(predicted_high_res)
        
        if original_low_res.shape != aggregated.shape:
            logger.error(f"Shape mismatch: {original_low_res.shape} vs {aggregated.shape}")
            return {}
            
        metrics = self.evaluator.evaluate(original_low_res, aggregated)
        
        logger.info("Back-aggregation validation results:")
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.4f}")
            
        return metrics
    
    def plot_comparison(
        self,
        original_low_res: np.ndarray,
        predicted_high_res: np.ndarray,
        time_step: int = 0,
        save_path: Optional[str] = None
    ):
        """Plot comparison of original, predicted, and back-aggregated"""
        
        aggregated = self.aggregate(predicted_high_res)
        
        if original_low_res.ndim == 3:
            original = original_low_res[time_step]
            predicted = predicted_high_res[time_step]
            agg = aggregated[time_step]
        else:
            original = original_low_res
            predicted = predicted_high_res
            agg = aggregated
            
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        im1 = axes[0].imshow(original, cmap='YlOrRd', aspect='auto')
        axes[0].set_title('Original (Low-res)')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(predicted, cmap='YlOrRd', aspect='auto')
        axes[1].set_title('Predicted (High-res)')
        plt.colorbar(im2, ax=axes[1])
        
        im3 = axes[2].imshow(agg, cmap='YlOrRd', aspect='auto')
        axes[2].set_title('Back-aggregated')
        plt.colorbar(im3, ax=axes[2])
        
        diff = original - agg
        im4 = axes[3].imshow(diff, cmap='RdBu_r', aspect='auto')
        axes[3].set_title('Difference (Original - Aggregated)')
        plt.colorbar(im4, ax=axes[3])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
            
        return fig