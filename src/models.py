"""Model implementations for FWI downscaling"""

import numpy as np
from scipy import interpolate
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from typing import Dict, Any, Optional, Tuple
import logging

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


class BaseDownscaler:
    """Base class for all downscaling models"""
    
    def __init__(self, upscale_factor: int = 4):
        self.upscale_factor = upscale_factor
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        raise NotImplementedError
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        return self.predict(X)


class BilinearInterpolation(BaseDownscaler):
    """Method 1: Traditional bilinear interpolation"""
    
    def __init__(self, upscale_factor: int = 4):
        super().__init__(upscale_factor)
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Upscale using bilinear interpolation"""
        
        if X.ndim == 2:
            X = X.reshape(1, X.shape[0], X.shape[1])
            
        n_time, height, width = X.shape
        new_height = height * self.upscale_factor
        new_width = width * self.upscale_factor
        
        result = np.zeros((n_time, new_height, new_width))
        
        for t in range(n_time):
            old_y = np.linspace(0, 1, height)
            old_x = np.linspace(0, 1, width)
            new_y = np.linspace(0, 1, new_height)
            new_x = np.linspace(0, 1, new_width)
            
            f = interpolate.interp2d(old_x, old_y, X[t], kind='linear')
            result[t] = f(new_x, new_y)
            
        return result if n_time > 1 else result[0]


class MLTerrainDownscaler(BaseDownscaler):
    """Method 2: Machine Learning with terrain features"""
    
    def __init__(
        self, 
        upscale_factor: int = 4,
        model_type: str = 'random_forest',
        **model_kwargs
    ):
        super().__init__(upscale_factor)
        
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=model_kwargs.get('n_estimators', 100),
                max_depth=model_kwargs.get('max_depth', 10),
                random_state=model_kwargs.get('random_state', 42)
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=model_kwargs.get('n_estimators', 100),
                max_depth=model_kwargs.get('max_depth', 5),
                learning_rate=model_kwargs.get('learning_rate', 0.1),
                random_state=model_kwargs.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        self.interpolator = BilinearInterpolation(upscale_factor)
        
    def _create_features(self, X: np.ndarray, terrain: Optional[np.ndarray] = None) -> np.ndarray:
        """Create features including spatial context and terrain"""
        
        # Use interpolated values as base features
        interp = self.interpolator.predict(X)
        features = interp.flatten().reshape(-1, 1)
        
        # Add original low-res features
        X_flat = X.flatten().reshape(-1, 1)
        
        # Repeat low-res features to match high-res size
        upscale_factor_sq = self.upscale_factor ** 2
        X_repeated = np.repeat(X_flat, upscale_factor_sq, axis=0)
        
        features = np.column_stack([features.flatten(), X_repeated.flatten()])
        
        return features
        
    def fit(self, X: np.ndarray, y: np.ndarray, terrain: Optional[np.ndarray] = None):
        """Fit the ML model"""
        
        features = self._create_features(X, terrain)
        target = y.flatten() if y.ndim > 1 else y
        
        self.model.fit(features, target)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray, terrain: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict high-resolution output"""
        
        if not self.is_fitted:
            logger.warning("Model not fitted, using interpolation only")
            return self.interpolator.predict(X)
            
        features = self._create_features(X, terrain)
        predictions = self.model.predict(features)
        
        if X.ndim == 3:
            n_time, h, w = X.shape
            new_h = h * self.upscale_factor
            new_w = w * self.upscale_factor
            return predictions.reshape(n_time, new_h, new_w)
        else:
            h, w = X.shape
            new_h = h * self.upscale_factor
            new_w = w * self.upscale_factor
            return predictions.reshape(new_h, new_w)


class PhysicsMLDownscaler(BaseDownscaler):
    """Method 3: Physics-informed machine learning"""
    
    def __init__(
        self,
        upscale_factor: int = 4,
        physics_weight: float = 0.5,
        **model_kwargs
    ):
        super().__init__(upscale_factor)
        self.physics_weight = physics_weight
        self.ml_model = MLTerrainDownscaler(upscale_factor, **model_kwargs)
        self.interpolator = BilinearInterpolation(upscale_factor)
        
    def _apply_physics_constraints(self, predictions: np.ndarray) -> np.ndarray:
        """Apply physical constraints to predictions"""
        
        predictions = np.maximum(predictions, 0)
        
        if predictions.ndim == 3:
            for t in range(1, predictions.shape[0]):
                max_change = 50.0
                change = predictions[t] - predictions[t-1]
                change = np.clip(change, -max_change, max_change)
                predictions[t] = predictions[t-1] + change
                
        return predictions
        
    def fit(self, X: np.ndarray, y: np.ndarray, atmospheric: Optional[np.ndarray] = None):
        """Fit the physics-informed model"""
        
        self.ml_model.fit(X, y, atmospheric)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray, atmospheric: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict with physics constraints"""
        
        ml_pred = self.ml_model.predict(X, atmospheric)
        
        physics_pred = self.interpolator.predict(X)
        
        combined = (self.physics_weight * physics_pred + 
                   (1 - self.physics_weight) * ml_pred)
        
        return self._apply_physics_constraints(combined)


class XGBoostDownscaler(BaseDownscaler):
    """XGBoost-based downscaling from your ML experiments"""
    
    def __init__(self, upscale_factor: int = 4, **xgb_params):
        super().__init__(upscale_factor)
        
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Run: uv add xgboost")
            
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'objective': 'reg:squarederror',
            'random_state': 42
        }
        default_params.update(xgb_params)
        
        self.model = xgb.XGBRegressor(**default_params)
        self.interpolator = BilinearInterpolation(upscale_factor)
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit XGBoost model"""
        
        X_interp = self.interpolator.predict(X)
        features = X_interp.flatten()
        target = y.flatten() if y is not None else features
        
        self.model.fit(features.reshape(-1, 1), target)
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using XGBoost"""
        
        if not self.is_fitted:
            return self.interpolator.predict(X)
            
        X_interp = self.interpolator.predict(X)
        features = X_interp.flatten().reshape(-1, 1)
        predictions = self.model.predict(features)
        
        original_shape = X_interp.shape
        return predictions.reshape(original_shape)


class EnsembleDownscaler(BaseDownscaler):
    """Ensemble of multiple models from your 2017 experiments"""
    
    def __init__(self, upscale_factor: int = 4, models: Optional[list] = None):
        super().__init__(upscale_factor)
        
        if models is None:
            models = [
                ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42))
            ]
            
            if HAS_XGBOOST:
                models.append(('xgb', xgb.XGBRegressor(n_estimators=100, max_depth=6)))
                
        self.ensemble = VotingRegressor(models)
        self.interpolator = BilinearInterpolation(upscale_factor)
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit ensemble model"""
        
        X_interp = self.interpolator.predict(X)
        features = self._create_feature_matrix(X, X_interp)
        target = y.flatten() if y is not None else X_interp.flatten()
        
        self.ensemble.fit(features, target)
        self.is_fitted = True
        return self
        
    def _create_feature_matrix(self, X_low: np.ndarray, X_high: np.ndarray) -> np.ndarray:
        """Create feature matrix with multiple resolutions"""
        
        # Use high-resolution interpolated features as base
        features = X_high.flatten().reshape(-1, 1)
        
        # Add low-resolution features repeated to match high-res size
        upscale_factor_sq = self.upscale_factor ** 2
        X_low_repeated = np.repeat(X_low.flatten(), upscale_factor_sq)
        
        return np.column_stack([features.flatten(), X_low_repeated])
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using ensemble"""
        
        if not self.is_fitted:
            return self.interpolator.predict(X)
            
        X_interp = self.interpolator.predict(X)
        features = self._create_feature_matrix(X, X_interp)
        predictions = self.ensemble.predict(features)
        
        return predictions.reshape(X_interp.shape)


if HAS_TORCH:
    class CNNDownscaler(nn.Module, BaseDownscaler):
        """CNN-based downscaling from your deep learning experiments"""
        
        def __init__(self, input_channels: int = 1, upscale_factor: int = 4):
            nn.Module.__init__(self)
            BaseDownscaler.__init__(self, upscale_factor)
            
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            
            self.upsampler = nn.Sequential(
                nn.Conv2d(32, 32 * (upscale_factor ** 2), kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor),
                nn.Conv2d(32, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            
        def forward(self, x):
            features = self.feature_extractor(x)
            output = self.upsampler(features)
            return output


def get_model(method: str, config: Dict[str, Any]) -> BaseDownscaler:
    """Factory function to get model by method name"""
    
    models = {
        'interpolation': BilinearInterpolation,
        'ml_terrain': MLTerrainDownscaler,
        'physics_ml': PhysicsMLDownscaler,
        'xgboost': XGBoostDownscaler,
        'ensemble': EnsembleDownscaler,
    }
    
    if HAS_TORCH:
        models['cnn'] = CNNDownscaler
    
    if method not in models:
        raise ValueError(f"Unknown method: {method}. Available: {list(models.keys())}")
        
    return models[method](**config)