#!/usr/bin/env python3
"""
ML-Based FWI Downscaling: 25km → 10km

This script implements machine learning approaches for downscaling 
Fire Weather Index from 25km to 10km resolution.
"""

import numpy as np
import xarray as xr
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class FWIDownscalingDataset(Dataset):
    """Dataset for FWI downscaling training"""
    
    def __init__(self, coarse_fwi, fine_target, auxiliary_data=None):
        """
        Parameters:
        -----------
        coarse_fwi : xarray.DataArray
            25km FWI data (input)
        fine_target : xarray.DataArray  
            10km FWI data (target from formula)
        auxiliary_data : dict of xarray.DataArray
            Additional high-resolution features
        """
        self.coarse_fwi = coarse_fwi
        self.fine_target = fine_target
        self.auxiliary_data = auxiliary_data or {}
        
    def __len__(self):
        return len(self.coarse_fwi.time)
    
    def __getitem__(self, idx):
        # Get daily data
        coarse_day = self.coarse_fwi.isel(time=idx).values
        target_day = self.fine_target.isel(time=idx).values
        
        # Add auxiliary features if available
        features = [coarse_day]
        for key, data in self.auxiliary_data.items():
            aux_day = data.isel(time=idx).values if 'time' in data.dims else data.values
            features.append(aux_day)
        
        # Stack features along channel dimension
        input_tensor = torch.FloatTensor(np.stack(features, axis=0))
        target_tensor = torch.FloatTensor(target_day)
        
        return input_tensor, target_tensor

class CNNDownscaler(nn.Module):
    """CNN-based super-resolution model for FWI downscaling"""
    
    def __init__(self, input_channels=1, upscale_factor=2.5):
        """
        Parameters:
        -----------
        input_channels : int
            Number of input feature channels
        upscale_factor : float
            Spatial upscaling factor (25km → 10km ≈ 2.5x)
        """
        super(CNNDownscaler, self).__init__()
        
        self.upscale_factor = upscale_factor
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling layers
        self.upsampler = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=upscale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)  # Ensure non-negative FWI
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.upsampler(features)
        return output

class XGBoostDownscaler:
    """XGBoost-based regression model for FWI downscaling"""
    
    def __init__(self, n_estimators=100, max_depth=6):
        """
        Parameters:
        -----------
        n_estimators : int
            Number of boosting rounds
        max_depth : int
            Maximum tree depth
        """
        from xgboost import XGBRegressor
        
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            random_state=42
        )
        
    def prepare_features(self, coarse_fwi, auxiliary_data, target_coords):
        """
        Prepare point-wise features for XGBoost training
        
        Parameters:
        -----------
        coarse_fwi : xarray.DataArray
            25km FWI data
        auxiliary_data : dict
            Additional feature arrays
        target_coords : tuple
            Target grid coordinates
            
        Returns:
        --------
        features : pandas.DataFrame
            Feature matrix for training
        """
        features_list = []
        
        # Get target grid coordinates
        target_lons, target_lats = target_coords
        
        for t in range(len(coarse_fwi.time)):
            for i, lat in enumerate(target_lats):
                for j, lon in enumerate(target_lons):
                    
                    # Base features
                    row = {
                        'time_idx': t,
                        'lat': lat,
                        'lon': lon,
                        'lat_idx': i,
                        'lon_idx': j
                    }
                    
                    # Interpolate coarse FWI to target location
                    coarse_value = coarse_fwi.isel(time=t).interp(
                        latitude=lat, longitude=lon, method='linear'
                    ).values
                    row['coarse_fwi'] = float(coarse_value)
                    
                    # Add auxiliary features
                    for key, data in auxiliary_data.items():
                        if 'time' in data.dims:
                            aux_value = data.isel(time=t).interp(
                                latitude=lat, longitude=lon, method='linear'
                            ).values
                        else:
                            aux_value = data.interp(
                                latitude=lat, longitude=lon, method='linear'  
                            ).values
                        row[f'aux_{key}'] = float(aux_value)
                    
                    # Spatial context features (neighboring coarse values)
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            try:
                                neighbor_lat = lat + di * 0.25  # 25km ≈ 0.25°
                                neighbor_lon = lon + dj * 0.25
                                neighbor_value = coarse_fwi.isel(time=t).interp(
                                    latitude=neighbor_lat, longitude=neighbor_lon, 
                                    method='linear'
                                ).values
                                row[f'neighbor_{di}_{dj}'] = float(neighbor_value)
                            except:
                                row[f'neighbor_{di}_{dj}'] = row['coarse_fwi']
                    
                    features_list.append(row)
        
        return pd.DataFrame(features_list)
    
    def train(self, features_df, target_values):
        """
        Train XGBoost model
        
        Parameters:
        -----------
        features_df : pandas.DataFrame
            Feature matrix
        target_values : array
            Target FWI values (10km)
        """
        # Remove non-feature columns
        feature_cols = [col for col in features_df.columns 
                       if col not in ['time_idx', 'lat', 'lon', 'lat_idx', 'lon_idx']]
        
        X = features_df[feature_cols]
        y = target_values
        
        # Remove NaN values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]
        
        print(f"Training on {len(X_clean)} samples with {len(feature_cols)} features")
        
        # Train model
        self.model.fit(X_clean, y_clean)
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 10 most important features:")
        print(importance.head(10))
        
        return self.model
    
    def predict(self, features_df):
        """Predict 10km FWI values"""
        feature_cols = [col for col in features_df.columns 
                       if col not in ['time_idx', 'lat', 'lon', 'lat_idx', 'lon_idx']]
        
        X = features_df[feature_cols]
        predictions = self.model.predict(X)
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions

class HybridDownscaler:
    """
    Hybrid approach combining physical formula with ML correction
    """
    
    def __init__(self):
        """Initialize hybrid downscaler"""
        self.physical_model = None  # Would use FWI calculator
        self.ml_corrector = RandomForestRegressor(n_estimators=50, random_state=42)
        
    def train(self, coarse_fwi, era5_land_data, target_fwi):
        """
        Train hybrid model
        
        1. Calculate physical baseline using formula
        2. Train ML model to predict residuals
        """
        print("Training hybrid physical + ML model...")
        
        # Step 1: Calculate physical baseline (simplified)
        # In practice, would use full FWI calculator
        physical_baseline = self.calculate_physical_baseline(era5_land_data)
        
        # Step 2: Calculate residuals
        residuals = target_fwi - physical_baseline
        
        # Step 3: Train ML model to predict residuals
        features = self.extract_residual_features(coarse_fwi, era5_land_data)
        self.ml_corrector.fit(features, residuals.flatten())
        
        print("Hybrid model training complete")
        
    def calculate_physical_baseline(self, era5_land_data):
        """Calculate FWI using physical formula (simplified)"""
        # Simplified - would use full Canadian FWI calculator
        temp = era5_land_data['t2m'] - 273.15
        # ... other calculations
        baseline = temp * 0.5  # Placeholder
        return baseline
        
    def extract_residual_features(self, coarse_fwi, era5_land_data):
        """Extract features for residual prediction"""
        # Simplified feature extraction
        features = np.stack([
            coarse_fwi.values.flatten(),
            era5_land_data['t2m'].values.flatten(),
            # ... other features
        ], axis=1)
        return features
        
    def predict(self, coarse_fwi, era5_land_data):
        """Predict using hybrid approach"""
        # Physical baseline
        physical = self.calculate_physical_baseline(era5_land_data)
        
        # ML correction
        features = self.extract_residual_features(coarse_fwi, era5_land_data)
        correction = self.ml_corrector.predict(features)
        correction = correction.reshape(physical.shape)
        
        # Combined prediction
        prediction = physical + correction
        
        # Ensure non-negative
        prediction = np.maximum(prediction, 0)
        
        return prediction

def train_fwi_downscaling_model(model_type='cnn'):
    """
    Main training function for FWI downscaling
    
    Parameters:
    -----------
    model_type : str
        'cnn', 'xgboost', or 'hybrid'
    """
    print(f"=== Training {model_type.upper()} FWI Downscaling Model ===")
    
    # Load data (placeholder - would load actual data)
    print("Loading training data...")
    print("- 25km FWI (ERA5)")
    print("- 10km FWI target (formula-calculated)")
    print("- 10km auxiliary data (ERA5-Land)")
    
    if model_type == 'cnn':
        model = CNNDownscaler(input_channels=5)  # FWI + 4 auxiliary channels
        print("CNN model initialized")
        
    elif model_type == 'xgboost':
        model = XGBoostDownscaler()
        print("XGBoost model initialized")
        
    elif model_type == 'hybrid':
        model = HybridDownscaler()
        print("Hybrid model initialized")
    
    print(f"\nTraining {model_type} model...")
    print("This would train on the actual data...")
    
    return model

if __name__ == "__main__":
    print("FWI Downscaling ML Pipeline")
    print("="*30)
    print("Available models:")
    print("1. CNN Super-Resolution")
    print("2. XGBoost Regression") 
    print("3. Hybrid Physical+ML")
    print("\nNext: Load data and train models")
    
    # Example usage
    # model = train_fwi_downscaling_model('cnn')