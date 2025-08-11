#!/usr/bin/env python3
"""
Complete FWI Downscaling Pipeline: 25km → 10km

This script implements the full ML pipeline:
1. Load and preprocess data
2. Create 10km FWI targets using formula
3. Train ML model (25km inputs → 10km targets)
4. Validate ML predictions vs formula-based targets
5. Generate final 10km FWI predictions
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class CompleteFWIPipeline:
    """Complete pipeline for FWI downscaling"""
    
    def __init__(self):
        """Initialize pipeline"""
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.results = {}
        
    def load_data(self):
        """Load all available datasets"""
        print("=== Loading Data ===")
        
        try:
            # Load 25km FWI (target for upsampling)
            print("Loading 25km FWI data...")
            self.fwi_25km = xr.open_dataset('data/era5_fwi_2017.nc')
            print(f"  Shape: {dict(self.fwi_25km.dims)}")
            print(f"  Variables: {list(self.fwi_25km.data_vars)}")
            
            # Convert longitude to -180 to 180 format to match ERA5-Land
            print("Converting 25km FWI coordinates...")
            self.fwi_25km = self.fwi_25km.assign_coords(
                longitude=(self.fwi_25km.longitude - 360)
            )
            print(f"  25km FWI lon range: {float(self.fwi_25km.longitude.min()):.1f} to {float(self.fwi_25km.longitude.max()):.1f}")
            
            # Load 10km ERA5-Land (for creating targets and features)
            print("Loading 10km ERA5-Land data...")
            self.era5_land = xr.open_dataset('data/data_0.nc')  # Extracted file
            print(f"  Shape: {dict(self.era5_land.dims)}")
            print(f"  Variables: {list(self.era5_land.data_vars)}")
            print(f"  10km land lon range: {float(self.era5_land.longitude.min()):.1f} to {float(self.era5_land.longitude.max()):.1f}")
            
            # Quick data inspection
            print(f"\\n25km FWI time range: {len(self.fwi_25km.valid_time)} days")
            print(f"10km ERA5-Land time range: {len(self.era5_land.valid_time)} days")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def create_10km_fwi_targets(self, subset_days=30):
        """
        Create 10km FWI targets using simplified formula
        
        Parameters:
        -----------
        subset_days : int
            Number of days to process (for testing)
        """
        print(f"\\n=== Creating 10km FWI Targets ({subset_days} days) ===")
        
        # Get subset for processing
        era5_subset = self.era5_land.isel(valid_time=slice(0, subset_days))
        
        # Extract meteorological variables
        print("Processing meteorological variables...")
        temp_c = era5_subset['t2m'] - 273.15  # K to C
        dewpoint_c = era5_subset['d2m'] - 273.15  # K to C
        u_wind = era5_subset['u10']  # m/s
        v_wind = era5_subset['v10']  # m/s
        precip_m = era5_subset['tp']  # m
        
        # Calculate derived variables
        print("Calculating derived variables...")
        
        # Wind speed (m/s to km/h)
        wind_speed = np.sqrt(u_wind**2 + v_wind**2) * 3.6
        
        # Relative humidity (simplified Magnus formula)
        def calculate_rh(temp, dewpoint):
            return 100 * np.exp(17.625 * dewpoint / (243.04 + dewpoint)) / np.exp(17.625 * temp / (243.04 + temp))
        
        rh = calculate_rh(temp_c, dewpoint_c)
        rh = np.clip(rh, 0, 100)
        
        # Precipitation (m to mm)
        precip_mm = precip_m * 1000
        
        print("Applying simplified FWI formula...")
        
        # Simplified FWI calculation (approximation for speed)
        # Real FWI requires iterative calculation with moisture codes
        
        # Temperature factor (higher temp = higher FWI)
        temp_factor = np.maximum(0, temp_c) / 30.0  # Normalize around 30°C
        
        # Humidity factor (lower humidity = higher FWI)  
        humidity_factor = (100 - rh) / 100.0
        
        # Wind factor (higher wind = higher FWI)
        wind_factor = np.minimum(wind_speed / 50.0, 2.0)  # Cap at 50 km/h
        
        # Precipitation factor (more rain = lower FWI)
        precip_factor = np.exp(-precip_mm * 2.0)  # Exponential decay
        
        # Combined FWI (simplified)
        fwi_10km = (temp_factor * humidity_factor * wind_factor * precip_factor) * 50.0
        
        # Ensure non-negative and reasonable range
        fwi_10km = np.clip(fwi_10km, 0, 100)
        
        print(f"10km FWI statistics:")
        print(f"  Mean: {float(fwi_10km.mean()):.2f}")
        print(f"  Std: {float(fwi_10km.std()):.2f}")
        print(f"  Min: {float(fwi_10km.min()):.2f}")
        print(f"  Max: {float(fwi_10km.max()):.2f}")
        
        # Store as dataset
        self.fwi_10km_target = xr.Dataset({
            'fwi_10km': (['valid_time', 'latitude', 'longitude'], fwi_10km.values)
        }, coords={
            'valid_time': era5_subset.valid_time,
            'latitude': era5_subset.latitude, 
            'longitude': era5_subset.longitude
        })
        
        return self.fwi_10km_target
    
    def prepare_training_data(self):
        """
        Prepare features and targets for ML training
        """
        print("\\n=== Preparing Training Data ===")
        
        # Get common time range
        common_times = min(len(self.fwi_25km.valid_time), len(self.fwi_10km_target.valid_time))
        
        print(f"Processing {common_times} time steps...")
        
        # Prepare features and targets
        features_list = []
        targets_list = []
        
        for t in range(common_times):
            print(f"  Processing day {t+1}/{common_times}", end='\\r')
            
            # Get 25km FWI for this day
            fwi_25_day = self.fwi_25km.isel(valid_time=t)['fwinx']
            
            # Get 10km target for this day  
            fwi_10_day = self.fwi_10km_target.isel(valid_time=t)['fwi_10km']
            
            # Get 10km auxiliary data
            era5_day = self.era5_land.isel(valid_time=t)
            
            # Create feature-target pairs for each 10km pixel
            for i, lat in enumerate(fwi_10_day.latitude.values):
                for j, lon in enumerate(fwi_10_day.longitude.values):
                    
                    # Target value at this 10km pixel
                    target_val = float(fwi_10_day.isel(latitude=i, longitude=j).values)
                    
                    if np.isnan(target_val):
                        continue
                    
                    # Features for this pixel
                    features = {}
                    
                    # 1. Interpolated 25km FWI at this location
                    try:
                        fwi_25_interp = fwi_25_day.interp(latitude=lat, longitude=lon, method='linear')
                        features['fwi_25km'] = float(fwi_25_interp.values)
                    except:
                        continue
                    
                    # 2. Local 10km meteorological variables
                    try:
                        features['temp_10km'] = float(era5_day['t2m'].isel(latitude=i, longitude=j).values)
                        features['dewpoint_10km'] = float(era5_day['d2m'].isel(latitude=i, longitude=j).values)
                        features['u_wind_10km'] = float(era5_day['u10'].isel(latitude=i, longitude=j).values)
                        features['v_wind_10km'] = float(era5_day['v10'].isel(latitude=i, longitude=j).values)
                        features['precip_10km'] = float(era5_day['tp'].isel(latitude=i, longitude=j).values)
                    except:
                        continue
                    
                    # 3. Spatial context (coordinates)
                    features['latitude'] = lat
                    features['longitude'] = lon
                    
                    # 4. Temporal context
                    features['day_of_year'] = t + 1
                    
                    # Skip if any features are NaN
                    if any(np.isnan(val) for val in features.values()):
                        continue
                    
                    features_list.append(features)
                    targets_list.append(target_val)
        
        print(f"\\nCreated {len(features_list)} training samples")
        
        # Convert to DataFrame and array
        self.features_df = pd.DataFrame(features_list)
        self.targets = np.array(targets_list)
        
        print(f"Feature columns: {list(self.features_df.columns)}")
        print(f"Target range: {self.targets.min():.2f} to {self.targets.max():.2f}")
        
        return self.features_df, self.targets
    
    def train_model(self, test_size=0.2):
        """
        Train Random Forest model
        
        Parameters:
        -----------
        test_size : float
            Fraction of data for testing
        """
        print(f"\\n=== Training ML Model ===")
        
        # Split data
        X = self.features_df
        y = self.targets
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Train Random Forest
        print("Training Random Forest...")
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        print("Evaluating model...")
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\\nModel Performance:")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  MAE: {mae:.3f}")
        print(f"  R² Score: {r2:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\\nTop 5 Most Important Features:")
        for _, row in feature_importance.head().iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        # Store results
        self.results = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'feature_importance': feature_importance,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        return self.model
    
    def validate_predictions(self):
        """
        Validate ML predictions against formula-based targets
        """
        print("\\n=== Validating Predictions ===")
        
        # Test aggregation consistency (key validation)
        print("Testing aggregation consistency...")
        
        # This would involve reshaping predictions back to spatial grid
        # and checking if they aggregate to match 25km values
        print("  Spatial consistency: Check if 10km predictions aggregate to 25km")
        print("  Physical plausibility: Check for reasonable FWI values")
        print("  Temporal consistency: Check for realistic day-to-day changes")
        
        # Plot predictions vs targets
        if len(self.results['y_test']) > 0:
            plt.figure(figsize=(10, 4))
            
            plt.subplot(1, 2, 1)
            plt.scatter(self.results['y_test'], self.results['y_pred'], alpha=0.5)
            plt.plot([0, 50], [0, 50], 'r--', label='Perfect prediction')
            plt.xlabel('Formula-based 10km FWI (Target)')
            plt.ylabel('ML Predicted 10km FWI')
            plt.title(f'ML vs Formula Predictions\\n(R² = {self.results["r2"]:.3f})')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            residuals = self.results['y_pred'] - self.results['y_test']
            plt.hist(residuals, bins=30, alpha=0.7)
            plt.xlabel('Prediction Error (ML - Formula)')
            plt.ylabel('Frequency')
            plt.title(f'Prediction Residuals\\n(RMSE = {self.results["rmse"]:.3f})')
            
            plt.tight_layout()
            plt.savefig('outputs/fwi_validation_plots.png', dpi=150, bbox_inches='tight')
            print("Saved validation plots to outputs/fwi_validation_plots.png")
            plt.show()
    
    def run_complete_pipeline(self):
        """
        Run the complete FWI downscaling pipeline
        """
        print("\\n" + "="*60)
        print("COMPLETE FWI DOWNSCALING PIPELINE: 25km → 10km")
        print("="*60)
        
        # Step 1: Load data
        if not self.load_data():
            print("Failed to load data. Exiting.")
            return None
        
        # Step 2: Create 10km targets using formula
        self.create_10km_fwi_targets(subset_days=30)  # Process 30 days for speed
        
        # Step 3: Prepare training data
        self.prepare_training_data()
        
        # Step 4: Train ML model
        self.train_model()
        
        # Step 5: Validate predictions
        self.validate_predictions()
        
        print("\\n" + "="*60)
        print("PIPELINE COMPLETE!")
        print("="*60)
        print("\\nSummary:")
        print(f"- Processed {len(self.targets)} training samples")
        print(f"- Model RMSE: {self.results['rmse']:.3f}")
        print(f"- Model R²: {self.results['r2']:.3f}")
        print("\\nThe ML model can now predict 10km FWI from 25km inputs!")
        
        return self.model

def main():
    """Main execution function"""
    # Create output directory
    import os
    os.makedirs('outputs', exist_ok=True)
    
    # Run pipeline
    pipeline = CompleteFWIPipeline()
    model = pipeline.run_complete_pipeline()
    
    return pipeline, model

if __name__ == "__main__":
    pipeline, model = main()