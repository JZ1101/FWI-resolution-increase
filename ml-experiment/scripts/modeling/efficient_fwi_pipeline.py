#!/usr/bin/env python3
"""
Efficient FWI Downscaling Pipeline
Optimized for speed while maintaining quality
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

def run_efficient_pipeline():
    """Run optimized FWI downscaling pipeline"""
    print("=== Efficient FWI Downscaling Pipeline ===")
    
    # Load data
    print("Loading data...")
    fwi_25km = xr.open_dataset('data/era5_fwi_2017.nc')
    fwi_10km_target = xr.open_dataset('data/fwi_10km_full_year.nc')
    era5_land = xr.open_dataset('data/data_0.nc')
    
    # Fix coordinates
    fwi_25km = fwi_25km.assign_coords(longitude=(fwi_25km.longitude - 360))
    
    print(f"Loaded data: 25km FWI {fwi_25km.fwinx.shape}, 10km targets {fwi_10km_target.fwi_10km.shape}")
    
    # Create training samples (efficient sampling)
    print("Creating training samples...")
    features_list = []
    targets_list = []
    
    # Process 20 days with larger spatial stride for efficiency
    max_days = 20
    spatial_stride = 5
    
    for day in range(max_days):
        if day % 5 == 0:
            print(f"  Day {day+1}/{max_days}")
        
        # Get daily data
        fwi_25_day = fwi_25km.isel(valid_time=day)['fwinx']
        fwi_10_day = fwi_10km_target.isel(time=day)['fwi_10km']
        land_day = era5_land.isel(valid_time=day)
        
        # Sample every 5th pixel
        for i in range(0, len(era5_land.latitude), spatial_stride):
            for j in range(0, len(era5_land.longitude), spatial_stride):
                
                lat = float(era5_land.latitude[i])
                lon = float(era5_land.longitude[j])
                
                try:
                    # Get 10km target
                    target_val = float(fwi_10_day.isel(latitude=i, longitude=j).values)
                    if np.isnan(target_val) or target_val < 0:
                        continue
                    
                    # Get 25km FWI interpolated
                    fwi_25_interp = fwi_25_day.interp(latitude=lat, longitude=lon, method='linear')
                    fwi_25_val = float(fwi_25_interp.values)
                    if np.isnan(fwi_25_val):
                        continue
                    
                    # Get meteorological features
                    temp = float(land_day['t2m'].isel(latitude=i, longitude=j).values) - 273.15
                    u_wind = float(land_day['u10'].isel(latitude=i, longitude=j).values)
                    v_wind = float(land_day['v10'].isel(latitude=i, longitude=j).values)
                    precip = float(land_day['tp'].isel(latitude=i, longitude=j).values) * 1000
                    
                    wind_speed = np.sqrt(u_wind**2 + v_wind**2)
                    
                    # Create feature vector
                    features = [fwi_25_val, temp, wind_speed, precip, lat, lon, day + 1]
                    
                    if not any(np.isnan(val) or np.isinf(val) for val in features):
                        features_list.append(features)
                        targets_list.append(target_val)
                
                except:
                    continue
    
    print(f"Created {len(features_list)} samples")
    
    # Prepare data
    feature_names = ['fwi_25km', 'temp_10km', 'wind_speed_10km', 'precip_10km', 'lat', 'lon', 'day']
    X = pd.DataFrame(features_list, columns=feature_names)
    y = np.array(targets_list)
    
    print(f"Target range: {y.min():.2f} to {y.max():.2f} (mean: {y.mean():.2f})")
    
    # Train model
    print("Training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\\nResults:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE: {mae:.3f}")
    print(f"  R²: {r2:.3f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\\nFeature Importance:")
    for _, row in importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    # Create validation plot
    print("\\nCreating validation plot...")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_pred, alpha=0.6)
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
    plt.xlabel('Actual 10km FWI (Formula)')
    plt.ylabel('Predicted 10km FWI (ML)')
    plt.title(f'Predictions (R² = {r2:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    residuals = y_pred - y_test
    plt.hist(residuals, bins=20, alpha=0.7)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title(f'Residuals (RMSE = {rmse:.3f})')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.barh(range(len(importance)), importance['importance'])
    plt.yticks(range(len(importance)), importance['feature'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/fwi_results_efficient.png', dpi=150, bbox_inches='tight')
    print("Saved results to outputs/fwi_results_efficient.png")
    plt.show()
    
    print("\\n" + "="*60)
    print("✅ EFFICIENT FWI DOWNSCALING COMPLETE!")
    print("="*60)
    print(f"Successfully trained ML model to enhance 25km FWI to 10km resolution")
    print(f"Model accuracy: R² = {r2:.3f}, RMSE = {rmse:.3f}")
    print(f"Model can now predict high-resolution FWI from coarse inputs!")
    
    return model, X_test, y_test, y_pred

if __name__ == "__main__":
    import os
    os.makedirs('outputs', exist_ok=True)
    model, X_test, y_test, y_pred = run_efficient_pipeline()