#!/usr/bin/env python3
"""
Quick FWI Downscaling Test
Test coordinate alignment and basic functionality
"""

import numpy as np
import xarray as xr
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def quick_test():
    """Quick test of FWI downscaling"""
    print("=== Quick FWI Downscaling Test ===")
    
    # Load data
    print("Loading data...")
    fwi_25km = xr.open_dataset('data/era5_fwi_2017.nc')
    era5_land = xr.open_dataset('data/data_0.nc')
    
    # Fix coordinates
    print("Fixing coordinates...")
    fwi_25km = fwi_25km.assign_coords(longitude=(fwi_25km.longitude - 360))
    
    print(f"25km FWI: lat {fwi_25km.latitude.min().values:.1f}-{fwi_25km.latitude.max().values:.1f}, lon {fwi_25km.longitude.min().values:.1f}-{fwi_25km.longitude.max().values:.1f}")
    print(f"10km Land: lat {era5_land.latitude.min().values:.1f}-{era5_land.latitude.max().values:.1f}, lon {era5_land.longitude.min().values:.1f}-{era5_land.longitude.max().values:.1f}")
    
    # Test coordinate overlap
    lat_overlap = (fwi_25km.latitude.min() <= era5_land.latitude.max()) and (fwi_25km.latitude.max() >= era5_land.latitude.min())
    lon_overlap = (fwi_25km.longitude.min() <= era5_land.longitude.max()) and (fwi_25km.longitude.max() >= era5_land.longitude.min())
    
    print(f"Coordinate overlap: lat={lat_overlap}, lon={lon_overlap}")
    
    if not (lat_overlap and lon_overlap):
        print("ERROR: No coordinate overlap!")
        return
    
    # Create simple features for first day
    print("Creating sample features...")
    day_0 = 0
    
    # Get 25km FWI for day 0
    fwi_25_day0 = fwi_25km.isel(valid_time=day_0)['fwinx']
    
    # Get 10km met data for day 0
    land_day0 = era5_land.isel(valid_time=day_0)
    
    # Create targets: simplified FWI from met data
    temp_c = land_day0['t2m'] - 273.15
    humidity_approx = 50.0  # Simplified
    wind_speed = np.sqrt(land_day0['u10']**2 + land_day0['v10']**2)
    
    # Simplified FWI target
    fwi_target = np.clip((temp_c / 20.0) * (wind_speed / 10.0) * 10.0, 0, 50)
    
    # Sample features and targets
    features = []
    targets = []
    
    # Sample every 5th pixel to reduce computation
    for i in range(0, len(land_day0.latitude), 5):
        for j in range(0, len(land_day0.longitude), 5):
            lat = float(land_day0.latitude[i])
            lon = float(land_day0.longitude[j])
            
            # Get target
            target = float(fwi_target.isel(latitude=i, longitude=j).values)
            
            if np.isnan(target):
                continue
            
            # Get 25km FWI interpolated to this location
            try:
                fwi_25_interp = fwi_25_day0.interp(latitude=lat, longitude=lon, method='linear')
                fwi_25_val = float(fwi_25_interp.values)
                
                if np.isnan(fwi_25_val):
                    continue
                
                # Features: 25km FWI + local met data
                features.append([
                    fwi_25_val,  # 25km FWI
                    float(land_day0['t2m'].isel(latitude=i, longitude=j).values),  # temp
                    float(land_day0['u10'].isel(latitude=i, longitude=j).values),  # u wind
                    float(land_day0['v10'].isel(latitude=i, longitude=j).values),  # v wind
                    lat,  # latitude
                    lon   # longitude
                ])
                targets.append(target)
                
            except Exception as e:
                continue
    
    print(f"Created {len(features)} samples")
    
    if len(features) < 10:
        print("ERROR: Too few samples created!")
        return
    
    # Train simple model
    print("Training model...")
    X = np.array(features)
    y = np.array(targets)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model performance:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  R²: {r2:.3f}")
    print(f"  Target range: {y.min():.2f} to {y.max():.2f}")
    
    # Feature importance
    feature_names = ['fwi_25km', 'temp', 'u_wind', 'v_wind', 'lat', 'lon']
    importance = model.feature_importances_
    
    print("Feature importance:")
    for name, imp in zip(feature_names, importance):
        print(f"  {name}: {imp:.3f}")
    
    print("\n✅ Quick test completed successfully!")
    print("The coordinate systems are now aligned and ML training works.")
    
    return model

if __name__ == "__main__":
    model = quick_test()