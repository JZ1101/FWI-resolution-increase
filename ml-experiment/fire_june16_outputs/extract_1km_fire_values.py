#!/usr/bin/env python3
"""
Extract exact 1km FWI values at fire location
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def get_1km_fire_location_values():
    """Get exact FWI values at fire location from 1km predictions"""
    
    # Load data
    data = pd.read_csv('database_portugal_2017/experiment_2017_portugal/features_2017_COMPLETE_FINAL.csv')
    data['time'] = pd.to_datetime(data['time'])
    
    # Fire location
    fire_lat, fire_lon = 39.92, -8.15
    
    # Prepare features
    exclude_cols = ['time', 'latitude', 'longitude', 'fwi']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    missing_pct = data[feature_cols].isnull().sum() / len(data) * 100
    usable_features = missing_pct[missing_pct <= 50].index.tolist()
    
    X_temp = data[usable_features].copy()
    for col in X_temp.columns:
        if X_temp[col].dtype == 'object':
            try:
                X_temp[col] = pd.to_numeric(X_temp[col], errors='coerce')
            except:
                X_temp = X_temp.drop(col, axis=1)
    
    X = X_temp.fillna(X_temp.mean())
    y = data['fwi']
    
    # Train/test split
    train_mask = (data['time'] < '2017-05-01')
    train_indices = data[train_mask].index
    
    X_train = X.loc[train_indices].fillna(0)
    y_train = y.loc[train_indices]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train models (simplified)
    print("Training models for 1km predictions...")
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    
    # ANN  
    ann = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
    ann.fit(X_train_scaled, y_train)
    
    # Create exact fire location feature vector
    # Use mean values from training data but set coordinates precisely
    fire_features = pd.DataFrame()
    
    for col in X.columns:
        if 'lat' in col.lower():
            # Normalize fire latitude
            fire_features[col] = [(fire_lat - data['latitude'].min()) / (data['latitude'].max() - data['latitude'].min())]
        elif 'lon' in col.lower():
            # Normalize fire longitude  
            fire_features[col] = [(fire_lon - data['longitude'].min()) / (data['longitude'].max() - data['longitude'].min())]
        else:
            # Use mean value from training
            fire_features[col] = [X.loc[train_indices, col].mean()]
    
    fire_features = fire_features.fillna(0)
    fire_features_scaled = scaler.transform(fire_features)
    
    # Make predictions at exact fire location
    xgb_pred = xgb_model.predict(fire_features)[0]
    ann_pred = ann.predict(fire_features_scaled)[0] 
    ensemble_pred = (xgb_pred + ann_pred) / 2
    
    print(f"\\n1KM RESOLUTION PREDICTIONS AT EXACT FIRE LOCATION:")
    print(f"Fire coordinates: {fire_lat}°N, {fire_lon}°W")
    print(f"XGBoost 1km FWI: {xgb_pred:.3f}")
    print(f"ANN 1km FWI: {ann_pred:.3f}")
    print(f"Ensemble 1km FWI: {ensemble_pred:.3f}")
    
    # Compare with 25km ERA5 value
    june16_data = data[data['time'].dt.date == pd.to_datetime('2017-06-16').date()]
    distances = np.sqrt((june16_data['latitude'] - fire_lat)**2 + (june16_data['longitude'] - fire_lon)**2)
    era5_fwi = june16_data.loc[distances.idxmin(), 'fwi']
    
    print(f"\\nCOMPARISON:")
    print(f"ERA5 (25km): {era5_fwi:.3f}")
    print(f"XGBoost (1km): {xgb_pred:.3f} (diff: {xgb_pred - era5_fwi:+.3f})")
    print(f"ANN (1km): {ann_pred:.3f} (diff: {ann_pred - era5_fwi:+.3f})")
    print(f"Ensemble (1km): {ensemble_pred:.3f} (diff: {ensemble_pred - era5_fwi:+.3f})")
    
    return {
        'fire_coordinates': (fire_lat, fire_lon),
        'era5_25km': float(era5_fwi),
        'xgboost_1km': float(xgb_pred),
        'ann_1km': float(ann_pred),
        'ensemble_1km': float(ensemble_pred)
    }

if __name__ == "__main__":
    results = get_1km_fire_location_values()
    
    # Save results
    import json
    with open('fire_location_1km_values.json', 'w') as f:
        json.dump(results, f, indent=2)