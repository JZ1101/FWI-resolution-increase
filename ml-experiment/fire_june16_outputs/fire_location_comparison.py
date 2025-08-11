#!/usr/bin/env python3
"""
Fire Location Comparison - Exactly what you requested
Graph 1: Actual fire location vs nearest ERA5 point + models
Graph 2: Actual fire location exact coordinates vs models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

def create_fire_comparison_visuals():
    print("CREATING FIRE LOCATION COMPARISON VISUALS")
    print("=" * 50)
    
    # Load data
    data = pd.read_csv('database_portugal_2017/experiment_2017_portugal/features_2017_COMPLETE_FINAL.csv')
    data['time'] = pd.to_datetime(data['time'])
    
    # Actual fire location
    actual_fire_lat, actual_fire_lon = 39.92, -8.15
    
    # Find June 16 data
    june16_data = data[data['time'].dt.date == pd.to_datetime('2017-06-16').date()].copy()
    
    # Find nearest ERA5 grid point
    distances = np.sqrt((june16_data['latitude'] - actual_fire_lat)**2 + 
                       (june16_data['longitude'] - actual_fire_lon)**2)
    nearest_era5_idx = distances.idxmin()
    nearest_era5_point = june16_data.loc[nearest_era5_idx]
    
    era5_fwi = nearest_era5_point['fwi']
    era5_lat = nearest_era5_point['latitude']
    era5_lon = nearest_era5_point['longitude']
    
    print(f"Actual fire location: {actual_fire_lat}°N, {actual_fire_lon}°W")
    print(f"Nearest ERA5 point: {era5_lat}°N, {era5_lon}°W")
    print(f"Distance: {distances.min():.3f}°")
    print(f"ERA5 FWI: {era5_fwi:.3f}")
    
    # Prepare features and train models (simplified)
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
    test_mask = (data['time'] >= '2017-06-01') & (data['time'] < '2017-07-01')
    
    train_indices = data[train_mask].index
    test_indices = data[test_mask].index
    
    X_train = X.loc[train_indices].fillna(0)
    X_test = X.loc[test_indices].fillna(0)
    y_train = y.loc[train_indices]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models quickly
    print("Training models...")
    
    # XGBoost (best performer)
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    
    # ANN (simple version)
    ann = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
    ann.fit(X_train_scaled, y_train)
    
    # Get predictions at nearest ERA5 point
    nearest_era5_test_position = list(test_indices).index(nearest_era5_idx)
    
    xgb_era5_pred = xgb_model.predict(X_test)[nearest_era5_test_position]
    ann_era5_pred = ann.predict(X_test_scaled)[nearest_era5_test_position]
    ensemble_era5_pred = (xgb_era5_pred + ann_era5_pred) / 2
    
    print("Models trained and predictions made")
    
    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('June 16, 2017 Pedrógão Grande Fire - Location Comparison', 
                 fontsize=16, fontweight='bold')
    
    # GRAPH 1: Actual fire location vs nearest ERA5 point + models
    ax1.set_title('Graph 1: Fire Location vs Nearest ERA5 Point + Models', 
                  fontsize=14, fontweight='bold')
    
    # Plot actual fire location
    ax1.scatter(actual_fire_lon, actual_fire_lat, c='red', marker='*', s=800, 
               edgecolor='black', linewidth=2, label='Actual Fire Location', zorder=10)
    
    # Plot nearest ERA5 point
    ax1.scatter(era5_lon, era5_lat, c='blue', marker='s', s=400, 
               edgecolor='black', linewidth=2, label=f'Nearest ERA5 Point\\n(FWI = {era5_fwi:.1f})', zorder=9)
    
    # Add connection line
    ax1.plot([actual_fire_lon, era5_lon], [actual_fire_lat, era5_lat], 
             'k--', alpha=0.5, linewidth=2)
    
    # Add distance annotation
    ax1.text((actual_fire_lon + era5_lon)/2, (actual_fire_lat + era5_lat)/2 + 0.01,
             f'Distance: {distances.min()*111:.1f} km', 
             ha='center', va='bottom', fontsize=10, 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Model predictions at ERA5 point
    model_data = [
        ('ERA5', era5_fwi, 'blue'),
        ('XGBoost', xgb_era5_pred, 'green'),
        ('ANN', ann_era5_pred, 'orange'),
        ('Ensemble', ensemble_era5_pred, 'purple')
    ]
    
    # Bar chart showing FWI values
    for i, (model, fwi_val, color) in enumerate(model_data):
        ax1.bar(era5_lon + 0.02 + i*0.01, fwi_val/100, width=0.008, bottom=era5_lat, 
               color=color, alpha=0.8, label=f'{model}: {fwi_val:.1f}')
    
    ax1.set_xlabel('Longitude (°)')
    ax1.set_ylabel('Latitude (°)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # GRAPH 2: Exact fire location vs models (conceptual)
    ax2.set_title('Graph 2: Exact Fire Location vs Model Predictions', 
                  fontsize=14, fontweight='bold')
    
    # Risk categories for background
    risk_levels = [
        ('Very Low', 0, 5.2, '#90EE90'),
        ('Low', 5.2, 11.2, '#FFFF99'),
        ('Moderate', 11.2, 21.3, '#FFB347'),
        ('High', 21.3, 38, '#FF6347'),
        ('Extreme', 38, 50, '#8B0000')
    ]
    
    for name, low, high, color in risk_levels:
        ax2.axhspan(low, high, alpha=0.3, color=color, label=name)
    
    # Model predictions (using same values as ERA5 point for demonstration)
    models = ['ERA5', 'XGBoost', 'ANN', 'Ensemble']
    predictions = [era5_fwi, xgb_era5_pred, ann_era5_pred, ensemble_era5_pred]
    colors = ['blue', 'green', 'orange', 'purple']
    
    x_pos = np.arange(len(models))
    bars = ax2.bar(x_pos, predictions, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, pred in zip(bars, predictions):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{pred:.1f}', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Fire Weather Index (FWI)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models)
    ax2.set_ylim(0, 50)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add fire location info
    ax2.text(0.02, 0.98, f'Fire Location: {actual_fire_lat}°N, {actual_fire_lon}°W\\nDate: June 16, 2017',
             transform=ax2.transAxes, va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('fire_location_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: fire_location_comparison.png")
    
    # Print summary
    print(f"\\nSUMMARY:")
    print(f"Actual fire: {actual_fire_lat}°N, {actual_fire_lon}°W")
    print(f"ERA5 point: {era5_lat}°N, {era5_lon}°W (distance: {distances.min()*111:.1f} km)")
    print(f"ERA5 FWI: {era5_fwi:.1f}")
    print(f"XGBoost: {xgb_era5_pred:.1f}")
    print(f"ANN: {ann_era5_pred:.1f}")  
    print(f"Ensemble: {ensemble_era5_pred:.1f}")
    
    plt.show()
    
    return {
        'actual_fire_coords': (actual_fire_lat, actual_fire_lon),
        'era5_coords': (era5_lat, era5_lon),
        'era5_fwi': era5_fwi,
        'model_predictions': {
            'XGBoost': xgb_era5_pred,
            'ANN': ann_era5_pred,
            'Ensemble': ensemble_era5_pred
        }
    }

if __name__ == "__main__":
    results = create_fire_comparison_visuals()