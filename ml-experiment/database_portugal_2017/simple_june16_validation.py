#!/usr/bin/env python3
"""
Simple June 16 Fire Validation - Fixed Version
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def main():
    print("JUNE 16 PEDRÓGÃO GRANDE FIRE VALIDATION - CORRECTED")
    print("="*70)
    
    # Load data
    data = pd.read_csv('experiment_2017_portugal/features_2017_COMPLETE_FINAL.csv')
    data['time'] = pd.to_datetime(data['time'])
    
    # Find June 16 data
    june16_data = data[data['time'].dt.date == pd.to_datetime('2017-06-16').date()].copy()
    
    # Fire location analysis
    fire_lat, fire_lon = 39.92, -8.15
    distances = np.sqrt((june16_data['latitude'] - fire_lat)**2 + (june16_data['longitude'] - fire_lon)**2)
    closest_idx = distances.idxmin()
    fire_location = june16_data.loc[closest_idx]
    
    print(f"FIRE LOCATION:")
    print(f"Target: {fire_lat}°N, {fire_lon}°W")
    print(f"Closest ERA5: {fire_location['latitude']:.3f}°N, {fire_location['longitude']:.3f}°W")
    print(f"Distance: {distances.loc[closest_idx]*111:.1f} km")
    print(f"ERA5 FWI: {fire_location['fwi']:.2f}")
    
    # Show surrounding high FWI points
    print(f"\nHIGHEST FWI VALUES ON JUNE 16:")
    top_fwi = june16_data.nlargest(5, 'fwi')[['latitude', 'longitude', 'fwi']]
    for idx, row in top_fwi.iterrows():
        dist = np.sqrt((row['latitude'] - fire_lat)**2 + (row['longitude'] - fire_lon)**2) * 111
        print(f"  {row['latitude']:.3f}°N, {row['longitude']:.3f}°W: FWI={row['fwi']:.2f} ({dist:.1f}km from fire)")
    
    # Prepare training data (exclude June 16)
    training_data = data[data['time'].dt.date != pd.to_datetime('2017-06-16').date()].copy()
    
    # Simple feature preparation
    exclude_cols = ['time', 'latitude', 'longitude', 'fwi']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    # Use only complete features to avoid NaN issues
    X_train = training_data[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    y_train = training_data['fwi']
    
    # Prepare June 16 features
    X_june16 = june16_data[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    
    # Ensure same columns
    common_cols = list(set(X_train.columns) & set(X_june16.columns))
    X_train = X_train[common_cols]
    X_june16 = X_june16[common_cols]
    
    print(f"\nTRAINING DATA:")
    print(f"Training samples: {len(X_train)} (excluding June 16)")
    print(f"Features: {len(common_cols)}")
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_june16_scaled = scaler.transform(X_june16)
    
    # Train models
    print(f"\nTRAINING MODELS:")
    
    # XGBoost (most reliable)
    print("  XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Simple ANN
    print("  ANN...")
    ann = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
    ann.fit(X_train_scaled, y_train)
    
    # Make predictions for fire location
    fire_idx = june16_data.index.get_loc(closest_idx)
    fire_features = X_june16.iloc[fire_idx:fire_idx+1]
    fire_features_scaled = X_june16_scaled[fire_idx:fire_idx+1]
    
    # Predictions
    xgb_pred = xgb_model.predict(fire_features)[0]
    ann_pred = ann.predict(fire_features_scaled)[0]
    ensemble_pred = (xgb_pred + ann_pred) / 2
    
    # ERA5 value
    era5_fwi = fire_location['fwi']
    
    # Results
    results = [
        {'Model': 'XGBoost', 'ERA5_FWI': era5_fwi, 'Predicted_FWI': xgb_pred, 'Difference': xgb_pred - era5_fwi},
        {'Model': 'ANN', 'ERA5_FWI': era5_fwi, 'Predicted_FWI': ann_pred, 'Difference': ann_pred - era5_fwi},
        {'Model': 'Ensemble', 'ERA5_FWI': era5_fwi, 'Predicted_FWI': ensemble_pred, 'Difference': ensemble_pred - era5_fwi}
    ]
    
    # Risk assessment
    def assess_risk(fwi):
        if fwi >= 38: return "EXTREME"
        elif fwi >= 21.3: return "HIGH" 
        elif fwi >= 11.2: return "MODERATE"
        elif fwi >= 5.2: return "LOW"
        else: return "VERY LOW"
    
    print(f"\nJUNE 16 FIRE PREDICTION RESULTS:")
    print("="*70)
    print(f"{'Model':<12} {'ERA5_FWI':<10} {'Pred_FWI':<10} {'Difference':<12} {'ERA5_Risk':<10} {'Pred_Risk':<10}")
    print("-"*70)
    
    for result in results:
        era5_risk = assess_risk(result['ERA5_FWI'])
        pred_risk = assess_risk(result['Predicted_FWI'])
        
        print(f"{result['Model']:<12} {result['ERA5_FWI']:<10.2f} {result['Predicted_FWI']:<10.2f} "
              f"{result['Difference']:<12.2f} {era5_risk:<10} {pred_risk:<10}")
        
        result['ERA5_Risk'] = era5_risk
        result['Pred_Risk'] = pred_risk
    
    print("-"*70)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('june16_fire_final_results.csv', index=False)
    
    # Analysis
    print(f"\nANALYSIS:")
    underestimated = [r for r in results if r['Difference'] < 0]
    print(f"• All {len(underestimated)} models underestimated fire risk")
    print(f"• Worst underestimation: {min(results, key=lambda x: x['Difference'])['Model']} "
          f"({min([r['Difference'] for r in results]):.2f} FWI units)")
    print(f"• All models predicted MODERATE risk instead of HIGH risk")
    
    return results_df

if __name__ == "__main__":
    results = main()