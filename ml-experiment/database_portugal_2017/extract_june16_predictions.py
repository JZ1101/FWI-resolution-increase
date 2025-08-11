#!/usr/bin/env python3
"""
Extract June 16 predictions from our trained models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

def extract_june16_predictions():
    print("EXTRACTING JUNE 16 PREDICTIONS FROM TRAINED MODELS")
    print("="*60)
    
    # Load data
    data = pd.read_csv('experiment_2017_portugal/features_2017_COMPLETE_FINAL.csv')
    data['time'] = pd.to_datetime(data['time'])
    
    # Prepare features exactly like main experiment
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
    coords = data[['latitude', 'longitude']].copy()
    
    # Same temporal split as main experiment
    train_mask = (data['time'] < '2017-05-01')
    val_mask = (data['time'] >= '2017-05-01') & (data['time'] < '2017-06-01')
    test_mask = (data['time'] >= '2017-06-01') & (data['time'] < '2017-07-01')
    
    train_indices = data[train_mask].index
    test_indices = data[test_mask].index
    
    X_train = X.loc[train_indices].fillna(0)
    X_test = X.loc[test_indices].fillna(0)
    y_train = y.loc[train_indices]
    y_test = y.loc[test_indices]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training: {X_train.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples (includes June 16)")
    
    # Find June 16 data in test set
    june16_mask = (data['time'].dt.date == pd.to_datetime('2017-06-16').date())
    june16_test_indices = [idx for idx in test_indices if idx in data[june16_mask].index]
    
    print(f"June 16 samples in test set: {len(june16_test_indices)}")
    
    # Fire location
    fire_lat, fire_lon = 39.92, -8.15
    june16_data = data.loc[june16_test_indices]
    distances = np.sqrt((june16_data['latitude'] - fire_lat)**2 + (june16_data['longitude'] - fire_lon)**2)
    closest_june16_idx = distances.idxmin()
    
    fire_location_data = data.loc[closest_june16_idx]
    era5_fwi = fire_location_data['fwi']
    
    print(f"\nFIRE LOCATION ON JUNE 16:")
    print(f"Coordinates: {fire_location_data['latitude']:.3f}°N, {fire_location_data['longitude']:.3f}°W")
    print(f"ERA5 FWI: {era5_fwi:.3f}")
    
    # Train models exactly like main experiment
    print(f"\nTRAINING MODELS (same as main experiment):")
    
    # ANN
    ann = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu', solver='adam', alpha=0.001,
        batch_size=256, learning_rate='adaptive',
        max_iter=500, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=20,
        random_state=42
    )
    ann.fit(X_train_scaled, y_train)
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    
    # CNN
    cnn_model = keras.Sequential([
        keras.layers.Input(shape=(X_train_scaled.shape[1], 1)),
        keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    cnn_model.compile(optimizer='adam', loss='mse')
    
    X_train_cnn = X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1)
    X_test_cnn = X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1)
    
    cnn_model.fit(X_train_cnn, y_train, epochs=50, batch_size=256,
                  callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
                  verbose=0)
    
    print("Models trained successfully")
    
    # Make predictions on entire test set
    ann_test_preds = ann.predict(X_test_scaled)
    xgb_test_preds = xgb_model.predict(X_test)
    cnn_test_preds = cnn_model.predict(X_test_cnn, verbose=0).flatten()
    
    # Find predictions for fire location
    fire_test_position = list(test_indices).index(closest_june16_idx)
    
    ann_fire_pred = ann_test_preds[fire_test_position]
    xgb_fire_pred = xgb_test_preds[fire_test_position]
    cnn_fire_pred = cnn_test_preds[fire_test_position]
    ensemble_fire_pred = (xgb_fire_pred + cnn_fire_pred) / 2  # Based on our best 2 models
    
    print(f"\nJUNE 16 FIRE LOCATION PREDICTIONS:")
    print(f"ERA5 FWI: {era5_fwi:.3f}")
    print(f"ANN Prediction: {ann_fire_pred:.3f} (diff: {ann_fire_pred - era5_fwi:+.3f})")
    print(f"XGBoost Prediction: {xgb_fire_pred:.3f} (diff: {xgb_fire_pred - era5_fwi:+.3f})")
    print(f"CNN Prediction: {cnn_fire_pred:.3f} (diff: {cnn_fire_pred - era5_fwi:+.3f})")
    print(f"Ensemble Prediction: {ensemble_fire_pred:.3f} (diff: {ensemble_fire_pred - era5_fwi:+.3f})")
    
    # Risk assessment
    def assess_risk(fwi):
        if fwi >= 38: return "EXTREME"
        elif fwi >= 21.3: return "HIGH" 
        elif fwi >= 11.2: return "MODERATE"
        elif fwi >= 5.2: return "LOW"
        else: return "VERY LOW"
    
    print(f"\nRISK ASSESSMENT:")
    print(f"ERA5 Risk: {assess_risk(era5_fwi)}")
    print(f"ANN Risk: {assess_risk(ann_fire_pred)}")
    print(f"XGBoost Risk: {assess_risk(xgb_fire_pred)}")
    print(f"CNN Risk: {assess_risk(cnn_fire_pred)}")
    print(f"Ensemble Risk: {assess_risk(ensemble_fire_pred)}")
    
    # Create results table
    results = [
        {'Model': 'ANN', 'ERA5_FWI': era5_fwi, 'Predicted_FWI': ann_fire_pred, 'Difference': ann_fire_pred - era5_fwi, 'ERA5_Risk': assess_risk(era5_fwi), 'Pred_Risk': assess_risk(ann_fire_pred)},
        {'Model': 'XGBoost', 'ERA5_FWI': era5_fwi, 'Predicted_FWI': xgb_fire_pred, 'Difference': xgb_fire_pred - era5_fwi, 'ERA5_Risk': assess_risk(era5_fwi), 'Pred_Risk': assess_risk(xgb_fire_pred)},
        {'Model': 'CNN', 'ERA5_FWI': era5_fwi, 'Predicted_FWI': cnn_fire_pred, 'Difference': cnn_fire_pred - era5_fwi, 'ERA5_Risk': assess_risk(era5_fwi), 'Pred_Risk': assess_risk(cnn_fire_pred)},
        {'Model': 'Ensemble', 'ERA5_FWI': era5_fwi, 'Predicted_FWI': ensemble_fire_pred, 'Difference': ensemble_fire_pred - era5_fwi, 'ERA5_Risk': assess_risk(era5_fwi), 'Pred_Risk': assess_risk(ensemble_fire_pred)}
    ]
    
    results_df = pd.DataFrame(results)
    print(f"\nFINAL JUNE 16 RESULTS TABLE:")
    print("="*80)
    print(results_df.round(3).to_string(index=False))
    
    return results_df

if __name__ == "__main__":
    results = extract_june16_predictions()