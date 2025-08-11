#!/usr/bin/env python3
"""
June 16 Fire Validation for ALL models
Pedrógão Grande fire event analysis
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

class June16FireValidator:
    """Validate all models against June 16 Pedrógão Grande fire"""
    
    def __init__(self):
        self.data_path = 'experiment_2017_portugal/features_2017_COMPLETE_FINAL.csv'
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def load_and_train_models(self):
        """Load data and train all models"""
        print("LOADING DATA AND TRAINING MODELS FOR JUNE 16 VALIDATION")
        print("="*70)
        
        self.data = pd.read_csv(self.data_path)
        self.data['time'] = pd.to_datetime(self.data['time'])
        
        # Prepare features
        exclude_cols = ['time', 'latitude', 'longitude', 'fwi']
        feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        missing_pct = self.data[feature_cols].isnull().sum() / len(self.data) * 100
        usable_features = missing_pct[missing_pct <= 50].index.tolist()
        
        X_temp = self.data[usable_features].copy()
        for col in X_temp.columns:
            if X_temp[col].dtype == 'object':
                try:
                    X_temp[col] = pd.to_numeric(X_temp[col], errors='coerce')
                except:
                    X_temp = X_temp.drop(col, axis=1)
        
        self.X = X_temp.fillna(X_temp.mean())
        self.y = self.data['fwi']
        self.coords = self.data[['latitude', 'longitude']].copy()
        
        # Find June 16 data
        june16_mask = (self.data['time'].dt.date == pd.to_datetime('2017-06-16').date())
        if june16_mask.sum() > 0:
            self.june16_data = self.data[june16_mask].copy()
            self.june16_X = self.X.loc[self.june16_data.index].fillna(0)
            self.june16_y = self.y.loc[self.june16_data.index]
            print(f"June 16 data found: {self.june16_X.shape[0]} coordinates")
        else:
            print("ERROR: June 16 data not found!")
            return
            
        # Temporal split for training
        train_mask = (self.data['time'] < '2017-05-01')
        val_mask = (self.data['time'] >= '2017-05-01') & (self.data['time'] < '2017-06-01')
        
        train_indices = self.data[train_mask].index
        val_indices = self.data[val_mask].index
        
        self.X_train = self.X.loc[train_indices].fillna(0)
        self.X_val = self.X.loc[val_indices].fillna(0)
        self.y_train = self.y.loc[train_indices]
        self.y_val = self.y.loc[val_indices]
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        
        print("Training models...")
        self.train_all_models()
        
    def train_all_models(self):
        """Train all models quickly"""
        
        # ANN
        ann = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu', solver='adam', alpha=0.001,
            batch_size=256, max_iter=200, random_state=42
        )
        ann.fit(self.X_train_scaled, self.y_train)
        self.models['ANN'] = ann
        
        # XGBoost  
        xgb_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1
        )
        xgb_model.fit(self.X_train, self.y_train, 
                     eval_set=[(self.X_val, self.y_val)], verbose=False)
        self.models['XGBoost'] = xgb_model
        
        # CNN
        model = keras.Sequential([
            keras.layers.Input(shape=(self.X_train_scaled.shape[1], 1)),
            keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        X_train_cnn = self.X_train_scaled.reshape(-1, self.X_train_scaled.shape[1], 1)
        X_val_cnn = self.X_val_scaled.reshape(-1, self.X_val_scaled.shape[1], 1)
        
        model.fit(X_train_cnn, self.y_train,
                  validation_data=(X_val_cnn, self.y_val),
                  epochs=20, batch_size=256, verbose=0)
        self.models['CNN'] = model
        
        print("All models trained successfully")
        
    def validate_june16_fire(self):
        """Validate all models against June 16 fire event"""
        print("\nJUNE 16 PEDRÓGÃO GRANDE FIRE VALIDATION")
        print("="*60)
        
        # Pedrógão Grande fire location: 39.92°N, 8.15°W  
        fire_lat, fire_lon = 39.92, -8.15
        
        # Find closest coordinate to fire location
        distances = np.sqrt(
            (self.june16_data['latitude'] - fire_lat)**2 + 
            (self.june16_data['longitude'] - fire_lon)**2
        )
        closest_idx = distances.idxmin()
        closest_coord = self.june16_data.loc[closest_idx]
        
        # Original ERA5 FWI at fire location
        era5_fwi = closest_coord['fwi']
        
        print(f"Fire Location: {fire_lat}°N, {fire_lon}°W")
        print(f"Closest Coordinate: {closest_coord['latitude']:.3f}°N, {closest_coord['longitude']:.3f}°W")
        print(f"Distance to Fire: {distances.loc[closest_idx]*111:.1f} km")
        print(f"ERA5 FWI (25km): {era5_fwi:.2f}")
        print()
        
        # Get features for June 16 at fire location
        june16_features = self.june16_X.loc[closest_idx:closest_idx]
        
        fire_results = []
        
        # Test each model
        for model_name in ['ANN', 'XGBoost', 'CNN']:
            print(f"Testing {model_name}...")
            
            if model_name == 'ANN':
                june16_scaled = self.scaler.transform(june16_features)
                enhanced_fwi = self.models['ANN'].predict(june16_scaled)[0]
                
            elif model_name == 'XGBoost':
                enhanced_fwi = self.models['XGBoost'].predict(june16_features)[0]
                
            elif model_name == 'CNN':
                june16_scaled = self.scaler.transform(june16_features)
                june16_cnn = june16_scaled.reshape(-1, june16_scaled.shape[1], 1)
                enhanced_fwi = self.models['CNN'].predict(june16_cnn, verbose=0).flatten()[0]
            
            difference = enhanced_fwi - era5_fwi
            
            # Risk assessment
            def assess_fire_risk(fwi):
                if fwi >= 38: return "EXTREME"
                elif fwi >= 21.3: return "HIGH" 
                elif fwi >= 11.2: return "MODERATE"
                elif fwi >= 5.2: return "LOW"
                else: return "VERY LOW"
            
            era5_risk = assess_fire_risk(era5_fwi)
            enhanced_risk = assess_fire_risk(enhanced_fwi)
            
            fire_results.append({
                'Model': model_name,
                'ERA5_FWI': era5_fwi,
                'Enhanced_FWI': enhanced_fwi,
                'Difference': difference,
                'ERA5_Risk': era5_risk,
                'Enhanced_Risk': enhanced_risk
            })
            
            print(f"  Enhanced FWI: {enhanced_fwi:.2f}")
            print(f"  Difference: {difference:+.2f}")
            print(f"  Risk Change: {era5_risk} → {enhanced_risk}")
            print()
        
        # Test Ensemble (best 2 models based on previous results)
        print("Testing Ensemble (XGBoost + CNN)...")
        
        # XGBoost prediction
        xgb_pred = self.models['XGBoost'].predict(june16_features)[0]
        
        # CNN prediction  
        june16_scaled = self.scaler.transform(june16_features)
        june16_cnn = june16_scaled.reshape(-1, june16_scaled.shape[1], 1)
        cnn_pred = self.models['CNN'].predict(june16_cnn, verbose=0).flatten()[0]
        
        # Ensemble average
        ensemble_fwi = (xgb_pred + cnn_pred) / 2
        ensemble_diff = ensemble_fwi - era5_fwi
        ensemble_risk = assess_fire_risk(ensemble_fwi)
        
        fire_results.append({
            'Model': 'Ensemble',
            'ERA5_FWI': era5_fwi,
            'Enhanced_FWI': ensemble_fwi,
            'Difference': ensemble_diff,
            'ERA5_Risk': era5_risk,
            'Enhanced_Risk': ensemble_risk
        })
        
        print(f"  Enhanced FWI: {ensemble_fwi:.2f}")
        print(f"  Difference: {ensemble_diff:+.2f}")
        print(f"  Risk Change: {era5_risk} → {ensemble_risk}")
        
        # Save and display results
        results_df = pd.DataFrame(fire_results)
        results_df.to_csv('june16_fire_validation_all_models.csv', index=False)
        
        print("\n" + "="*60)
        print("JUNE 16 FIRE EVENT RESULTS SUMMARY")
        print("="*60)
        print(results_df.round(2).to_string(index=False))
        
        return results_df
        
    def run_validation(self):
        """Run complete June 16 validation"""
        self.load_and_train_models()
        results = self.validate_june16_fire()
        return results

if __name__ == "__main__":
    validator = June16FireValidator()
    results = validator.run_validation()