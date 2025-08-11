#!/usr/bin/env python3
"""
Correct June 16 Fire Validation
Get exact ERA5 FWI values and model predictions for fire region
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

class CorrectJune16Validator:
    """Correct June 16 validation with proper ERA5 values"""
    
    def __init__(self):
        self.data_path = 'experiment_2017_portugal/features_2017_COMPLETE_FINAL.csv'
        self.models = {}
        self.scaler = StandardScaler()
        
    def load_data_and_find_fire_location(self):
        """Load data and find exact fire location data"""
        print("CORRECT JUNE 16 FIRE VALIDATION")
        print("="*60)
        
        self.data = pd.read_csv(self.data_path)
        self.data['time'] = pd.to_datetime(self.data['time'])
        
        # Find June 16 data
        june16_mask = (self.data['time'].dt.date == pd.to_datetime('2017-06-16').date())
        self.june16_data = self.data[june16_mask].copy()
        
        print(f"June 16 data: {len(self.june16_data)} coordinates")
        
        # Pedrógão Grande fire location: 39.92°N, -8.15°W
        fire_lat, fire_lon = 39.92, -8.15
        
        # Find closest coordinate in June 16 data
        distances = np.sqrt(
            (self.june16_data['latitude'] - fire_lat)**2 + 
            (self.june16_data['longitude'] - fire_lon)**2
        )
        closest_idx = distances.idxmin()
        self.fire_location_data = self.june16_data.loc[closest_idx]
        
        print(f"\nFIRE LOCATION ANALYSIS:")
        print(f"Target Fire Location: {fire_lat}°N, {fire_lon}°W")
        print(f"Closest ERA5 Grid Point: {self.fire_location_data['latitude']:.3f}°N, {self.fire_location_data['longitude']:.3f}°W")
        print(f"Distance: {distances.loc[closest_idx]*111:.1f} km")
        print(f"ERA5 FWI on June 16: {self.fire_location_data['fwi']:.2f}")
        
        # Also show surrounding points for context
        print(f"\nSURROUNDING ERA5 GRID POINTS ON JUNE 16:")
        nearby_mask = distances <= distances.quantile(0.1)  # Top 10% closest points
        nearby_data = self.june16_data[nearby_mask].sort_values('fwi', ascending=False)
        
        for idx, row in nearby_data.head(5).iterrows():
            dist = np.sqrt((row['latitude'] - fire_lat)**2 + (row['longitude'] - fire_lon)**2) * 111
            print(f"  {row['latitude']:.3f}°N, {row['longitude']:.3f}°W: FWI={row['fwi']:.2f} (distance: {dist:.1f}km)")
        
        return self.fire_location_data['fwi']
        
    def prepare_training_data(self):
        """Prepare training data (excluding June 16)"""
        print(f"\nPREPARING TRAINING DATA:")
        
        # Exclude June 16 from training (proper temporal validation)
        training_mask = self.data['time'].dt.date != pd.to_datetime('2017-06-16').date()
        training_data = self.data[training_mask].copy()
        
        print(f"Total data: {len(self.data)} samples")
        print(f"Training data (excluding June 16): {len(training_data)} samples")
        
        # Prepare features
        exclude_cols = ['time', 'latitude', 'longitude', 'fwi']
        feature_cols = [col for col in training_data.columns if col not in exclude_cols]
        
        # Remove high missing features
        missing_pct = training_data[feature_cols].isnull().sum() / len(training_data) * 100
        usable_features = missing_pct[missing_pct <= 50].index.tolist()
        
        X_temp = training_data[usable_features].copy()
        for col in X_temp.columns:
            if X_temp[col].dtype == 'object':
                try:
                    X_temp[col] = pd.to_numeric(X_temp[col], errors='coerce')
                except:
                    X_temp = X_temp.drop(col, axis=1)
        
        self.X_train = X_temp.fillna(X_temp.mean())
        self.y_train = training_data['fwi']
        
        # Prepare June 16 features using the same processing
        june16_features = self.data[self.data['time'].dt.date == pd.to_datetime('2017-06-16').date()]
        X_june16_temp = june16_features[usable_features].copy()
        for col in X_june16_temp.columns:
            if X_june16_temp[col].dtype == 'object':
                try:
                    X_june16_temp[col] = pd.to_numeric(X_june16_temp[col], errors='coerce')
                except:
                    X_june16_temp[col] = X_june16_temp.get(col, 0)  # Handle missing columns
        
        # Fill with training data means
        for col in self.X_train.columns:
            if col in X_june16_temp.columns:
                X_june16_temp[col] = X_june16_temp[col].fillna(self.X_train[col].mean())
            else:
                X_june16_temp[col] = self.X_train[col].mean()
        
        self.X_june16 = X_june16_temp[self.X_train.columns]
        
        # Scale features
        self.scaler.fit(self.X_train)
        self.X_train_scaled = self.scaler.transform(self.X_train)
        self.X_june16_scaled = self.scaler.transform(self.X_june16)
        
        print(f"Feature dimensions: {self.X_train.shape[1]} features")
        
    def train_all_models(self):
        """Train all models on data excluding June 16"""
        print(f"\nTRAINING MODELS (excluding June 16):")
        
        # ANN
        print("  Training ANN...")
        ann = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu', solver='adam', alpha=0.001,
            max_iter=300, random_state=42
        )
        ann.fit(self.X_train_scaled, self.y_train)
        self.models['ANN'] = ann
        
        # XGBoost
        print("  Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=200, max_depth=8, learning_rate=0.05,
            random_state=42, n_jobs=-1
        )
        xgb_model.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = xgb_model
        
        # CNN
        print("  Training CNN...")
        model = keras.Sequential([
            keras.layers.Input(shape=(self.X_train_scaled.shape[1], 1)),
            keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
            keras.layers.MaxPooling1D(2),
            keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        X_train_cnn = self.X_train_scaled.reshape(-1, self.X_train_scaled.shape[1], 1)
        model.fit(X_train_cnn, self.y_train, epochs=30, batch_size=256, verbose=0)
        self.models['CNN'] = model
        
        print("  All models trained successfully")
        
    def predict_june16_fire_location(self):
        """Make predictions for exact fire location on June 16"""
        print(f"\nPREDICTING FIRE LOCATION ON JUNE 16:")
        
        # Find the exact row for our fire location
        fire_lat, fire_lon = 39.92, -8.15
        distances = np.sqrt(
            (self.june16_data['latitude'] - fire_lat)**2 + 
            (self.june16_data['longitude'] - fire_lon)**2
        )
        closest_idx = distances.idxmin()
        
        # Get the features for this specific location
        fire_location_features = self.X_june16.loc[closest_idx:closest_idx]
        fire_location_scaled = self.X_june16_scaled[self.june16_data.index.get_loc(closest_idx):self.june16_data.index.get_loc(closest_idx)+1]
        
        # Get ERA5 FWI for comparison
        era5_fwi = self.fire_location_data['fwi']
        
        results = []
        
        # Predict with each model
        for model_name in ['ANN', 'XGBoost', 'CNN']:
            print(f"  Predicting with {model_name}...")
            
            if model_name == 'ANN':
                prediction = self.models['ANN'].predict(fire_location_scaled)[0]
                
            elif model_name == 'XGBoost':
                prediction = self.models['XGBoost'].predict(fire_location_features)[0]
                
            elif model_name == 'CNN':
                fire_cnn = fire_location_scaled.reshape(-1, fire_location_scaled.shape[1], 1)
                prediction = self.models['CNN'].predict(fire_cnn, verbose=0).flatten()[0]
            
            difference = prediction - era5_fwi
            
            results.append({
                'Model': model_name,
                'ERA5_FWI': era5_fwi,
                'Predicted_FWI': prediction,
                'Difference': difference
            })
        
        # Ensemble (XGBoost + CNN average)
        print("  Creating Ensemble prediction...")
        xgb_pred = self.models['XGBoost'].predict(fire_location_features)[0]
        
        fire_cnn = fire_location_scaled.reshape(-1, fire_location_scaled.shape[1], 1)
        cnn_pred = self.models['CNN'].predict(fire_cnn, verbose=0).flatten()[0]
        
        ensemble_pred = (xgb_pred + cnn_pred) / 2
        ensemble_diff = ensemble_pred - era5_fwi
        
        results.append({
            'Model': 'Ensemble',
            'ERA5_FWI': era5_fwi,
            'Predicted_FWI': ensemble_pred,
            'Difference': ensemble_diff
        })
        
        return results
        
    def assess_fire_risk(self, fwi):
        """Assess fire risk level"""
        if fwi >= 38: return "EXTREME"
        elif fwi >= 21.3: return "HIGH" 
        elif fwi >= 11.2: return "MODERATE"
        elif fwi >= 5.2: return "LOW"
        else: return "VERY LOW"
        
    def display_results(self, results):
        """Display final results"""
        print(f"\nJUNE 16 PEDRÓGÃO GRANDE FIRE - FINAL RESULTS")
        print("="*80)
        
        # Add risk assessment
        for result in results:
            result['ERA5_Risk'] = self.assess_fire_risk(result['ERA5_FWI'])
            result['Predicted_Risk'] = self.assess_fire_risk(result['Predicted_FWI'])
            result['Risk_Change'] = f"{result['ERA5_Risk']} → {result['Predicted_Risk']}"
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv('june16_fire_correct_validation.csv', index=False)
        
        # Display table
        print("\nFIRE LOCATION PREDICTIONS vs ERA5 FWI:")
        print("-" * 80)
        print(f"{'Model':<12} {'ERA5_FWI':<10} {'Pred_FWI':<10} {'Difference':<12} {'ERA5_Risk':<10} {'Pred_Risk':<10} {'Risk_Change':<15}")
        print("-" * 80)
        
        for result in results:
            print(f"{result['Model']:<12} {result['ERA5_FWI']:<10.2f} {result['Predicted_FWI']:<10.2f} "
                  f"{result['Difference']:<12.2f} {result['ERA5_Risk']:<10} {result['Predicted_Risk']:<10} "
                  f"{result['Risk_Change']:<15}")
        
        print("-" * 80)
        
        # Analysis
        print(f"\nANALYSIS:")
        underestimated = [r for r in results if r['Difference'] < 0]
        overestimated = [r for r in results if r['Difference'] > 0]
        
        print(f"• {len(underestimated)} models underestimated fire risk")
        print(f"• {len(overestimated)} models overestimated fire risk") 
        
        if underestimated:
            worst_underestimate = min(underestimated, key=lambda x: x['Difference'])
            print(f"• Worst underestimation: {worst_underestimate['Model']} ({worst_underestimate['Difference']:.2f} FWI units)")
            
        return results_df
        
    def run_correct_validation(self):
        """Run complete correct validation"""
        era5_fwi = self.load_data_and_find_fire_location()
        self.prepare_training_data()
        self.train_all_models()
        results = self.predict_june16_fire_location()
        results_df = self.display_results(results)
        
        return results_df

if __name__ == "__main__":
    validator = CorrectJune16Validator()
    results = validator.run_correct_validation()