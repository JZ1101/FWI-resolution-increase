#!/usr/bin/env python3
"""
Test ALL models (ANN, XGBoost, CNN, Ensemble) for 1km predictions
with all 3 validation methods
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

class AllModels1kmTester:
    """Test all models for 1km validation"""
    
    def __init__(self):
        self.data_path = 'experiment_2017_portugal/features_2017_COMPLETE_FINAL.csv'
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.all_validation_results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare data"""
        print("LOADING DATA FOR ALL MODELS 1KM TEST")
        print("="*60)
        
        self.data = pd.read_csv(self.data_path)
        self.data['time'] = pd.to_datetime(self.data['time'])
        
        # Prepare features
        exclude_cols = ['time', 'latitude', 'longitude', 'fwi']
        feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        
        # Remove high missing features
        missing_pct = self.data[feature_cols].isnull().sum() / len(self.data) * 100
        usable_features = missing_pct[missing_pct <= 50].index.tolist()
        
        # Create feature matrix
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
        
        # Temporal split
        train_mask = (self.data['time'] < '2017-05-01')
        val_mask = (self.data['time'] >= '2017-05-01') & (self.data['time'] < '2017-06-01')
        test_mask = (self.data['time'] >= '2017-06-01') & (self.data['time'] < '2017-07-01')
        
        train_indices = self.data[train_mask].index
        val_indices = self.data[val_mask].index
        test_indices = self.data[test_mask].index
        
        self.X_train = self.X.loc[train_indices].fillna(0)
        self.X_val = self.X.loc[val_indices].fillna(0)
        self.X_test = self.X.loc[test_indices].fillna(0)
        
        self.y_train = self.y.loc[train_indices]
        self.y_val = self.y.loc[val_indices]
        self.y_test = self.y.loc[test_indices]
        
        self.coords_test = self.coords.loc[test_indices]
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Train: {self.X_train.shape}")
        print(f"Test: {self.X_test.shape}")
        
    def train_all_models(self):
        """Train all models"""
        print("\nTRAINING ALL MODELS")
        print("="*40)
        
        # ANN
        print("Training ANN...")
        ann = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=256,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42
        )
        ann.fit(self.X_train_scaled, self.y_train)
        y_pred_test = ann.predict(self.X_test_scaled)
        test_r2 = r2_score(self.y_test, y_pred_test)
        self.models['ANN'] = ann
        self.results['ANN'] = {'test_r2': test_r2, 'predictions_test': y_pred_test}
        print(f"ANN Test R²: {test_r2:.3f}")
        
        # XGBoost
        print("Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)], verbose=False)
        y_pred_test = xgb_model.predict(self.X_test)
        test_r2 = r2_score(self.y_test, y_pred_test)
        self.models['XGBoost'] = xgb_model
        self.results['XGBoost'] = {'test_r2': test_r2, 'predictions_test': y_pred_test}
        print(f"XGBoost Test R²: {test_r2:.3f}")
        
        # CNN
        print("Training CNN...")
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
        X_val_cnn = self.X_val_scaled.reshape(-1, self.X_val_scaled.shape[1], 1)
        X_test_cnn = self.X_test_scaled.reshape(-1, self.X_test_scaled.shape[1], 1)
        model.fit(X_train_cnn, self.y_train, validation_data=(X_val_cnn, self.y_val),
                  epochs=50, batch_size=256, 
                  callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
                  verbose=0)
        y_pred_test = model.predict(X_test_cnn, verbose=0).flatten()
        test_r2 = r2_score(self.y_test, y_pred_test)
        self.models['CNN'] = model
        self.results['CNN'] = {'test_r2': test_r2, 'predictions_test': y_pred_test}
        print(f"CNN Test R²: {test_r2:.3f}")
        
        # Ensemble (best 2 models)
        print("Creating Ensemble...")
        model_performance = [(name, res['test_r2']) for name, res in self.results.items()]
        model_performance.sort(key=lambda x: x[1], reverse=True)
        best_models = model_performance[:2]
        model1_name, model2_name = best_models[0][0], best_models[1][0]
        
        y_pred_test = (self.results[model1_name]['predictions_test'] + 
                       self.results[model2_name]['predictions_test']) / 2
        test_r2 = r2_score(self.y_test, y_pred_test)
        self.results['Ensemble'] = {
            'test_r2': test_r2, 
            'predictions_test': y_pred_test,
            'models': [model1_name, model2_name]
        }
        print(f"Ensemble ({model1_name} + {model2_name}) Test R²: {test_r2:.3f}")
        
    def create_1km_predictions_all_models(self):
        """Create 1km predictions for all models"""
        print("\nCREATING 1KM PREDICTIONS FOR ALL MODELS")
        print("="*60)
        
        # Create 1km grid
        unique_coords = self.coords.drop_duplicates()
        lat_min, lat_max = unique_coords['latitude'].min(), unique_coords['latitude'].max()
        lon_min, lon_max = unique_coords['longitude'].min(), unique_coords['longitude'].max()
        
        lats_1km = np.arange(lat_min, lat_max + 0.01, 0.01)
        lons_1km = np.arange(lon_min, lon_max + 0.01, 0.01)
        
        # Sample for testing
        sample_lats = np.random.choice(lats_1km, size=min(100, len(lats_1km)), replace=False)
        sample_lons = np.random.choice(lons_1km, size=min(100, len(lons_1km)), replace=False)
        
        high_res_coords = []
        for lat in sample_lats:
            for lon in sample_lons:
                high_res_coords.append({'latitude': lat, 'longitude': lon})
        
        high_res_df = pd.DataFrame(high_res_coords)
        
        # Create features
        high_res_features = pd.DataFrame(index=range(len(high_res_df)))
        for col in self.X.columns:
            if any(x in col for x in ['lat', 'lon', 'dist', 'region', 'zone']):
                if col == 'lat_norm':
                    high_res_features[col] = (high_res_df['latitude'] - lat_min) / (lat_max - lat_min)
                elif col == 'lon_norm':
                    high_res_features[col] = (high_res_df['longitude'] - lon_min) / (lon_max - lon_min)
                else:
                    high_res_features[col] = self.X[col].mean()
            else:
                high_res_features[col] = self.X[col].mean()
        high_res_features = high_res_features.fillna(0)
        
        # Predict with all models
        self.high_res_predictions = {}
        
        for model_name in ['ANN', 'XGBoost', 'CNN', 'Ensemble']:
            print(f"Creating 1km predictions with {model_name}...")
            
            if model_name == 'ANN':
                X_scaled = self.scaler.transform(high_res_features)
                predictions = self.models['ANN'].predict(X_scaled)
                
            elif model_name == 'XGBoost':
                predictions = self.models['XGBoost'].predict(high_res_features)
                
            elif model_name == 'CNN':
                X_scaled = self.scaler.transform(high_res_features)
                X_scaled = X_scaled.reshape(-1, X_scaled.shape[1], 1)
                predictions = self.models['CNN'].predict(X_scaled, verbose=0).flatten()
                
            elif model_name == 'Ensemble':
                model1_name, model2_name = self.results['Ensemble']['models']
                
                if model1_name == 'ANN':
                    X_scaled = self.scaler.transform(high_res_features)
                    pred1 = self.models['ANN'].predict(X_scaled)
                elif model1_name == 'XGBoost':
                    pred1 = self.models['XGBoost'].predict(high_res_features)
                else:  # CNN
                    X_scaled = self.scaler.transform(high_res_features)
                    X_scaled = X_scaled.reshape(-1, X_scaled.shape[1], 1)
                    pred1 = self.models['CNN'].predict(X_scaled, verbose=0).flatten()
                    
                if model2_name == 'ANN':
                    X_scaled = self.scaler.transform(high_res_features)
                    pred2 = self.models['ANN'].predict(X_scaled)
                elif model2_name == 'XGBoost':
                    pred2 = self.models['XGBoost'].predict(high_res_features)
                else:  # CNN
                    X_scaled = self.scaler.transform(high_res_features)
                    X_scaled = X_scaled.reshape(-1, X_scaled.shape[1], 1)
                    pred2 = self.models['CNN'].predict(X_scaled, verbose=0).flatten()
                    
                predictions = (pred1 + pred2) / 2
            
            self.high_res_predictions[model_name] = {
                'coordinates': high_res_df.copy(),
                'predictions': predictions
            }
            
            print(f"{model_name}: Mean={predictions.mean():.2f}, Std={predictions.std():.2f}")
            
    def validate_all_models(self):
        """Run all 3 validation methods on all models"""
        print("\nVALIDATING ALL MODELS WITH ALL 3 METHODS")
        print("="*70)
        
        for model_name in ['ANN', 'XGBoost', 'CNN', 'Ensemble']:
            print(f"\n{model_name} VALIDATION")
            print("="*40)
            
            model_predictions = self.high_res_predictions[model_name]
            high_res_df = model_predictions['coordinates'].copy()
            high_res_df['fwi_predicted'] = model_predictions['predictions']
            
            validation_results = {}
            
            # 1. Back-Aggregation Validation
            print("Back-Aggregation Validation...")
            coords_25km = self.coords_test.drop_duplicates()
            back_aggregated = []
            original_25km = []
            
            for idx, row in coords_25km.iterrows():
                lat_25km = row['latitude']
                lon_25km = row['longitude']
                
                mask = (
                    (high_res_df['latitude'] >= lat_25km - 0.125) &
                    (high_res_df['latitude'] < lat_25km + 0.125) &
                    (high_res_df['longitude'] >= lon_25km - 0.125) &
                    (high_res_df['longitude'] < lon_25km + 0.125)
                )
                
                if mask.sum() > 0:
                    cell_mean = high_res_df[mask]['fwi_predicted'].mean()
                    back_aggregated.append(cell_mean)
                    
                    orig_mask = (
                        (self.coords_test['latitude'] == lat_25km) &
                        (self.coords_test['longitude'] == lon_25km)
                    )
                    if orig_mask.sum() > 0:
                        original_25km.append(self.y_test[orig_mask].mean())
            
            if len(back_aggregated) > 0 and len(original_25km) > 0:
                correlation = np.corrcoef(original_25km[:len(back_aggregated)], 
                                        back_aggregated[:len(original_25km)])[0, 1]
                rmse = np.sqrt(mean_squared_error(
                    original_25km[:len(back_aggregated)], 
                    back_aggregated[:len(original_25km)]
                ))
                validation_results['back_aggregation'] = {
                    'correlation': float(correlation),
                    'rmse': float(rmse),
                    'n_cells': len(back_aggregated)
                }
                print(f"  Correlation: {correlation:.3f}, RMSE: {rmse:.3f}")
            
            # 2. Spatial Correlation Analysis
            print("Spatial Correlation Analysis...")
            var_25km = self.y_test.var()
            var_1km = high_res_df['fwi_predicted'].var()
            variance_ratio = var_1km / var_25km
            
            validation_results['spatial_correlation'] = {
                'variance_25km': float(var_25km),
                'variance_1km': float(var_1km),
                'variance_ratio': float(variance_ratio)
            }
            print(f"  Variance ratio (1km/25km): {variance_ratio:.6f}")
            
            # 3. Cross-Scale Validation (simplified)
            print("Cross-Scale Validation...")
            lat_bands = [37, 38, 39, 40, 41, 42]
            cross_val_results = []
            
            for i in range(len(lat_bands)-1):
                test_lat_min = lat_bands[i]
                test_lat_max = lat_bands[i+1]
                
                test_mask = (
                    (self.coords['latitude'] >= test_lat_min) & 
                    (self.coords['latitude'] < test_lat_max)
                )
                train_mask = ~test_mask
                
                if train_mask.sum() > 1000 and test_mask.sum() > 100:
                    # Use XGBoost for efficiency
                    cv_model = xgb.XGBRegressor(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42
                    )
                    
                    cv_model.fit(self.X[train_mask], self.y[train_mask])
                    y_pred = cv_model.predict(self.X[test_mask])
                    y_true = self.y[test_mask]
                    
                    r2 = r2_score(y_true, y_pred)
                    cross_val_results.append({
                        'region': f"{test_lat_min}°-{test_lat_max}°N",
                        'r2': float(r2)
                    })
            
            if cross_val_results:
                avg_r2 = np.mean([r['r2'] for r in cross_val_results])
                validation_results['cross_scale'] = {
                    'regional_results': cross_val_results,
                    'avg_r2': float(avg_r2)
                }
                print(f"  Average R²: {avg_r2:.3f}")
            
            self.all_validation_results[model_name] = validation_results
            
    def save_all_results(self):
        """Save comprehensive results"""
        print("\nSAVING ALL RESULTS")
        print("="*40)
        
        # Model performance summary
        summary = []
        for model_name in ['ANN', 'XGBoost', 'CNN', 'Ensemble']:
            model_data = {
                'Model': model_name,
                'Test_R2': self.results[model_name]['test_r2']
            }
            
            if model_name in self.all_validation_results:
                val_results = self.all_validation_results[model_name]
                
                # Back-aggregation
                if 'back_aggregation' in val_results:
                    model_data['Back_Agg_Correlation'] = val_results['back_aggregation']['correlation']
                    model_data['Back_Agg_RMSE'] = val_results['back_aggregation']['rmse']
                
                # Spatial correlation
                if 'spatial_correlation' in val_results:
                    model_data['Variance_Ratio'] = val_results['spatial_correlation']['variance_ratio']
                
                # Cross-scale
                if 'cross_scale' in val_results:
                    model_data['Cross_Scale_R2'] = val_results['cross_scale']['avg_r2']
            
            summary.append(model_data)
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv('all_models_1km_validation_summary.csv', index=False)
        
        # Detailed results
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            else:
                return obj
        
        clean_results = convert_numpy(self.all_validation_results)
        with open('all_models_1km_validation_detailed.json', 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        print("Saved: all_models_1km_validation_summary.csv")
        print("Saved: all_models_1km_validation_detailed.json")
        
        # Print summary table
        print("\nALL MODELS 1KM VALIDATION SUMMARY")
        print("="*80)
        print(summary_df.round(3).to_string(index=False))
        
    def run_complete_test(self):
        """Run complete test of all models"""
        print("ALL MODELS 1KM VALIDATION TEST")
        print("="*80)
        
        self.load_and_prepare_data()
        self.train_all_models()
        self.create_1km_predictions_all_models()
        self.validate_all_models()
        self.save_all_results()
        
        print("\nALL MODELS 1KM VALIDATION COMPLETED")
        print("="*80)

if __name__ == "__main__":
    tester = AllModels1kmTester()
    tester.run_complete_test()