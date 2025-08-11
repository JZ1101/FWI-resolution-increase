#!/usr/bin/env python3
"""
Complete FWI Resolution Enhancement Experiment 2017 Portugal
ANN, XGBoost, CNN + Ensemble with proper validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingRegressor
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from scipy import stats
import warnings
import json
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class FWIExperiment:
    """Complete FWI resolution enhancement experiment"""
    
    def __init__(self):
        self.data_path = 'experiment_2017_portugal/features_2017_COMPLETE_FINAL.csv'
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and prepare data"""
        print("LOADING DATA")
        print("="*50)
        
        self.data = pd.read_csv(self.data_path)
        self.data['time'] = pd.to_datetime(self.data['time'])
        
        print(f"Dataset: {self.data.shape}")
        
        # Prepare features
        exclude_cols = ['time', 'latitude', 'longitude', 'fwi']
        feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        
        # Remove high missing features
        missing_pct = self.data[feature_cols].isnull().sum() / len(self.data) * 100
        usable_features = missing_pct[missing_pct <= 50].index.tolist()
        
        print(f"Features used: {len(usable_features)}")
        
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
        self.time = self.data['time'].copy()
        
        # Temporal split based on dates
        # Training: Jan 1 - May 1, Validation: May 1 - June 1, Test: June 1 - July 1
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
        self.time_test = self.time.loc[test_indices]
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Train: {self.X_train.shape} (Jan 1 - May 1)")
        print(f"Validation: {self.X_val.shape} (May 1 - June 1)")
        print(f"Test: {self.X_test.shape} (June 1 - July 1)")
        
        # Find June 16 data for fire event validation
        june16_mask = (self.data['time'].dt.date == pd.to_datetime('2017-06-16').date())
        if june16_mask.sum() > 0:
            self.june16_data = self.data[june16_mask].copy()
            self.june16_X = self.X.loc[self.june16_data.index].fillna(0)
            self.june16_y = self.y.loc[self.june16_data.index]
            print(f"June 16 fire event: {self.june16_X.shape[0]} coordinates")
        else:
            print("June 16 data not found")
        
    def train_ann(self):
        """Train ANN"""
        print("\nTRAINING ANN")
        print("="*30)
        
        ann = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=256,
            learning_rate='adaptive',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42
        )
        
        ann.fit(self.X_train_scaled, self.y_train)
        
        y_pred_val = ann.predict(self.X_val_scaled)
        y_pred_test = ann.predict(self.X_test_scaled)
        
        val_rmse = np.sqrt(mean_squared_error(self.y_val, y_pred_val))
        val_r2 = r2_score(self.y_val, y_pred_val)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        test_r2 = r2_score(self.y_test, y_pred_test)
        
        print(f"Validation - RMSE: {val_rmse:.3f}, R²: {val_r2:.3f}")
        print(f"Test - RMSE: {test_rmse:.3f}, R²: {test_r2:.3f}")
        
        self.models['ANN'] = ann
        self.results['ANN'] = {
            'val_rmse': val_rmse, 'val_r2': val_r2,
            'test_rmse': test_rmse, 'test_r2': test_r2,
            'predictions_test': y_pred_test
        }
        
    def train_xgboost(self):
        """Train XGBoost"""
        print("\nTRAINING XGBOOST")
        print("="*30)
        
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
        
        xgb_model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=False
        )
        
        y_pred_val = xgb_model.predict(self.X_val)
        y_pred_test = xgb_model.predict(self.X_test)
        
        val_rmse = np.sqrt(mean_squared_error(self.y_val, y_pred_val))
        val_r2 = r2_score(self.y_val, y_pred_val)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        test_r2 = r2_score(self.y_test, y_pred_test)
        
        print(f"Validation - RMSE: {val_rmse:.3f}, R²: {val_r2:.3f}")
        print(f"Test - RMSE: {test_rmse:.3f}, R²: {test_r2:.3f}")
        
        self.models['XGBoost'] = xgb_model
        self.results['XGBoost'] = {
            'val_rmse': val_rmse, 'val_r2': val_r2,
            'test_rmse': test_rmse, 'test_r2': test_r2,
            'predictions_test': y_pred_test
        }
        
    def train_cnn(self):
        """Train CNN"""
        print("\nTRAINING CNN")
        print("="*30)
        
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
        
        model.fit(
            X_train_cnn, self.y_train,
            validation_data=(X_val_cnn, self.y_val),
            epochs=50,
            batch_size=256,
            callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
            verbose=0
        )
        
        y_pred_val = model.predict(X_val_cnn, verbose=0).flatten()
        y_pred_test = model.predict(X_test_cnn, verbose=0).flatten()
        
        val_rmse = np.sqrt(mean_squared_error(self.y_val, y_pred_val))
        val_r2 = r2_score(self.y_val, y_pred_val)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        test_r2 = r2_score(self.y_test, y_pred_test)
        
        print(f"Validation - RMSE: {val_rmse:.3f}, R²: {val_r2:.3f}")
        print(f"Test - RMSE: {test_rmse:.3f}, R²: {test_r2:.3f}")
        
        self.models['CNN'] = model
        self.results['CNN'] = {
            'val_rmse': val_rmse, 'val_r2': val_r2,
            'test_rmse': test_rmse, 'test_r2': test_r2,
            'predictions_test': y_pred_test
        }
        
    def create_ensemble(self):
        """Create ensemble of 2 best models"""
        print("\nCREATING ENSEMBLE")
        print("="*30)
        
        # Find 2 best models by validation R²
        model_performance = [(name, res['val_r2']) for name, res in self.results.items()]
        model_performance.sort(key=lambda x: x[1], reverse=True)
        
        best_models = model_performance[:2]
        print(f"Best models: {best_models[0][0]} (R²={best_models[0][1]:.3f}), {best_models[1][0]} (R²={best_models[1][1]:.3f})")
        
        # Ensemble predictions (average)
        model1_name, model2_name = best_models[0][0], best_models[1][0]
        
        y_pred_test = (self.results[model1_name]['predictions_test'] + 
                       self.results[model2_name]['predictions_test']) / 2
        
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        test_r2 = r2_score(self.y_test, y_pred_test)
        
        print(f"Ensemble Test - RMSE: {test_rmse:.3f}, R²: {test_r2:.3f}")
        
        self.results['Ensemble'] = {
            'test_rmse': test_rmse, 'test_r2': test_r2,
            'predictions_test': y_pred_test,
            'models': [model1_name, model2_name]
        }
        
    def create_high_res_predictions(self):
        """Create 1km predictions"""
        print("\nCREATING 1KM PREDICTIONS")
        print("="*30)
        
        # Get unique coordinates
        unique_coords = self.coords.drop_duplicates()
        lat_min, lat_max = unique_coords['latitude'].min(), unique_coords['latitude'].max()
        lon_min, lon_max = unique_coords['longitude'].min(), unique_coords['longitude'].max()
        
        # Create 1km grid (0.01 degrees ≈ 1km)
        lats_1km = np.arange(lat_min, lat_max + 0.01, 0.01)
        lons_1km = np.arange(lon_min, lon_max + 0.01, 0.01)
        
        print(f"25km grid: {len(unique_coords)} points")
        print(f"1km grid: {len(lats_1km) * len(lons_1km)} points")
        
        # Sample for demo
        sample_lats = np.random.choice(lats_1km, size=min(100, len(lats_1km)), replace=False)
        sample_lons = np.random.choice(lons_1km, size=min(100, len(lons_1km)), replace=False)
        
        high_res_coords = []
        for lat in sample_lats:
            for lon in sample_lons:
                high_res_coords.append({'latitude': lat, 'longitude': lon})
        
        self.high_res_df = pd.DataFrame(high_res_coords)
        
        # Create features for high-res grid
        high_res_features = pd.DataFrame(index=range(len(self.high_res_df)))
        
        for col in self.X.columns:
            if any(x in col for x in ['lat', 'lon', 'dist', 'region', 'zone']):
                if col == 'lat_norm':
                    high_res_features[col] = (self.high_res_df['latitude'] - lat_min) / (lat_max - lat_min)
                elif col == 'lon_norm':
                    high_res_features[col] = (self.high_res_df['longitude'] - lon_min) / (lon_max - lon_min)
                else:
                    high_res_features[col] = self.X[col].mean()
            else:
                high_res_features[col] = self.X[col].mean()
        
        high_res_features = high_res_features.fillna(0)
        
        # Make predictions with ensemble
        model1_name, model2_name = self.results['Ensemble']['models']
        
        if model1_name == 'ANN':
            X_scaled = self.scaler.transform(high_res_features)
            pred1 = self.models[model1_name].predict(X_scaled)
        elif model1_name == 'CNN':
            X_scaled = self.scaler.transform(high_res_features)
            X_scaled = X_scaled.reshape(-1, X_scaled.shape[1], 1)
            pred1 = self.models[model1_name].predict(X_scaled, verbose=0).flatten()
        else:
            pred1 = self.models[model1_name].predict(high_res_features)
            
        if model2_name == 'ANN':
            X_scaled = self.scaler.transform(high_res_features)
            pred2 = self.models[model2_name].predict(X_scaled)
        elif model2_name == 'CNN':
            X_scaled = self.scaler.transform(high_res_features)
            X_scaled = X_scaled.reshape(-1, X_scaled.shape[1], 1)
            pred2 = self.models[model2_name].predict(X_scaled, verbose=0).flatten()
        else:
            pred2 = self.models[model2_name].predict(high_res_features)
            
        self.high_res_df['fwi_predicted'] = (pred1 + pred2) / 2
        
        print(f"Created {len(self.high_res_df)} 1km predictions")
        print(f"Mean: {self.high_res_df['fwi_predicted'].mean():.2f}, Std: {self.high_res_df['fwi_predicted'].std():.2f}")
        
    def back_aggregation_validation(self):
        """Back-aggregation validation"""
        print("\nBACK-AGGREGATION VALIDATION")
        print("="*40)
        
        coords_25km = self.coords_test.drop_duplicates()
        
        back_aggregated = []
        original_25km = []
        
        for idx, row in coords_25km.iterrows():
            lat_25km = row['latitude']
            lon_25km = row['longitude']
            
            # Find 1km points in this 25km cell
            mask = (
                (self.high_res_df['latitude'] >= lat_25km - 0.125) &
                (self.high_res_df['latitude'] < lat_25km + 0.125) &
                (self.high_res_df['longitude'] >= lon_25km - 0.125) &
                (self.high_res_df['longitude'] < lon_25km + 0.125)
            )
            
            if mask.sum() > 0:
                cell_mean = self.high_res_df[mask]['fwi_predicted'].mean()
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
            
            print(f"Correlation: {correlation:.3f}")
            print(f"RMSE: {rmse:.3f}")
            print(f"Cells compared: {len(back_aggregated)}")
            
            self.validation_results = {
                'back_aggregation': {
                    'correlation': correlation,
                    'rmse': rmse,
                    'n_cells': len(back_aggregated)
                }
            }
        else:
            print("Insufficient data for validation")
            
    def spatial_correlation_analysis(self):
        """Spatial correlation analysis"""
        print("\nSPATIAL CORRELATION ANALYSIS")
        print("="*40)
        
        # Variance analysis
        var_25km = self.y_test.var()
        var_1km = self.high_res_df['fwi_predicted'].var()
        variance_ratio = var_1km / var_25km
        
        print(f"25km variance: {var_25km:.2f}")
        print(f"1km variance: {var_1km:.2f}")
        print(f"Variance ratio (1km/25km): {variance_ratio:.3f}")
        
        if 'validation_results' not in self.__dict__:
            self.validation_results = {}
            
        self.validation_results['spatial_correlation'] = {
            'variance_25km': var_25km,
            'variance_1km': var_1km,
            'variance_ratio': variance_ratio
        }
        
    def cross_scale_validation(self):
        """Cross-scale validation"""
        print("\nCROSS-SCALE VALIDATION")
        print("="*40)
        
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
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                
                model.fit(self.X[train_mask], self.y[train_mask])
                y_pred = model.predict(self.X[test_mask])
                y_true = self.y[test_mask]
                
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)
                
                cross_val_results.append({
                    'region': f"{test_lat_min}°-{test_lat_max}°N",
                    'rmse': rmse,
                    'r2': r2,
                    'n_test': test_mask.sum()
                })
                
                print(f"Region {test_lat_min}°-{test_lat_max}°N: RMSE={rmse:.2f}, R²={r2:.3f}")
        
        if cross_val_results:
            avg_r2 = np.mean([r['r2'] for r in cross_val_results])
            print(f"Average R²: {avg_r2:.3f}")
            
            if 'validation_results' not in self.__dict__:
                self.validation_results = {}
                
            self.validation_results['cross_scale'] = {
                'regional_results': cross_val_results,
                'avg_r2': avg_r2
            }
            
    def june16_fire_validation(self):
        """Validate against June 16 fire event"""
        print("\nJUNE 16 FIRE EVENT VALIDATION")
        print("="*40)
        
        if not hasattr(self, 'june16_data'):
            print("No June 16 data available")
            return
            
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
        
        # Predict with ensemble model
        model1_name, model2_name = self.results['Ensemble']['models']
        
        # Get features for June 16 at fire location
        june16_features = self.june16_X.loc[closest_idx:closest_idx]
        
        if model1_name == 'ANN':
            june16_scaled = self.scaler.transform(june16_features)
            pred1 = self.models[model1_name].predict(june16_scaled)[0]
        elif model1_name == 'CNN':
            june16_scaled = self.scaler.transform(june16_features)
            june16_cnn = june16_scaled.reshape(-1, june16_scaled.shape[1], 1)
            pred1 = self.models[model1_name].predict(june16_cnn, verbose=0).flatten()[0]
        else:
            pred1 = self.models[model1_name].predict(june16_features)[0]
            
        if model2_name == 'ANN':
            june16_scaled = self.scaler.transform(june16_features)
            pred2 = self.models[model2_name].predict(june16_scaled)[0]
        elif model2_name == 'CNN':
            june16_scaled = self.scaler.transform(june16_features)
            june16_cnn = june16_scaled.reshape(-1, june16_scaled.shape[1], 1)
            pred2 = self.models[model2_name].predict(june16_cnn, verbose=0).flatten()[0]
        else:
            pred2 = self.models[model2_name].predict(june16_features)[0]
            
        enhanced_fwi = (pred1 + pred2) / 2
        
        print(f"Fire location: {fire_lat}°N, {fire_lon}°W")
        print(f"Closest coordinate: {closest_coord['latitude']:.3f}°N, {closest_coord['longitude']:.3f}°W")
        print(f"Distance to fire: {distances.loc[closest_idx]*111:.1f} km")
        print(f"ERA5 FWI (25km): {era5_fwi:.2f}")
        print(f"Enhanced FWI (1km): {enhanced_fwi:.2f}")
        print(f"Difference: {enhanced_fwi - era5_fwi:.2f}")
        
        # Risk assessment
        def assess_fire_risk(fwi):
            if fwi >= 38: return "EXTREME"
            elif fwi >= 21.3: return "HIGH" 
            elif fwi >= 11.2: return "MODERATE"
            elif fwi >= 5.2: return "LOW"
            else: return "VERY LOW"
            
        era5_risk = assess_fire_risk(era5_fwi)
        enhanced_risk = assess_fire_risk(enhanced_fwi)
        
        print(f"ERA5 Risk Level: {era5_risk}")
        print(f"Enhanced Risk Level: {enhanced_risk}")
        
        if 'validation_results' not in self.__dict__:
            self.validation_results = {}
            
        self.validation_results['june16_fire'] = {
            'fire_location': [fire_lat, fire_lon],
            'closest_coord': [float(closest_coord['latitude']), float(closest_coord['longitude'])],
            'distance_km': float(distances.loc[closest_idx]*111),
            'era5_fwi': float(era5_fwi),
            'enhanced_fwi': float(enhanced_fwi),
            'difference': float(enhanced_fwi - era5_fwi),
            'era5_risk': era5_risk,
            'enhanced_risk': enhanced_risk
        }
            
    def save_results(self):
        """Save results"""
        print("\nSAVING RESULTS")
        print("="*30)
        
        # Model comparison
        comparison_df = pd.DataFrame([
            {
                'Model': name,
                'Test_RMSE': res['test_rmse'],
                'Test_R2': res['test_r2']
            }
            for name, res in self.results.items()
        ])
        comparison_df.to_csv('model_results.csv', index=False)
        print("Saved: model_results.csv")
        
        # High-res predictions
        self.high_res_df.to_csv('predictions_1km.csv', index=False)
        print("Saved: predictions_1km.csv")
        
        # Validation results
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
        
        validation_clean = convert_numpy(self.validation_results)
        with open('validation_results.json', 'w') as f:
            json.dump(validation_clean, f, indent=2)
        print("Saved: validation_results.json")
        
    def run_experiment(self):
        """Run complete experiment"""
        print("FWI RESOLUTION ENHANCEMENT EXPERIMENT")
        print("="*60)
        
        self.load_data()
        
        # Train models
        self.train_ann()
        self.train_xgboost()
        self.train_cnn()
        
        # Create ensemble
        self.create_ensemble()
        
        # High-res predictions
        self.create_high_res_predictions()
        
        # Validation
        self.back_aggregation_validation()
        self.spatial_correlation_analysis()
        self.cross_scale_validation()
        self.june16_fire_validation()
        
        # Save
        self.save_results()
        
        print("\nEXPERIMENT COMPLETED")
        print("="*60)

if __name__ == "__main__":
    experiment = FWIExperiment()
    experiment.run_experiment()