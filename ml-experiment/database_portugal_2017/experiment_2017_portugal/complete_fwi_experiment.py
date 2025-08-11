#!/usr/bin/env python3
"""
Complete FWI Resolution Enhancement Experiment
With proper validation methods and multiple models
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
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class FWIResolutionEnhancement:
    """Complete FWI resolution enhancement with proper validation"""
    
    def __init__(self, data_path='features_2017_COMPLETE_FINAL.csv'):
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def load_and_prepare_data(self):
        """Load data and prepare features"""
        print("="*80)
        print("LOADING AND PREPARING DATA")
        print("="*80)
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        self.data['time'] = pd.to_datetime(self.data['time'])
        
        print(f"Dataset shape: {self.data.shape}")
        
        # Identify usable features (exclude identifiers and target)
        exclude_cols = ['time', 'latitude', 'longitude', 'fwi']
        feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        
        # Remove features with >50% missing
        missing_pct = self.data[feature_cols].isnull().sum() / len(self.data) * 100
        usable_features = missing_pct[missing_pct <= 50].index.tolist()
        
        print(f"Usable features: {len(usable_features)}")
        
        # Prepare feature matrix
        # Convert non-numeric columns to numeric
        X_temp = self.data[usable_features].copy()
        for col in X_temp.columns:
            if X_temp[col].dtype == 'object':
                try:
                    X_temp[col] = pd.to_numeric(X_temp[col], errors='coerce')
                except:
                    # Drop non-numeric columns
                    X_temp = X_temp.drop(col, axis=1)
        
        self.X = X_temp.fillna(X_temp.mean())
        self.y = self.data['fwi']
        self.coords = self.data[['latitude', 'longitude']].copy()
        self.time = self.data['time'].copy()
        
        # Create temporal split (70% train, 15% val, 15% test)
        self.data_sorted = self.data.sort_values('time').reset_index(drop=True)
        train_end = int(0.7 * len(self.X))
        val_end = int(0.85 * len(self.X))
        
        self.X_train = self.X[:train_end].copy()
        self.X_val = self.X[train_end:val_end].copy()
        self.X_test = self.X[val_end:].copy()
        
        self.y_train = self.y[:train_end].copy()
        self.y_val = self.y[train_end:val_end].copy()
        self.y_test = self.y[val_end:].copy()
        
        self.coords_train = self.coords[:train_end].copy()
        self.coords_val = self.coords[train_end:val_end].copy()
        self.coords_test = self.coords[val_end:].copy()
        
        # Handle any remaining NaNs
        self.X_train = self.X_train.fillna(0)
        self.X_val = self.X_val.fillna(0)
        self.X_test = self.X_test.fillna(0)
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"\nData splits:")
        print(f"Train: {self.X_train.shape}")
        print(f"Validation: {self.X_val.shape}")
        print(f"Test: {self.X_test.shape}")
        
    def train_ann(self):
        """Train Artificial Neural Network"""
        print("\n" + "="*50)
        print("TRAINING ARTIFICIAL NEURAL NETWORK (ANN)")
        print("="*50)
        
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
        
        # Predictions
        y_pred_val = ann.predict(self.X_val_scaled)
        y_pred_test = ann.predict(self.X_test_scaled)
        
        # Metrics
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
            'predictions_val': y_pred_val, 'predictions_test': y_pred_test
        }
        
    def train_xgboost(self):
        """Train XGBoost"""
        print("\n" + "="*50)
        print("TRAINING XGBOOST")
        print("="*50)
        
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
        
        # Predictions
        y_pred_val = xgb_model.predict(self.X_val)
        y_pred_test = xgb_model.predict(self.X_test)
        
        # Metrics
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
            'predictions_val': y_pred_val, 'predictions_test': y_pred_test
        }
        
    def train_cnn(self):
        """Train Convolutional Neural Network for spatial patterns"""
        print("\n" + "="*50)
        print("TRAINING CONVOLUTIONAL NEURAL NETWORK (CNN)")
        print("="*50)
        
        # Reshape data for CNN (add spatial dimension)
        # Group by unique coordinates to create spatial structure
        unique_coords = self.coords.drop_duplicates().sort_values(['latitude', 'longitude'])
        n_lats = len(unique_coords['latitude'].unique())
        n_lons = len(unique_coords['longitude'].unique())
        
        print(f"Spatial grid: {n_lats} x {n_lons}")
        
        # Create simple 1D CNN (treating features as spatial sequence)
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
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Reshape data for CNN
        X_train_cnn = self.X_train_scaled.reshape(-1, self.X_train_scaled.shape[1], 1)
        X_val_cnn = self.X_val_scaled.reshape(-1, self.X_val_scaled.shape[1], 1)
        X_test_cnn = self.X_test_scaled.reshape(-1, self.X_test_scaled.shape[1], 1)
        
        # Train
        history = model.fit(
            X_train_cnn, self.y_train,
            validation_data=(X_val_cnn, self.y_val),
            epochs=50,
            batch_size=256,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ],
            verbose=0
        )
        
        # Predictions
        y_pred_val = model.predict(X_val_cnn, verbose=0).flatten()
        y_pred_test = model.predict(X_test_cnn, verbose=0).flatten()
        
        # Metrics
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
            'predictions_val': y_pred_val, 'predictions_test': y_pred_test
        }
        
    def create_ensemble(self):
        """Create ensemble of two best models"""
        print("\n" + "="*50)
        print("CREATING ENSEMBLE MODEL")
        print("="*50)
        
        # Find two best models based on validation R²
        model_performance = [(name, res['val_r2']) for name, res in self.results.items()]
        model_performance.sort(key=lambda x: x[1], reverse=True)
        
        best_models = model_performance[:2]
        print(f"Best models: {best_models[0][0]} (R²={best_models[0][1]:.3f}), "
              f"{best_models[1][0]} (R²={best_models[1][1]:.3f})")
        
        # Create ensemble predictions (simple average)
        model1_name, model2_name = best_models[0][0], best_models[1][0]
        
        y_pred_val = (self.results[model1_name]['predictions_val'] + 
                      self.results[model2_name]['predictions_val']) / 2
        y_pred_test = (self.results[model1_name]['predictions_test'] + 
                       self.results[model2_name]['predictions_test']) / 2
        
        # Metrics
        val_rmse = np.sqrt(mean_squared_error(self.y_val, y_pred_val))
        val_r2 = r2_score(self.y_val, y_pred_val)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        test_r2 = r2_score(self.y_test, y_pred_test)
        
        print(f"Ensemble Validation - RMSE: {val_rmse:.3f}, R²: {val_r2:.3f}")
        print(f"Ensemble Test - RMSE: {test_rmse:.3f}, R²: {test_r2:.3f}")
        
        self.results['Ensemble'] = {
            'val_rmse': val_rmse, 'val_r2': val_r2,
            'test_rmse': test_rmse, 'test_r2': test_r2,
            'predictions_val': y_pred_val, 'predictions_test': y_pred_test,
            'models': [model1_name, model2_name]
        }
        
    def create_high_resolution_predictions(self, model_name='Ensemble'):
        """Create 1km resolution predictions"""
        print("\n" + "="*50)
        print("CREATING HIGH-RESOLUTION PREDICTIONS")
        print("="*50)
        
        # Get unique 25km coordinates
        unique_coords_25km = self.coords.drop_duplicates()
        
        # Create 1km grid (25x enhancement)
        lat_min, lat_max = unique_coords_25km['latitude'].min(), unique_coords_25km['latitude'].max()
        lon_min, lon_max = unique_coords_25km['longitude'].min(), unique_coords_25km['longitude'].max()
        
        # 0.01 degrees ≈ 1km
        lats_1km = np.arange(lat_min, lat_max + 0.01, 0.01)
        lons_1km = np.arange(lon_min, lon_max + 0.01, 0.01)
        
        print(f"25km grid: {len(unique_coords_25km)} points")
        print(f"1km grid: {len(lats_1km) * len(lons_1km)} points")
        
        # For demonstration, use a subset
        sample_lats = np.random.choice(lats_1km, size=min(100, len(lats_1km)), replace=False)
        sample_lons = np.random.choice(lons_1km, size=min(100, len(lons_1km)), replace=False)
        
        # Create sample high-res coordinates
        high_res_coords = []
        for lat in sample_lats:
            for lon in sample_lons:
                high_res_coords.append({'latitude': lat, 'longitude': lon})
        
        self.high_res_df = pd.DataFrame(high_res_coords)
        
        # Create features for high-res grid (using mean values for non-spatial features)
        high_res_features = pd.DataFrame(index=range(len(self.high_res_df)))
        
        # Add spatial features
        for col in self.X.columns:
            if any(x in col for x in ['lat', 'lon', 'dist', 'region', 'zone']):
                # Recalculate spatial features
                if col == 'lat_norm':
                    high_res_features[col] = (self.high_res_df['latitude'] - lat_min) / (lat_max - lat_min)
                elif col == 'lon_norm':
                    high_res_features[col] = (self.high_res_df['longitude'] - lon_min) / (lon_max - lon_min)
                else:
                    high_res_features[col] = self.X[col].mean()
            else:
                # Use mean for temporal features
                high_res_features[col] = self.X[col].mean()
        
        # Ensure no NaNs
        high_res_features = high_res_features.fillna(0)
        
        # Make predictions
        if model_name == 'Ensemble':
            # Use ensemble predictions
            model1_name = self.results['Ensemble']['models'][0]
            model2_name = self.results['Ensemble']['models'][1]
            
            if model1_name in ['ANN', 'CNN']:
                X_scaled = self.scaler.transform(high_res_features)
                if model1_name == 'CNN':
                    X_scaled = X_scaled.reshape(-1, X_scaled.shape[1], 1)
                    pred1 = self.models[model1_name].predict(X_scaled, verbose=0).flatten()
                else:
                    pred1 = self.models[model1_name].predict(X_scaled)
            else:
                pred1 = self.models[model1_name].predict(high_res_features)
                
            if model2_name in ['ANN', 'CNN']:
                X_scaled = self.scaler.transform(high_res_features)
                if model2_name == 'CNN':
                    X_scaled = X_scaled.reshape(-1, X_scaled.shape[1], 1)
                    pred2 = self.models[model2_name].predict(X_scaled, verbose=0).flatten()
                else:
                    pred2 = self.models[model2_name].predict(X_scaled)
            else:
                pred2 = self.models[model2_name].predict(high_res_features)
                
            self.high_res_df['fwi_predicted'] = (pred1 + pred2) / 2
        
        print(f"Created {len(self.high_res_df)} high-resolution predictions")
        print(f"Prediction statistics: mean={self.high_res_df['fwi_predicted'].mean():.2f}, "
              f"std={self.high_res_df['fwi_predicted'].std():.2f}")
        
    def back_aggregation_validation(self):
        """Validate by aggregating 1km predictions back to 25km"""
        print("\n" + "="*50)
        print("BACK-AGGREGATION VALIDATION")
        print("="*50)
        
        # Get unique 25km coordinates
        coords_25km = self.coords_test.drop_duplicates()
        
        # For each 25km cell, aggregate 1km predictions
        back_aggregated = []
        original_25km = []
        
        for idx, row in coords_25km.iterrows():
            lat_25km = row['latitude']
            lon_25km = row['longitude']
            
            # Find all 1km points within this 25km cell
            # 0.25 degrees ≈ 25km, so ±0.125 degrees
            mask = (
                (self.high_res_df['latitude'] >= lat_25km - 0.125) &
                (self.high_res_df['latitude'] < lat_25km + 0.125) &
                (self.high_res_df['longitude'] >= lon_25km - 0.125) &
                (self.high_res_df['longitude'] < lon_25km + 0.125)
            )
            
            if mask.sum() > 0:
                # Average 1km predictions within this cell
                cell_mean = self.high_res_df[mask]['fwi_predicted'].mean()
                back_aggregated.append(cell_mean)
                
                # Get original 25km FWI for this cell
                orig_mask = (
                    (self.coords_test['latitude'] == lat_25km) &
                    (self.coords_test['longitude'] == lon_25km)
                )
                if orig_mask.sum() > 0:
                    original_25km.append(self.y_test[orig_mask].mean())
        
        if len(back_aggregated) > 0 and len(original_25km) > 0:
            # Calculate correlation
            correlation = np.corrcoef(original_25km[:len(back_aggregated)], 
                                    back_aggregated[:len(original_25km)])[0, 1]
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(
                original_25km[:len(back_aggregated)], 
                back_aggregated[:len(original_25km)]
            ))
            
            print(f"Back-aggregation results:")
            print(f"  Correlation: {correlation:.3f}")
            print(f"  RMSE: {rmse:.3f}")
            print(f"  Number of cells compared: {len(back_aggregated)}")
            
            self.validation_results = {
                'back_aggregation': {
                    'correlation': correlation,
                    'rmse': rmse,
                    'n_cells': len(back_aggregated)
                }
            }
        else:
            print("  Insufficient data for back-aggregation validation")
            
    def spatial_correlation_analysis(self):
        """Analyze spatial correlation patterns"""
        print("\n" + "="*50)
        print("SPATIAL CORRELATION ANALYSIS")
        print("="*50)
        
        # Calculate spatial autocorrelation for original 25km data
        coords_25km = self.coords_test.drop_duplicates()
        
        # Simple distance-based correlation
        distances = []
        correlations = []
        
        # Sample points for efficiency
        sample_size = min(50, len(coords_25km))
        sample_indices = np.random.choice(coords_25km.index, size=sample_size, replace=False)
        
        for i, idx1 in enumerate(sample_indices):
            for idx2 in sample_indices[i+1:]:
                # Calculate distance
                lat1, lon1 = coords_25km.loc[idx1, ['latitude', 'longitude']]
                lat2, lon2 = coords_25km.loc[idx2, ['latitude', 'longitude']]
                
                dist = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111  # Convert to km
                
                # Get FWI values
                mask1 = (self.coords_test['latitude'] == lat1) & (self.coords_test['longitude'] == lon1)
                mask2 = (self.coords_test['latitude'] == lat2) & (self.coords_test['longitude'] == lon2)
                
                if mask1.sum() > 0 and mask2.sum() > 0:
                    fwi1 = self.y_test[mask1].values[0]
                    fwi2 = self.y_test[mask2].values[0]
                    
                    distances.append(dist)
                    correlations.append(abs(fwi1 - fwi2))
        
        if len(distances) > 0:
            # Bin distances and calculate average correlation
            bins = [0, 50, 100, 200, 500]
            binned_corr = []
            
            for i in range(len(bins)-1):
                mask = (np.array(distances) >= bins[i]) & (np.array(distances) < bins[i+1])
                if mask.sum() > 0:
                    avg_diff = np.mean(np.array(correlations)[mask])
                    binned_corr.append({
                        'distance_range': f"{bins[i]}-{bins[i+1]}km",
                        'avg_fwi_difference': avg_diff,
                        'n_pairs': mask.sum()
                    })
            
            print("Spatial correlation by distance:")
            for item in binned_corr:
                print(f"  {item['distance_range']}: "
                      f"avg FWI diff = {item['avg_fwi_difference']:.2f} "
                      f"(n={item['n_pairs']})")
                
            # Variance analysis
            var_25km = self.y_test.var()
            var_1km = self.high_res_df['fwi_predicted'].var()
            variance_ratio = var_1km / var_25km
            
            print(f"\nVariance analysis:")
            print(f"  25km variance: {var_25km:.2f}")
            print(f"  1km variance: {var_1km:.2f}")
            print(f"  Variance ratio (1km/25km): {variance_ratio:.3f}")
            
            if 'validation_results' not in self.__dict__:
                self.validation_results = {}
                
            self.validation_results['spatial_correlation'] = {
                'distance_bins': binned_corr,
                'variance_25km': var_25km,
                'variance_1km': var_1km,
                'variance_ratio': variance_ratio
            }
            
    def cross_scale_validation(self):
        """Leave-one-region-out cross-validation"""
        print("\n" + "="*50)
        print("CROSS-SCALE VALIDATION")
        print("="*50)
        
        # Define regions based on latitude bands
        lat_bands = [37, 38, 39, 40, 41, 42]
        
        cross_val_results = []
        
        for i in range(len(lat_bands)-1):
            # Define test region
            test_lat_min = lat_bands[i]
            test_lat_max = lat_bands[i+1]
            
            # Split data
            test_mask = (
                (self.coords['latitude'] >= test_lat_min) & 
                (self.coords['latitude'] < test_lat_max)
            )
            
            train_mask = ~test_mask
            
            if train_mask.sum() > 1000 and test_mask.sum() > 100:
                # Use XGBoost for efficiency
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                
                # Train on other regions
                model.fit(self.X[train_mask], self.y[train_mask])
                
                # Predict on test region
                y_pred = model.predict(self.X[test_mask])
                y_true = self.y[test_mask]
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)
                
                cross_val_results.append({
                    'region': f"{test_lat_min}°-{test_lat_max}°N",
                    'rmse': rmse,
                    'r2': r2,
                    'n_test': test_mask.sum()
                })
                
                print(f"  Region {test_lat_min}°-{test_lat_max}°N: "
                      f"RMSE={rmse:.2f}, R²={r2:.3f} (n={test_mask.sum()})")
        
        if cross_val_results:
            avg_rmse = np.mean([r['rmse'] for r in cross_val_results])
            avg_r2 = np.mean([r['r2'] for r in cross_val_results])
            
            print(f"\nAverage cross-validation performance:")
            print(f"  RMSE: {avg_rmse:.2f}")
            print(f"  R²: {avg_r2:.3f}")
            
            if 'validation_results' not in self.__dict__:
                self.validation_results = {}
                
            self.validation_results['cross_scale'] = {
                'regional_results': cross_val_results,
                'avg_rmse': avg_rmse,
                'avg_r2': avg_r2
            }
            
    def save_results(self):
        """Save all results"""
        print("\n" + "="*50)
        print("SAVING RESULTS")
        print("="*50)
        
        # Model comparison
        comparison_df = pd.DataFrame([
            {
                'Model': name,
                'Val_RMSE': res['val_rmse'],
                'Val_R2': res['val_r2'],
                'Test_RMSE': res['test_rmse'],
                'Test_R2': res['test_r2']
            }
            for name, res in self.results.items()
        ])
        comparison_df.to_csv('model_comparison_complete.csv', index=False)
        print("  Saved: model_comparison_complete.csv")
        
        # High-resolution predictions
        self.high_res_df.to_csv('high_resolution_predictions_complete.csv', index=False)
        print("  Saved: high_resolution_predictions_complete.csv")
        
        # Validation results
        import json
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            else:
                return obj
        
        validation_results_serializable = convert_to_serializable(self.validation_results)
        
        with open('validation_results_complete.json', 'w') as f:
            json.dump(validation_results_serializable, f, indent=2)
        print("  Saved: validation_results_complete.json")
        
        # Summary
        summary = {
            'best_model': max(self.results.items(), key=lambda x: x[1]['test_r2'])[0],
            'best_test_r2': max(self.results.items(), key=lambda x: x[1]['test_r2'])[1]['test_r2'],
            'ensemble_models': self.results['Ensemble']['models'] if 'Ensemble' in self.results else None,
            'ensemble_test_r2': self.results['Ensemble']['test_r2'] if 'Ensemble' in self.results else None,
            'back_aggregation_correlation': self.validation_results.get('back_aggregation', {}).get('correlation', None),
            'variance_ratio_1km_25km': self.validation_results.get('spatial_correlation', {}).get('variance_ratio', None),
            'cross_scale_avg_r2': self.validation_results.get('cross_scale', {}).get('avg_r2', None)
        }
        
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv('experiment_summary_complete.csv', index=False)
        print("  Saved: experiment_summary_complete.csv")
        
    def run_complete_experiment(self):
        """Run the complete experiment"""
        print("\n" + "="*80)
        print("COMPLETE FWI RESOLUTION ENHANCEMENT EXPERIMENT")
        print("="*80)
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Train models
        self.train_ann()
        self.train_xgboost()
        self.train_cnn()
        
        # Create ensemble
        self.create_ensemble()
        
        # Create high-resolution predictions
        self.create_high_resolution_predictions()
        
        # Run validation methods
        self.back_aggregation_validation()
        self.spatial_correlation_analysis()
        self.cross_scale_validation()
        
        # Save results
        self.save_results()
        
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*80)

if __name__ == "__main__":
    experiment = FWIResolutionEnhancement()
    experiment.run_complete_experiment()