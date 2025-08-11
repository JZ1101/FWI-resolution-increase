#!/usr/bin/env python3
"""
Final FWI Downscaling Pipeline: 25km → 10km
Using mathematically calculated 10km FWI as pseudo-ground truth
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class FinalFWIPipeline:
    """Production-ready FWI downscaling pipeline"""
    
    def __init__(self):
        """Initialize pipeline"""
        self.model = None
        self.results = {}
        
    def load_data(self):
        """Load all datasets with coordinate fixing"""
        print("=== Loading Data ===")
        
        # Load 25km FWI
        print("Loading 25km ERA5 FWI...")
        self.fwi_25km = xr.open_dataset('data/era5_fwi_2017.nc')
        
        # Fix longitude coordinates (360° to ±180° format)
        self.fwi_25km = self.fwi_25km.assign_coords(longitude=(self.fwi_25km.longitude - 360))
        print(f"  25km FWI: {self.fwi_25km.fwinx.shape}")
        
        # Load 10km calculated FWI targets
        print("Loading 10km calculated FWI targets...")
        self.fwi_10km_target = xr.open_dataset('data/fwi_10km_full_year.nc')
        print(f"  10km FWI targets: {self.fwi_10km_target.fwi_10km.shape}")
        
        # Load 10km ERA5-Land features
        print("Loading 10km ERA5-Land features...")
        self.era5_land = xr.open_dataset('data/data_0.nc')
        print(f"  10km features: {self.era5_land.t2m.shape}")
        
        print(f"\\nCoordinate ranges:")
        print(f"  25km: lat {self.fwi_25km.latitude.min().values:.1f}-{self.fwi_25km.latitude.max().values:.1f}, lon {self.fwi_25km.longitude.min().values:.1f}-{self.fwi_25km.longitude.max().values:.1f}")
        print(f"  10km: lat {self.era5_land.latitude.min().values:.1f}-{self.era5_land.latitude.max().values:.1f}, lon {self.era5_land.longitude.min().values:.1f}-{self.era5_land.longitude.max().values:.1f}")
        
        return True
    
    def create_training_samples(self, max_days=60, spatial_stride=3):
        """
        Create training samples efficiently
        
        Parameters:
        -----------
        max_days : int
            Maximum days to process
        spatial_stride : int  
            Sample every Nth pixel spatially
        """
        print(f"\\n=== Creating Training Samples ===")
        print(f"Processing {max_days} days, spatial stride {spatial_stride}")
        
        features_list = []
        targets_list = []
        
        # Process subset of days
        n_days = min(max_days, len(self.fwi_10km_target.time))
        
        for day in range(n_days):
            if day % 10 == 0:
                print(f"  Processing day {day+1}/{n_days}")
            
            # Get data for this day
            fwi_25_day = self.fwi_25km.isel(valid_time=day)['fwinx']
            fwi_10_day = self.fwi_10km_target.isel(time=day)['fwi_10km']
            land_day = self.era5_land.isel(valid_time=day)
            
            # Sample pixels with stride
            for i in range(0, len(self.era5_land.latitude), spatial_stride):
                for j in range(0, len(self.era5_land.longitude), spatial_stride):
                    
                    lat = float(self.era5_land.latitude[i])
                    lon = float(self.era5_land.longitude[j])
                    
                    # Get 10km target value
                    try:
                        target_val = float(fwi_10_day.isel(latitude=i, longitude=j).values)
                        if np.isnan(target_val) or target_val < 0:
                            continue
                    except:
                        continue
                    
                    # Get 25km FWI interpolated to this location
                    try:
                        fwi_25_interp = fwi_25_day.interp(latitude=lat, longitude=lon, method='linear')
                        fwi_25_val = float(fwi_25_interp.values)
                        if np.isnan(fwi_25_val):
                            continue
                    except:
                        continue
                    
                    # Get 10km meteorological features
                    try:
                        features = {
                            'fwi_25km': fwi_25_val,
                            'temp_10km': float(land_day['t2m'].isel(latitude=i, longitude=j).values) - 273.15,  # K to C
                            'dewpoint_10km': float(land_day['d2m'].isel(latitude=i, longitude=j).values) - 273.15,  # K to C
                            'u_wind_10km': float(land_day['u10'].isel(latitude=i, longitude=j).values),
                            'v_wind_10km': float(land_day['v10'].isel(latitude=i, longitude=j).values),
                            'precip_10km': float(land_day['tp'].isel(latitude=i, longitude=j).values) * 1000,  # m to mm
                            'latitude': lat,
                            'longitude': lon,
                            'day_of_year': day + 1
                        }
                        
                        # Calculate derived features
                        features['wind_speed_10km'] = np.sqrt(features['u_wind_10km']**2 + features['v_wind_10km']**2)
                        features['rh_10km'] = min(100, max(0, 100 * np.exp(17.625 * features['dewpoint_10km'] / (243.04 + features['dewpoint_10km'])) / np.exp(17.625 * features['temp_10km'] / (243.04 + features['temp_10km']))))
                        
                        # Skip if any critical features are invalid
                        if any(np.isnan(val) or np.isinf(val) for val in features.values()):
                            continue
                        
                        features_list.append(features)
                        targets_list.append(target_val)
                        
                    except Exception as e:
                        continue
        
        print(f"\\nCreated {len(features_list)} training samples")
        
        # Convert to DataFrame
        self.features_df = pd.DataFrame(features_list)
        self.targets = np.array(targets_list)
        
        print(f"Features: {list(self.features_df.columns)}")
        print(f"Target stats: mean={self.targets.mean():.2f}, std={self.targets.std():.2f}, range=[{self.targets.min():.2f}, {self.targets.max():.2f}]")
        
        return len(features_list)
    
    def train_and_validate(self, test_size=0.2):
        """Train model with comprehensive validation"""
        print(f"\\n=== Training and Validation ===")
        
        # Split data
        X = self.features_df
        y = self.targets
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Train Random Forest model
        print("Training Random Forest model...")
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        print(f"\\nModel Performance:")
        print(f"  Training RMSE: {train_rmse:.3f}")
        print(f"  Test RMSE: {test_rmse:.3f}")
        print(f"  Test MAE: {test_mae:.3f}")
        print(f"  Training R²: {train_r2:.3f}")
        print(f"  Test R²: {test_r2:.3f}")
        
        # Cross-validation
        print("\\nCross-validation (5-fold)...")
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        cv_rmse = np.sqrt(-cv_scores)
        print(f"  CV RMSE: {cv_rmse.mean():.3f} ± {cv_rmse.std():.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\\nTop 8 Most Important Features:")
        for _, row in feature_importance.head(8).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        # Store results
        self.results = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std(),
            'feature_importance': feature_importance,
            'y_test': y_test,
            'y_test_pred': y_test_pred,
            'y_train': y_train,
            'y_train_pred': y_train_pred
        }
        
        return self.model
    
    def create_validation_plots(self):
        """Create comprehensive validation plots"""
        print("\\n=== Creating Validation Plots ===")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('FWI Downscaling Model Validation (25km → 10km)', fontsize=14)
        
        # 1. Test predictions vs actual
        ax1 = axes[0, 0]
        ax1.scatter(self.results['y_test'], self.results['y_test_pred'], alpha=0.6, s=20)
        max_val = max(self.results['y_test'].max(), self.results['y_test_pred'].max())
        ax1.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
        ax1.set_xlabel('Actual 10km FWI (Formula-based)')
        ax1.set_ylabel('Predicted 10km FWI (ML)')
        ax1.set_title(f'Test Set Predictions\\n(R² = {self.results["test_r2"]:.3f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Training predictions vs actual
        ax2 = axes[0, 1]
        ax2.scatter(self.results['y_train'], self.results['y_train_pred'], alpha=0.3, s=10)
        max_val = max(self.results['y_train'].max(), self.results['y_train_pred'].max())
        ax2.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
        ax2.set_xlabel('Actual 10km FWI (Formula-based)')
        ax2.set_ylabel('Predicted 10km FWI (ML)')
        ax2.set_title(f'Training Set Predictions\\n(R² = {self.results["train_r2"]:.3f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Residuals histogram
        ax3 = axes[0, 2]
        residuals = self.results['y_test_pred'] - self.results['y_test']
        ax3.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        ax3.axvline(0, color='red', linestyle='--', label='Zero error')
        ax3.set_xlabel('Prediction Error (ML - Formula)')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'Residuals Distribution\\n(RMSE = {self.results["test_rmse"]:.3f})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Feature importance
        ax4 = axes[1, 0]
        top_features = self.results['feature_importance'].head(8)
        ax4.barh(range(len(top_features)), top_features['importance'])
        ax4.set_yticks(range(len(top_features)))
        ax4.set_yticklabels(top_features['feature'])
        ax4.set_xlabel('Feature Importance')
        ax4.set_title('Top 8 Feature Importance')
        ax4.grid(True, alpha=0.3)
        
        # 5. Error vs actual value
        ax5 = axes[1, 1]
        ax5.scatter(self.results['y_test'], residuals, alpha=0.6, s=20)
        ax5.axhline(0, color='red', linestyle='--', label='Zero error')
        ax5.set_xlabel('Actual 10km FWI')
        ax5.set_ylabel('Prediction Error')
        ax5.set_title('Error vs Actual Value')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Error distribution by FWI range
        ax6 = axes[1, 2]
        fwi_bins = np.linspace(self.results['y_test'].min(), self.results['y_test'].max(), 6)
        bin_centers = (fwi_bins[:-1] + fwi_bins[1:]) / 2
        bin_rmse = []
        
        for i in range(len(fwi_bins)-1):
            mask = (self.results['y_test'] >= fwi_bins[i]) & (self.results['y_test'] < fwi_bins[i+1])
            if mask.sum() > 0:
                bin_rmse.append(np.sqrt(mean_squared_error(self.results['y_test'][mask], self.results['y_test_pred'][mask])))
            else:
                bin_rmse.append(0)
        
        ax6.bar(bin_centers, bin_rmse, width=np.diff(fwi_bins)[0]*0.8)
        ax6.set_xlabel('FWI Range')
        ax6.set_ylabel('RMSE')
        ax6.set_title('RMSE by FWI Range')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/fwi_validation_comprehensive.png', dpi=150, bbox_inches='tight')
        print("Saved comprehensive validation plots to outputs/fwi_validation_comprehensive.png")
        plt.show()
    
    def run_complete_pipeline(self):
        """Run the complete pipeline"""
        print("\\n" + "="*70)
        print("FINAL FWI DOWNSCALING PIPELINE: 25km ERA5 → 10km ENHANCED FWI")
        print("="*70)
        
        # Load data
        self.load_data()
        
        # Create training samples
        n_samples = self.create_training_samples(max_days=60, spatial_stride=3)
        
        if n_samples < 100:
            print(f"ERROR: Only {n_samples} samples created. Need at least 100.")
            return None
        
        # Train and validate
        self.train_and_validate()
        
        # Create validation plots
        self.create_validation_plots()
        
        print("\\n" + "="*70)
        print("PIPELINE COMPLETE - RESULTS SUMMARY")
        print("="*70)
        print(f"Training samples: {len(self.targets):,}")
        print(f"Test RMSE: {self.results['test_rmse']:.3f}")
        print(f"Test R²: {self.results['test_r2']:.3f}")
        print(f"CV RMSE: {self.results['cv_rmse_mean']:.3f} ± {self.results['cv_rmse_std']:.3f}")
        print("\\n✅ FWI downscaling model successfully trained!")
        print("   Can now predict 10km FWI from 25km ERA5 + 10km meteorological features")
        
        return self.model

def main():
    """Main execution"""
    import os
    os.makedirs('outputs', exist_ok=True)
    
    pipeline = FinalFWIPipeline()
    model = pipeline.run_complete_pipeline()
    
    return pipeline, model

if __name__ == "__main__":
    pipeline, model = main()