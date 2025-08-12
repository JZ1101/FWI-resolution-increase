"""Data loading and preprocessing for FWI data"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class FWIDataLoader:
    """Load and preprocess FWI data from NetCDF files"""
    
    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        self.portugal_bounds = {
            'lat_min': 36.0, 'lat_max': 43.0,
            'lon_min': -10.0, 'lon_max': -6.0
        }
        
    def load_era5_fwi(self, year: int = 2017) -> xr.Dataset:
        """Load ERA5 FWI data (25km resolution)"""
        # Try NetCDF first
        fwi_path = self.data_dir / f"era5_fwi/fwi_{year}.nc"
        
        # Try CSV if NetCDF doesn't exist
        if not fwi_path.exists():
            csv_path = self.data_dir / f"2017_data/era5_fwi_{year}_portugal.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                # Convert CSV to xarray Dataset
                ds = self._csv_to_xarray(df, 'fwinx')
                logger.info(f"Loaded ERA5 FWI from CSV: {df.shape}")
                return ds
            else:
                raise FileNotFoundError(f"FWI data not found: {fwi_path} or {csv_path}")
            
        ds = xr.open_dataset(fwi_path)
        logger.info(f"Loaded ERA5 FWI: {ds.dims}")
        return ds
    
    def _csv_to_xarray(self, df: pd.DataFrame, value_col: str = 'fwinx') -> xr.Dataset:
        """Convert CSV dataframe to xarray Dataset"""
        # Assuming CSV has columns: time, latitude, longitude, value
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        
        # Get unique coordinates
        times = df['time'].unique() if 'time' in df.columns else [0]
        lats = sorted(df['latitude'].unique())
        lons = sorted(df['longitude'].unique())
        
        # Create 3D array
        data = np.zeros((len(times), len(lats), len(lons)))
        
        for i, t in enumerate(times):
            if 'time' in df.columns:
                time_data = df[df['time'] == t]
            else:
                time_data = df
            
            for _, row in time_data.iterrows():
                lat_idx = lats.index(row['latitude'])
                lon_idx = lons.index(row['longitude'])
                if value_col in row:
                    data[i, lat_idx, lon_idx] = row[value_col]
                elif 'fwi' in row:
                    data[i, lat_idx, lon_idx] = row['fwi']
        
        # Create xarray Dataset
        ds = xr.Dataset(
            {value_col: (['time', 'latitude', 'longitude'], data)},
            coords={
                'time': times,
                'latitude': lats,
                'longitude': lons
            }
        )
        
        return ds
    
    def load_era5_atmospheric(self, year: int = 2017) -> xr.Dataset:
        """Load ERA5 atmospheric variables"""
        temp_path = self.data_dir / f"era5_atmospheric/temp_{year}.nc"
        wind_path = self.data_dir / f"era5_atmospheric/wind_precip_{year}.nc"
        
        datasets = []
        if temp_path.exists():
            datasets.append(xr.open_dataset(temp_path))
        if wind_path.exists():
            datasets.append(xr.open_dataset(wind_path))
            
        if datasets:
            return xr.merge(datasets)
        else:
            logger.warning("No atmospheric data found")
            return None
    
    def load_era5_land(self, year: int = 2017) -> xr.Dataset:
        """Load ERA5-Land data (9km resolution)"""
        land_path = self.data_dir / f"era5_land/era5land_{year}.nc"
        
        if land_path.exists():
            ds = xr.open_dataset(land_path)
            logger.info(f"Loaded ERA5-Land: {ds.dims}")
            return ds
        return None
    
    def prepare_training_data(
        self, 
        low_res: xr.Dataset,
        high_res: Optional[xr.Dataset] = None,
        features: Optional[xr.Dataset] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Prepare data for ML training
        
        Args:
            low_res: Low resolution FWI data (input)
            high_res: High resolution target (for supervised learning)
            features: Additional features (terrain, atmospheric)
            
        Returns:
            X: Input features
            y: Target values
            aux_features: Additional features
        """
        
        X = low_res['fwinx'].values if 'fwinx' in low_res else low_res.to_array().values
        
        X = X.reshape(X.shape[0], -1) if X.ndim > 2 else X
        
        y = None
        if high_res is not None:
            y = high_res['fwinx'].values if 'fwinx' in high_res else high_res.to_array().values
            y = y.reshape(y.shape[0], -1) if y.ndim > 2 else y
            
        aux_features = None
        if features is not None:
            aux_features = features.to_array().values
            aux_features = aux_features.reshape(aux_features.shape[0], -1)
            
        return X, y, aux_features
    
    def create_patches(
        self, 
        data: np.ndarray, 
        patch_size: int = 8,
        stride: int = 4
    ) -> np.ndarray:
        """Create overlapping patches for training"""
        
        if data.ndim == 3:
            time, height, width = data.shape
            patches = []
            
            for t in range(time):
                for i in range(0, height - patch_size + 1, stride):
                    for j in range(0, width - patch_size + 1, stride):
                        patch = data[t, i:i+patch_size, j:j+patch_size]
                        patches.append(patch)
                        
            return np.array(patches)
        else:
            raise ValueError(f"Expected 3D array, got {data.ndim}D")
    
    def split_data(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_state: int = 42
    ) -> Dict[str, np.ndarray]:
        """Split data into train/val/test sets"""
        
        np.random.seed(random_state)
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        splits = {
            'X_train': X[train_idx],
            'X_val': X[val_idx],
            'X_test': X[test_idx]
        }
        
        if y is not None:
            splits.update({
                'y_train': y[train_idx],
                'y_val': y[val_idx],
                'y_test': y[test_idx]
            })
            
        return splits