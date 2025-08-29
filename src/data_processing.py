#!/usr/bin/env python3
"""
Unified data processing module for FWI super-resolution

This module contains all data processing functions migrated from the preprocessing pipeline.
Each function is designed to be reusable and configurable via parameters.
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import warnings
import zipfile
import tempfile
import os
from glob import glob
from typing import Dict, List, Tuple, Optional, Union
from scipy import ndimage
import rasterio
from rasterio.transform import from_bounds
from rasterio.enums import Resampling

warnings.filterwarnings('ignore')


def load_config(config_path: str = "configs/params.yaml") -> Dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_master_grid(config: Dict) -> xr.Dataset:
    """
    Create a 1km master grid for Portugal based on configuration parameters.
    
    Args:
        config: Configuration dictionary with grid parameters
        
    Returns:
        xr.Dataset: Master grid with latitude, longitude, and time coordinates
    """
    # Extract grid parameters
    lat_bounds = config['data']['grid_params']['lat_bounds']
    lon_bounds = config['data']['grid_params']['lon_bounds']
    resolution = config['data']['grid_params']['resolution_deg']
    
    # Create coordinate arrays
    latitudes = np.arange(lat_bounds[0], lat_bounds[1] + resolution, resolution)
    longitudes = np.arange(lon_bounds[0], lon_bounds[1] + resolution, resolution)
    
    # Create time range
    years = config['data']['years']
    months = config['data']['months']
    
    date_ranges = []
    for year in years:
        for month in months:
            start = pd.Timestamp(f'{year}-{month:02d}-01')
            if month == 11:
                end = pd.Timestamp(f'{year}-{month:02d}-30')
            else:
                end = pd.Timestamp(f'{year}-{month+1:02d}-01') - pd.Timedelta(days=1)
            dates = pd.date_range(start, end, freq='D')
            date_ranges.extend(dates)
    
    times = pd.DatetimeIndex(date_ranges)
    
    # Create the master grid dataset
    master_grid = xr.Dataset(
        coords={
            'latitude': latitudes,
            'longitude': longitudes, 
            'time': times
        }
    )
    
    print(f"✅ Created master grid: {len(latitudes)} × {len(longitudes)} × {len(times)}")
    print(f"   Resolution: {resolution}° (~{resolution*111:.1f} km)")
    print(f"   Spatial extent: {lat_bounds[0]}°N to {lat_bounds[1]}°N, {lon_bounds[0]}°E to {lon_bounds[1]}°E")
    print(f"   Temporal extent: {times[0].strftime('%Y-%m-%d')} to {times[-1].strftime('%Y-%m-%d')}")
    
    return master_grid


def process_landcover(master_grid: xr.Dataset, landcover_path: str) -> xr.Dataset:
    """
    Process ESA WorldCover land cover data and create a land mask.
    
    Args:
        master_grid: Master grid dataset
        landcover_path: Path to land cover data
        
    Returns:
        xr.Dataset: Land cover mask aligned to master grid
    """
    # For now, create a simple land mask based on coordinates
    # In full implementation, this would process ESA WorldCover tiles
    
    # Create meshgrid for lat/lon
    lon_grid, lat_grid = np.meshgrid(master_grid.longitude.values, master_grid.latitude.values)
    
    # Simple approximation: exclude far western ocean areas
    land_mask_array = np.where(
        (lon_grid > -9.5) | (lat_grid > 40.0),
        1.0, 0.0
    )
    
    land_mask_ds = xr.Dataset({
        'land_mask': (('latitude', 'longitude'), land_mask_array)
    }, coords={'latitude': master_grid.latitude, 'longitude': master_grid.longitude})
    
    land_pixels = int(land_mask_array.sum())
    total_pixels = land_mask_array.size
    print(f"✅ Created land mask: {land_pixels:,} land pixels ({100*land_pixels/total_pixels:.1f}%)")
    
    return land_mask_ds


def load_era5_fwi(fwi_path: Union[str, Path], master_grid: xr.Dataset, config: Dict) -> xr.Dataset:
    """
    Load and process ERA5 FWI data, regridding to 1km resolution.
    
    Args:
        fwi_path: Path to ERA5 FWI data file
        master_grid: Master grid to regrid to
        config: Configuration dictionary
        
    Returns:
        xr.Dataset: Regridded FWI data at 1km resolution
    """
    fwi_path = Path(fwi_path)
    
    if not fwi_path.exists():
        raise FileNotFoundError(f"FWI data not found: {fwi_path}")
    
    print(f"📂 Loading ERA5 FWI data: {fwi_path}")
    era5_ds = xr.open_dataset(fwi_path)
    
    # Handle coordinate naming and conversion
    if 'valid_time' in era5_ds.dims:
        era5_ds = era5_ds.rename({'valid_time': 'time'})
    if 'fwinx' in era5_ds.data_vars:
        era5_ds = era5_ds.rename({'fwinx': 'fwi'})
    
    # Convert longitude from 0-360 to -180-180 if needed
    if era5_ds.longitude.min() >= 0:
        era5_ds = era5_ds.assign_coords(longitude=(era5_ds.longitude + 180) % 360 - 180)
        era5_ds = era5_ds.sortby('longitude')
    
    # Extract region bounds
    lat_bounds = config['data']['grid_params']['lat_bounds']
    lon_bounds = config['data']['grid_params']['lon_bounds']
    
    portugal_fwi = era5_ds.sel(
        latitude=slice(lat_bounds[1], lat_bounds[0]),
        longitude=slice(lon_bounds[0], lon_bounds[1])
    )
    
    # Filter for configured time period
    test_dates = pd.DatetimeIndex(master_grid.time.values)
    portugal_fwi = portugal_fwi.sel(time=test_dates, method='nearest')
    
    # Regrid to 1km using spatial interpolation
    fwi_1km = portugal_fwi.interp(
        latitude=master_grid.latitude,
        longitude=master_grid.longitude,
        method='linear'
    )
    
    # Fill NaN values using nearest neighbor extrapolation
    fwi_1km_filled = portugal_fwi.interp(
        latitude=master_grid.latitude,
        longitude=master_grid.longitude,
        method='nearest',
        kwargs={'fill_value': 'extrapolate'}
    )
    fwi_1km['fwi'] = fwi_1km['fwi'].fillna(fwi_1km_filled['fwi'])
    
    print(f"✅ Regridded FWI to 1km: {fwi_1km.fwi.shape}")
    print(f"   Original resolution: {portugal_fwi.latitude.diff('latitude').mean().values:.3f}°")
    print(f"   Target resolution: {master_grid.latitude.diff('latitude').mean().values:.3f}°")
    
    return fwi_1km


def load_era5_atmospheric(data_path: Union[str, Path], master_grid: xr.Dataset) -> xr.Dataset:
    """
    Load and process ERA5 atmospheric parameters.
    
    Args:
        data_path: Path to ERA5 atmospheric data directory
        master_grid: Master grid to regrid to
        
    Returns:
        xr.Dataset: Atmospheric variables regridded to 1km
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        print(f"⚠️ Atmospheric data path not found: {data_path}")
        return xr.Dataset(coords=master_grid.coords)
    
    # Initialize combined dataset
    combined_ds = None
    
    # Load daily mean files
    mean_files = sorted(data_path.glob("era5_daily_mean_*.nc"))
    temp_files = sorted(data_path.glob("era5_daily_max_temp_*.nc"))
    precip_files = sorted(data_path.glob("era5_daily_total_precipitation_*.nc"))
    
    all_files = mean_files + temp_files + precip_files
    print(f"📂 Found {len(all_files)} atmospheric data files")
    
    if mean_files:
        # Load and combine all mean files
        print(f"  Loading {len(mean_files)} daily mean files...")
        datasets = []
        for file in mean_files:  # Load ALL files
            try:
                ds = xr.open_dataset(file)
                datasets.append(ds)
            except:
                continue
        
        if datasets:
            combined_ds = xr.concat(datasets, dim='time')
            
            # Regrid to 1km
            print("  Regridding atmospheric data to 1km...")
            combined_1km = combined_ds.interp(
                latitude=master_grid.latitude,
                longitude=master_grid.longitude,
                method='linear'
            )
            
            # Select time period matching master grid
            combined_1km = combined_1km.sel(time=master_grid.time, method='nearest')
            
            return combined_1km
    
    # Return empty dataset if no data
    return xr.Dataset(coords=master_grid.coords)


def load_era5_land(data_path: Union[str, Path], master_grid: xr.Dataset) -> xr.Dataset:
    """
    Load and process ERA5-Land surface parameters.
    
    Args:
        data_path: Path to ERA5-Land data
        master_grid: Master grid to regrid to
        
    Returns:
        xr.Dataset: Land surface variables regridded to 1km
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        print(f"⚠️ ERA5-Land data path not found: {data_path}")
        return xr.Dataset(coords=master_grid.coords)
    
    # Load ERA5-Land files
    land_files = sorted(data_path.glob("era5_land_may_nov_*.nc"))
    print(f"📂 Found {len(land_files)} ERA5-Land files")
    
    if land_files:
        # Load and combine all land files
        print(f"  Loading ERA5-Land data...")
        datasets = []
        for file in land_files:  # Load ALL files
            try:
                ds = xr.open_dataset(file)
                datasets.append(ds)
            except:
                continue
        
        if datasets:
            combined_ds = xr.concat(datasets, dim='time')
            
            # ERA5-Land is at 0.1° (~10km), regrid to 1km
            print("  Regridding ERA5-Land from 10km to 1km...")
            land_1km = combined_ds.interp(
                latitude=master_grid.latitude,
                longitude=master_grid.longitude,
                method='linear'
            )
            
            # Select time period matching master grid
            land_1km = land_1km.sel(time=master_grid.time, method='nearest')
            
            print(f"✅ Loaded land surface variables: {list(land_1km.data_vars)}")
            return land_1km
    
    return xr.Dataset(coords=master_grid.coords)


def load_uerra(data_path: Union[str, Path], master_grid: xr.Dataset) -> xr.Dataset:
    """
    Load and process UERRA high-resolution reanalysis data.
    
    Args:
        data_path: Path to UERRA data directory
        master_grid: Master grid to regrid to
        
    Returns:
        xr.Dataset: UERRA variables regridded to 1km
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        print(f"⚠️ UERRA data path not found: {data_path}")
        return xr.Dataset(coords=master_grid.coords)
    
    # Load UERRA files
    uerra_files = sorted(data_path.glob("uerra_mescan_may_nov_*.nc"))
    print(f"📂 Found {len(uerra_files)} UERRA files")
    
    if uerra_files:
        # Load all UERRA files
        print(f"  Loading UERRA high-resolution data...")
        datasets = []
        for file in uerra_files:  # Load ALL files
            try:
                ds = xr.open_dataset(file)
                datasets.append(ds)
                print(f"    Loaded {file.name}: {list(ds.data_vars)}")
            except Exception as e:
                print(f"    Error loading {file.name}: {e}")
                continue
        
        if datasets:
            combined_ds = xr.concat(datasets, dim='time')
            
            # UERRA is at ~5.5km resolution, regrid to 1km
            print("  Regridding UERRA from 5.5km to 1km...")
            uerra_1km = combined_ds.interp(
                latitude=master_grid.latitude,
                longitude=master_grid.longitude,
                method='linear'
            )
            
            # Select time period matching master grid
            uerra_1km = uerra_1km.sel(time=master_grid.time, method='nearest')
            
            print(f"✅ Loaded UERRA variables: {list(uerra_1km.data_vars)}")
            return uerra_1km
    
    return xr.Dataset(coords=master_grid.coords)


def unify_datasets(
    master_grid: xr.Dataset,
    fwi_data: xr.Dataset,
    landcover_data: xr.Dataset,
    atmospheric_data: Optional[xr.Dataset] = None,
    land_data: Optional[xr.Dataset] = None,
    uerra_data: Optional[xr.Dataset] = None
) -> xr.Dataset:
    """
    Unify all processed datasets into a single coherent dataset.
    
    Args:
        master_grid: Master grid with coordinates
        fwi_data: FWI data at 1km
        landcover_data: Land cover mask
        atmospheric_data: Atmospheric variables (optional)
        land_data: Land surface variables (optional)
        uerra_data: UERRA high-res data (optional)
        
    Returns:
        xr.Dataset: Unified dataset ready for ML
    """
    print("\n🔄 Unifying all datasets...")
    
    # Start with master grid
    unified = master_grid.copy()
    
    # Add FWI data
    if 'fwi' in fwi_data:
        unified['fwi'] = fwi_data['fwi']
        print("  ✓ Added FWI data")
    
    # Add land mask
    if 'land_mask' in landcover_data:
        unified['land_mask'] = landcover_data['land_mask']
        print("  ✓ Added land mask")
    
    # Add atmospheric data if available
    if atmospheric_data is not None and len(atmospheric_data.data_vars) > 0:
        for var in atmospheric_data.data_vars:
            unified[var] = atmospheric_data[var]
        print(f"  ✓ Added {len(atmospheric_data.data_vars)} atmospheric variables")
    
    # Add land surface data if available
    if land_data is not None and len(land_data.data_vars) > 0:
        for var in land_data.data_vars:
            unified[var] = land_data[var]
        print(f"  ✓ Added {len(land_data.data_vars)} land surface variables")
    
    # Add UERRA data if available
    if uerra_data is not None and len(uerra_data.data_vars) > 0:
        for var in uerra_data.data_vars:
            # Prefix UERRA variables to avoid conflicts
            unified[f'uerra_{var}'] = uerra_data[var]
        print(f"  ✓ Added {len(uerra_data.data_vars)} UERRA variables")
    
    # Apply land mask to all variables
    if 'land_mask' in unified:
        for var in unified.data_vars:
            if var != 'land_mask' and 'time' in unified[var].dims:
                # Mask ocean pixels
                unified[var] = unified[var].where(unified['land_mask'] > 0)
    
    # Add metadata
    unified.attrs.update({
        'title': 'Unified FWI Dataset for Portugal',
        'description': 'FWI super-resolution training dataset at 1km resolution',
        'creation_date': pd.Timestamp.now().isoformat(),
        'resolution': '0.01 degrees (~1km)',
        'region': 'Portugal',
        'temporal_coverage': f"{unified.time.values[0]} to {unified.time.values[-1]}",
        'n_variables': len(unified.data_vars),
        'n_timepoints': len(unified.time)
    })
    
    # Print summary
    print(f"\n✅ Unified dataset created:")
    print(f"   Dimensions: {dict(unified.dims)}")
    print(f"   Variables: {list(unified.data_vars)}")
    print(f"   Size: {unified.nbytes / 1e9:.2f} GB")
    
    return unified


def prepare_ml_data(
    dataset: xr.Dataset, 
    config: Dict
) -> Tuple[np.ndarray, np.ndarray, Dict[str, slice]]:
    """
    Prepare data for machine learning from unified dataset.
    
    Args:
        dataset: Unified xarray dataset
        config: Configuration dictionary
        
    Returns:
        Tuple of (X, y, split_indices)
    """
    # Extract FWI data as target
    if 'fwi' not in dataset.data_vars:
        raise ValueError("FWI variable not found in dataset")
    
    fwi_data = dataset['fwi'].values  # Shape: (time, lat, lon)
    
    # For super-resolution, create low-res input by downsampling
    # Here we'll use the original resolution as high-res target
    y = fwi_data  # High-resolution target
    
    # Create low-resolution input by downsampling
    scale_factor = config['data'].get('upscale_factor', 4)
    X = ndimage.zoom(fwi_data, (1, 1/scale_factor, 1/scale_factor), order=1)
    
    # Upsample back to original size for training
    X = ndimage.zoom(X, (1, scale_factor, scale_factor), order=1)
    
    # Apply land mask if available
    if 'land_mask' in dataset.data_vars:
        mask = dataset['land_mask'].values.astype(bool)
        # Flatten spatial dimensions for land pixels only
        n_time = y.shape[0]
        # Reshape to (time, lat*lon) first, then apply mask
        X_flat = X.reshape(n_time, -1)
        y_flat = y.reshape(n_time, -1)
        mask_flat = mask.flatten()
        
        # Apply mask only if dimensions match
        if X_flat.shape[1] == len(mask_flat):
            X_masked = X_flat[:, mask_flat]  # Shape: (time, n_land_pixels)
            y_masked = y_flat[:, mask_flat]
            X = X_masked
            y = y_masked
        else:
            print(f"  Warning: Mask dimension mismatch. Skipping masking.")
    
    # Handle NaN values
    nan_count_X = np.isnan(X).sum()
    nan_count_y = np.isnan(y).sum()
    
    if nan_count_X > 0:
        fill_value_X = np.nanmean(X)
        X = np.nan_to_num(X, nan=fill_value_X)
        print(f"  Filled {nan_count_X:,} NaN values in X with {fill_value_X:.3f}")
    
    if nan_count_y > 0:
        fill_value_y = np.nanmean(y)
        y = np.nan_to_num(y, nan=fill_value_y)
        print(f"  Filled {nan_count_y:,} NaN values in y with {fill_value_y:.3f}")
    
    # Create train/val/test splits
    n_samples = len(X)
    train_ratio = config['training']['train_ratio']
    val_ratio = config['training']['val_ratio']
    
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    split_indices = {
        'train': slice(0, train_end),
        'val': slice(train_end, val_end),
        'test': slice(val_end, None)
    }
    
    print(f"✅ Prepared ML data:")
    print(f"   X shape: {X.shape}, y shape: {y.shape}")
    print(f"   Train: {train_end} samples")
    print(f"   Val: {val_end - train_end} samples")
    print(f"   Test: {n_samples - val_end} samples")
    
    return X, y, split_indices


def save_dataset(dataset: xr.Dataset, output_path: Union[str, Path], compress: bool = True):
    """
    Save xarray dataset with optional compression.
    
    Args:
        dataset: Dataset to save
        output_path: Output file path
        compress: Whether to apply compression
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if compress:
        encoding = {}
        for var in dataset.data_vars:
            encoding[var] = {
                'zlib': True,
                'complevel': 4,
                'dtype': 'float32'
            }
        dataset.to_netcdf(output_path, encoding=encoding)
    else:
        dataset.to_netcdf(output_path)
    
    size_mb = output_path.stat().st_size / 1e6
    print(f"✅ Saved dataset to {output_path} ({size_mb:.1f} MB)")


def validate_fire_detection(dataset: xr.Dataset, config: Dict) -> bool:
    """
    Validate fire event detection (Pedrógão Grande fire).
    
    Args:
        dataset: Dataset with FWI data
        config: Configuration with fire event details
        
    Returns:
        bool: Whether fire was detected
    """
    fire_config = config['evaluation']['pedrogao_fire']
    
    if 'fwi' not in dataset.data_vars:
        return False
    
    try:
        # Get FWI value at fire location and date
        fire_fwi = dataset['fwi'].sel(
            time=fire_config['date'],
            latitude=fire_config['coordinates']['lat'],
            longitude=fire_config['coordinates']['lon'],
            method='nearest'
        )
        
        fwi_value = float(fire_fwi)
        threshold = fire_config['min_fwi_threshold']
        
        detected = fwi_value >= threshold
        print(f"🔥 Fire validation: FWI={fwi_value:.1f}, threshold={threshold}, detected={detected}")
        
        return detected
        
    except Exception as e:
        print(f"⚠️ Fire validation failed: {e}")
        return False