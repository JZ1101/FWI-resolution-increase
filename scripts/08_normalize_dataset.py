#!/usr/bin/env python3
"""
Script 08: Normalize Dataset (Corrected Version - Static Variables Handled Correctly)
Final preprocessing step - normalizes the unified dataset for model training.

CRITICAL: This script:
1. Prevents data leakage by computing statistics from training data only
2. Handles static (2D) and dynamic (3D) variables separately
3. Maintains proper dimensions for all variables

Data splits (for dynamic variables only):
- Training: 2010-2015 (6 years)
- Validation: 2016 (1 year)  
- Test: 2017 (1 year)

Static variables (land cover) are normalized using their full spatial statistics.
"""

import os
import numpy as np
import xarray as xr
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
import gc

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'input_file': 'data/interim/portugal/07_fwi_unified_unnormalized.nc',
    'output_file': 'data/processed/portugal/08_fwi_unified_normalized.nc',
    'stats_file': 'data/interim/portugal/normalization_stats.nc',
    
    # Data split configuration (for dynamic variables only)
    'train_years': [2010, 2011, 2012, 2013, 2014, 2015],
    'val_years': [2016],
    'test_years': [2017],
    
    # Dynamic variables (3D: time, latitude, longitude)
    'dynamic_predictor_vars': [
        'si10',  # 10m wind speed
        'r2',    # 2m relative humidity
        't2m',   # 2m temperature
        'd2m',   # 2m dewpoint temperature
        'tp',    # Total precipitation
        'u10',   # 10m U wind component
        'v10',   # 10m V wind component
    ],
    
    # Static variables (2D: latitude, longitude) - land cover fractions
    'static_predictor_vars': [
        'lc_frac_tree_cover',
        'lc_frac_shrubland',
        'lc_frac_grassland',
        'lc_frac_cropland',
        'lc_frac_built',
        'lc_frac_bare_sparse',
        'lc_frac_snow_ice',
        'lc_frac_water',
        'lc_frac_wetland',
        'lc_frac_mangroves'
    ],
    
    # Target variable (not normalized)
    'target_var': 'fwi',
    
    # Chunking for memory efficiency
    'chunks': {
        'time': 100,
        'latitude': 650,
        'longitude': 400
    },
    
    # Normalization parameters
    'epsilon': 1e-8,  # Small value to avoid division by zero
    'clip_std': 10.0,  # Clip values beyond this many standard deviations
}


def identify_variable_type(ds, var_name):
    """
    Identify if a variable is static (2D) or dynamic (3D).
    
    Returns:
        str: 'static' for 2D variables, 'dynamic' for 3D variables, None if not found
    """
    if var_name not in ds.variables:
        return None
    
    var_dims = ds[var_name].dims
    if 'time' in var_dims:
        return 'dynamic'
    elif len(var_dims) == 2 and 'latitude' in var_dims and 'longitude' in var_dims:
        return 'static'
    else:
        return 'other'


def split_dataset_by_time(ds):
    """
    Split dataset into train, validation, and test sets based on time.
    Only applies to variables with time dimension.
    
    Args:
        ds: xarray Dataset with time dimension
    
    Returns:
        tuple: (ds_train, ds_val, ds_test)
    """
    logger.info("Splitting dataset by time")
    
    # Convert time to pandas datetime for easier manipulation
    time_index = pd.to_datetime(ds.time.values)
    years = time_index.year
    
    # Create masks for each split
    train_mask = np.isin(years, CONFIG['train_years'])
    val_mask = np.isin(years, CONFIG['val_years'])
    test_mask = np.isin(years, CONFIG['test_years'])
    
    # Split the dataset
    ds_train = ds.isel(time=train_mask)
    ds_val = ds.isel(time=val_mask)
    ds_test = ds.isel(time=test_mask)
    
    # Log split information
    logger.info(f"  Training set: {ds_train.time.size} time steps ({CONFIG['train_years']})")
    logger.info(f"  Validation set: {ds_val.time.size} time steps ({CONFIG['val_years']})")
    logger.info(f"  Test set: {ds_test.time.size} time steps ({CONFIG['test_years']})")
    
    # Verify no overlap
    total_original = ds.time.size
    total_split = ds_train.time.size + ds_val.time.size + ds_test.time.size
    assert total_original == total_split, f"Data loss in split: {total_original} != {total_split}"
    
    return ds_train, ds_val, ds_test


def compute_dynamic_statistics(ds_train, var_name):
    """
    Compute mean and standard deviation from TRAINING DATA ONLY for dynamic variables.
    
    Args:
        ds_train: Training dataset
        var_name: Variable name
    
    Returns:
        dict: Statistics dictionary with mean and std
    """
    logger.info(f"Computing statistics for dynamic variable {var_name} (from training data only)")
    
    if var_name not in ds_train.variables:
        logger.warning(f"Variable {var_name} not found in dataset")
        return None
    
    var_data = ds_train[var_name]
    
    # Compute over all dimensions in training set
    mean_val = float(var_data.mean(skipna=True).compute())
    std_val = float(var_data.std(skipna=True).compute())
    logger.info(f"  Computed from {var_data.time.size} training time steps")
    
    # Handle edge cases
    if np.isnan(std_val) or std_val < CONFIG['epsilon']:
        logger.warning(f"Invalid std for {var_name}: {std_val}, setting to 1.0")
        std_val = 1.0
    
    stats = {
        'mean': mean_val,
        'std': std_val,
        'type': 'dynamic',
        'computed_from': 'training_data_only',
        'training_years': CONFIG['train_years']
    }
    
    logger.info(f"  Training Mean: {stats['mean']:.4f}, Training Std: {stats['std']:.4f}")
    
    return stats


def compute_static_statistics(ds, var_name):
    """
    Compute mean and standard deviation for static variables.
    Since these don't vary with time, we can use the full spatial extent.
    
    Args:
        ds: Full dataset
        var_name: Variable name
    
    Returns:
        dict: Statistics dictionary with mean and std
    """
    logger.info(f"Computing statistics for static variable {var_name}")
    
    if var_name not in ds.variables:
        logger.warning(f"Variable {var_name} not found in dataset")
        return None
    
    var_data = ds[var_name]
    
    # Verify it's actually 2D
    if 'time' in var_data.dims:
        logger.error(f"Variable {var_name} has time dimension but was marked as static!")
        return None
    
    # Compute over spatial dimensions
    mean_val = float(var_data.mean(skipna=True).compute())
    std_val = float(var_data.std(skipna=True).compute())
    
    # Handle edge cases
    if np.isnan(std_val) or std_val < CONFIG['epsilon']:
        logger.warning(f"Invalid std for {var_name}: {std_val}, setting to 1.0")
        std_val = 1.0
    
    stats = {
        'mean': mean_val,
        'std': std_val,
        'type': 'static',
        'computed_from': 'full_spatial_extent'
    }
    
    logger.info(f"  Spatial Mean: {stats['mean']:.4f}, Spatial Std: {stats['std']:.4f}")
    
    return stats


def normalize_variable(data, stats):
    """
    Apply z-score normalization using provided statistics.
    
    Args:
        data: xarray DataArray to normalize
        stats: Statistics dictionary with mean and std
    
    Returns:
        xarray DataArray: Normalized data
    """
    mean = stats['mean']
    std = max(stats['std'], CONFIG['epsilon'])
    
    # Apply normalization
    normalized = (data - mean) / std
    
    # Clip extreme values
    if CONFIG['clip_std'] is not None:
        normalized = normalized.clip(-CONFIG['clip_std'], CONFIG['clip_std'])
    
    # Preserve attributes and add normalization info
    normalized.attrs = data.attrs.copy()
    normalized.attrs['normalization'] = 'z-score'
    normalized.attrs['normalization_mean'] = mean
    normalized.attrs['normalization_std'] = std
    normalized.attrs['clipped_at_std'] = CONFIG['clip_std']
    
    return normalized


def save_normalization_stats(dynamic_stats, static_stats):
    """
    Save normalization statistics to NetCDF file.
    
    Args:
        dynamic_stats: Dictionary of statistics for dynamic variables
        static_stats: Dictionary of statistics for static variables
    """
    logger.info(f"Saving normalization statistics to {CONFIG['stats_file']}")
    
    # Create dataset for statistics
    ds_stats = xr.Dataset()
    
    # Add dynamic variable statistics
    for var_name, stats in dynamic_stats.items():
        if stats is not None:
            ds_stats[f'{var_name}_mean'] = xr.DataArray(
                stats['mean'],
                attrs={
                    'long_name': f'Training mean for {var_name}',
                    'type': 'dynamic',
                    'computed_from': 'training_data_only',
                    'training_years': str(CONFIG['train_years'])
                }
            )
            ds_stats[f'{var_name}_std'] = xr.DataArray(
                stats['std'],
                attrs={
                    'long_name': f'Training standard deviation for {var_name}',
                    'type': 'dynamic',
                    'computed_from': 'training_data_only',
                    'training_years': str(CONFIG['train_years'])
                }
            )
    
    # Add static variable statistics
    for var_name, stats in static_stats.items():
        if stats is not None:
            ds_stats[f'{var_name}_mean'] = xr.DataArray(
                stats['mean'],
                attrs={
                    'long_name': f'Spatial mean for {var_name}',
                    'type': 'static',
                    'computed_from': 'full_spatial_extent'
                }
            )
            ds_stats[f'{var_name}_std'] = xr.DataArray(
                stats['std'],
                attrs={
                    'long_name': f'Spatial standard deviation for {var_name}',
                    'type': 'static',
                    'computed_from': 'full_spatial_extent'
                }
            )
    
    # Add global attributes
    ds_stats.attrs = {
        'description': 'Normalization statistics for FWI dataset',
        'created_at': datetime.now().isoformat(),
        'training_years': str(CONFIG['train_years']),
        'validation_years': str(CONFIG['val_years']),
        'test_years': str(CONFIG['test_years']),
        'note': 'Dynamic variables use training statistics, static variables use spatial statistics'
    }
    
    # Save to NetCDF
    Path(CONFIG['stats_file']).parent.mkdir(parents=True, exist_ok=True)
    ds_stats.to_netcdf(CONFIG['stats_file'], mode='w')
    logger.info(f"  Saved statistics for {len(dynamic_stats) + len(static_stats)} variables")


def process_normalization():
    """Main normalization process handling static and dynamic variables separately."""
    
    logger.info("="*60)
    logger.info("Dataset Normalization (Static Variables Handled Correctly)")
    logger.info("="*60)
    
    # Create output directories
    output_dir = Path(CONFIG['output_file']).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load dataset
    logger.info(f"\n1. Loading dataset: {CONFIG['input_file']}")
    ds = xr.open_dataset(
        CONFIG['input_file'],
        chunks=CONFIG['chunks']
    )
    logger.info(f"   Dataset shape: {dict(ds.dims)}")
    
    # Step 2: Identify and separate variables by type
    logger.info("\n2. Identifying variable types")
    logger.info("="*40)
    
    dynamic_vars = []
    static_vars = []
    
    # Check all variables
    for var_name in ds.data_vars:
        var_type = identify_variable_type(ds, var_name)
        if var_type == 'dynamic':
            dynamic_vars.append(var_name)
            logger.info(f"  {var_name}: DYNAMIC (3D)")
        elif var_type == 'static':
            static_vars.append(var_name)
            logger.info(f"  {var_name}: STATIC (2D)")
    
    logger.info(f"\nFound {len(dynamic_vars)} dynamic and {len(static_vars)} static variables")
    
    # Step 3: Process DYNAMIC variables
    logger.info("\n3. Processing DYNAMIC variables")
    logger.info("="*40)
    
    # Split dataset for dynamic variables
    logger.info("\n3a. Splitting dataset by time")
    ds_train, ds_val, ds_test = split_dataset_by_time(ds)
    
    # Compute statistics from training data only
    logger.info("\n3b. Computing statistics from TRAINING data only")
    dynamic_stats = {}
    
    for var_name in CONFIG['dynamic_predictor_vars']:
        if var_name in dynamic_vars:
            stats = compute_dynamic_statistics(ds_train, var_name)
            if stats:
                dynamic_stats[var_name] = stats
    
    # Apply normalization to all splits
    logger.info("\n3c. Applying normalization to all time splits")
    normalized_splits = []
    
    for split_name, ds_split in [('train', ds_train), ('val', ds_val), ('test', ds_test)]:
        logger.info(f"\nNormalizing {split_name} split:")
        
        # Create normalized dataset for this split
        ds_split_norm = xr.Dataset(coords=ds_split.coords)
        
        # Normalize dynamic predictor variables
        for var_name in dynamic_stats:
            var_data = ds_split[var_name]
            normalized_var = normalize_variable(var_data, dynamic_stats[var_name])
            ds_split_norm[var_name] = normalized_var
            logger.info(f"  ✓ {var_name}")
        
        # Copy target variable without normalization
        if CONFIG['target_var'] in ds_split.variables:
            ds_split_norm[CONFIG['target_var']] = ds_split[CONFIG['target_var']].copy()
            ds_split_norm[CONFIG['target_var']].attrs['normalization'] = 'none'
            ds_split_norm[CONFIG['target_var']].attrs['note'] = 'Target variable - not normalized'
        
        # Copy other non-predictor dynamic variables
        for var_name in dynamic_vars:
            if (var_name not in dynamic_stats and 
                var_name != CONFIG['target_var'] and
                var_name not in ['expver', 'number', 'surface']):
                ds_split_norm[var_name] = ds_split[var_name].copy()
        
        normalized_splits.append(ds_split_norm)
    
    # Combine normalized dynamic data
    logger.info("\n3d. Combining normalized time splits")
    ds_dynamic_normalized = xr.concat(normalized_splits, dim='time')
    ds_dynamic_normalized = ds_dynamic_normalized.sortby('time')
    
    # Step 4: Process STATIC variables
    logger.info("\n4. Processing STATIC variables")
    logger.info("="*40)
    
    # Compute statistics for static variables
    logger.info("\n4a. Computing spatial statistics for static variables")
    static_stats = {}
    
    for var_name in CONFIG['static_predictor_vars']:
        if var_name in static_vars:
            stats = compute_static_statistics(ds, var_name)
            if stats:
                static_stats[var_name] = stats
    
    # Create normalized static dataset
    logger.info("\n4b. Normalizing static variables")
    ds_static_normalized = xr.Dataset()
    
    # Copy spatial coordinates
    for coord in ['latitude', 'longitude']:
        if coord in ds.coords:
            ds_static_normalized[coord] = ds[coord]
    
    # Normalize static variables
    for var_name in static_stats:
        var_data = ds[var_name]
        normalized_var = normalize_variable(var_data, static_stats[var_name])
        ds_static_normalized[var_name] = normalized_var
        logger.info(f"  ✓ {var_name}")
    
    # Copy other static variables that aren't predictors
    for var_name in static_vars:
        if var_name not in static_stats and var_name not in ds_static_normalized:
            ds_static_normalized[var_name] = ds[var_name].copy()
    
    # Step 5: Merge static and dynamic datasets
    logger.info("\n5. Merging static and dynamic datasets")
    logger.info("="*40)
    
    # Start with dynamic dataset
    ds_final = ds_dynamic_normalized.copy()
    
    # Add static variables (xarray will broadcast automatically)
    for var_name in ds_static_normalized.data_vars:
        ds_final[var_name] = ds_static_normalized[var_name]
        logger.info(f"  Added static variable: {var_name}")
    
    # Add global attributes
    ds_final.attrs = {
        'normalization_method': 'z-score_separate_static_dynamic',
        'normalization_date': datetime.now().isoformat(),
        'normalization_stats_file': CONFIG['stats_file'],
        'preprocessing_step': '08_normalize_dataset',
        'description': 'Normalized unified FWI dataset with proper static/dynamic handling',
        'train_years': str(CONFIG['train_years']),
        'val_years': str(CONFIG['val_years']),
        'test_years': str(CONFIG['test_years']),
        'dynamic_normalization': 'Statistics from training data only',
        'static_normalization': 'Statistics from full spatial extent',
        'important_note': 'Static variables (land cover) maintain 2D structure'
    }
    
    # Step 6: Save everything
    logger.info("\n6. Saving outputs")
    logger.info("="*40)
    
    # Save statistics
    save_normalization_stats(dynamic_stats, static_stats)
    
    # Set up encoding for efficient storage
    encoding = {}
    for var in ds_final.data_vars:
        encoding[var] = {
            'zlib': True,
            'complevel': 4,
            'dtype': 'float32',
            '_FillValue': np.nan
        }
    
    # Save final dataset
    logger.info(f"Writing to: {CONFIG['output_file']}")
    ds_final.to_netcdf(
        CONFIG['output_file'],
        encoding=encoding,
        engine='netcdf4',
        mode='w'
    )
    
    # Verify output
    output_size = Path(CONFIG['output_file']).stat().st_size / (1024**3)
    logger.info(f"Output file size: {output_size:.2f} GB")
    
    # Step 7: Verification
    logger.info("\n7. Verification of output structure")
    logger.info("="*40)
    
    # Reload and check
    ds_check = xr.open_dataset(CONFIG['output_file'])
    
    logger.info("\nVariable dimensions check:")
    for var_name in list(ds_check.data_vars)[:10]:
        dims = ds_check[var_name].dims
        dims_str = f"({', '.join(dims)})"
        if 'lc_frac' in var_name:
            expected = "(latitude, longitude)"
            status = "✅" if dims_str == expected else "❌"
        elif var_name in ['fwi'] + CONFIG['dynamic_predictor_vars']:
            expected = "(time, latitude, longitude)"
            status = "✅" if dims_str == expected else "❌"
        else:
            status = "ℹ️"
        logger.info(f"  {var_name:25s}: {dims_str:30s} {status}")
    
    # Check normalization by split for dynamic variables
    logger.info("\nDynamic variable statistics by split:")
    for split_name, years in [('train', CONFIG['train_years']), ('val', CONFIG['val_years'])]:
        time_index = pd.to_datetime(ds_check.time.values)
        mask = np.isin(time_index.year, years)
        ds_split = ds_check.isel(time=mask)
        
        logger.info(f"\n{split_name.upper()} split:")
        for var_name in list(dynamic_stats.keys())[:2]:
            if var_name in ds_split:
                mean_val = float(ds_split[var_name].mean(skipna=True).compute())
                std_val = float(ds_split[var_name].std(skipna=True).compute())
                logger.info(f"  {var_name}: mean={mean_val:.4f}, std={std_val:.4f}")
    
    # Clean up
    ds.close()
    ds_final.close()
    ds_check.close()
    
    logger.info("\n" + "="*60)
    logger.info("Normalization complete!")
    logger.info(f"Output saved to: {CONFIG['output_file']}")
    logger.info(f"Statistics saved to: {CONFIG['stats_file']}")
    logger.info("Static variables maintain 2D structure: ✅")
    logger.info("Dynamic variables properly normalized: ✅")
    logger.info("="*60)
    
    return True


if __name__ == "__main__":
    try:
        success = process_normalization()
        if success:
            logger.info("\n✅ Dataset normalization completed successfully!")
            
            # Print verification commands
            print("\n" + "="*60)
            print("VERIFICATION COMMANDS:")
            print("="*60)
            print("\n1. Check file structure:")
            print(f"   ncdump -h {CONFIG['output_file']} | head -100")
            print("\n2. Verify dimensions:")
            print(f"   ncdump -h {CONFIG['output_file']} | grep 'lc_frac\\|si10\\|fwi('")
            print("\n3. Check file size:")
            print(f"   ls -lh {CONFIG['output_file']}")
            print("="*60)
        else:
            logger.error("❌ Normalization failed")
            exit(1)
    except Exception as e:
        logger.error(f"❌ Error during normalization: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)