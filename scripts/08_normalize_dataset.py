#!/usr/bin/env python3
"""
Script 08: Normalize Dataset (Corrected Version - No Data Leakage)
Final preprocessing step - normalizes the unified dataset for model training.

CRITICAL: This script prevents data leakage by:
1. Splitting data into train/val/test BEFORE computing statistics
2. Computing normalization statistics ONLY from training data
3. Applying training statistics to all splits

Data splits:
- Training: 2010-2015 (6 years)
- Validation: 2016 (1 year)  
- Test: 2017 (1 year)
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
    
    # Data split configuration
    'train_years': [2010, 2011, 2012, 2013, 2014, 2015],
    'val_years': [2016],
    'test_years': [2017],
    
    # Variables to normalize (predictors)
    'predictor_vars': [
        'si10',  # 10m wind speed
        'r2',    # 2m relative humidity
        't2m',   # 2m temperature
        'd2m',   # 2m dewpoint temperature
        'tp',    # Total precipitation
        'u10',   # 10m U wind component
        'v10',   # 10m V wind component
        # Land cover fractions
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


def split_dataset_by_time(ds):
    """
    Split dataset into train, validation, and test sets based on time.
    
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


def compute_training_statistics(ds_train, var_name):
    """
    Compute mean and standard deviation from TRAINING DATA ONLY.
    
    Args:
        ds_train: Training dataset
        var_name: Variable name
    
    Returns:
        dict: Statistics dictionary with mean and std
    """
    logger.info(f"Computing statistics for {var_name} (from training data only)")
    
    if var_name not in ds_train.variables:
        logger.warning(f"Variable {var_name} not found in dataset")
        return None
    
    var_data = ds_train[var_name]
    
    # For land cover variables (2D), compute over spatial dimensions only
    if 'time' not in var_data.dims:
        # Land cover is static, so statistics are the same regardless of split
        mean_val = float(var_data.mean(skipna=True).compute())
        std_val = float(var_data.std(skipna=True).compute())
        logger.info(f"  Static variable (no time dimension)")
    else:
        # For time-varying variables, compute over all dimensions in training set
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
        'computed_from': 'training_data_only',
        'training_years': CONFIG['train_years']
    }
    
    logger.info(f"  Training Mean: {stats['mean']:.4f}, Training Std: {stats['std']:.4f}")
    
    return stats


def normalize_with_training_stats(ds, var_name, training_stats):
    """
    Apply normalization using statistics computed from training data.
    
    Args:
        ds: Dataset to normalize (can be train, val, or test)
        var_name: Variable name
        training_stats: Statistics computed from training data
    
    Returns:
        xarray DataArray: Normalized variable
    """
    if var_name not in ds.variables:
        return None
    
    var_data = ds[var_name]
    mean = training_stats['mean']
    std = max(training_stats['std'], CONFIG['epsilon'])
    
    # Apply normalization using training statistics
    normalized = (var_data - mean) / std
    
    # Clip extreme values
    if CONFIG['clip_std'] is not None:
        normalized = normalized.clip(-CONFIG['clip_std'], CONFIG['clip_std'])
    
    # Preserve attributes and add normalization info
    normalized.attrs = var_data.attrs.copy()
    normalized.attrs['normalization'] = 'z-score'
    normalized.attrs['training_mean'] = mean
    normalized.attrs['training_std'] = std
    normalized.attrs['normalized_with'] = 'training_statistics_only'
    normalized.attrs['clipped_at_std'] = CONFIG['clip_std']
    
    return normalized


def save_normalization_stats(stats_dict):
    """
    Save normalization statistics to NetCDF file.
    
    Args:
        stats_dict: Dictionary containing statistics for each variable
    """
    logger.info(f"Saving normalization statistics to {CONFIG['stats_file']}")
    
    # Create dataset for statistics
    ds_stats = xr.Dataset()
    
    # Add statistics for each variable
    for var_name, stats in stats_dict.items():
        if stats is not None:
            # Create data arrays for mean and std
            ds_stats[f'{var_name}_mean'] = xr.DataArray(
                stats['mean'],
                attrs={
                    'long_name': f'Training mean for {var_name}',
                    'computed_from': 'training_data_only',
                    'training_years': str(CONFIG['train_years'])
                }
            )
            ds_stats[f'{var_name}_std'] = xr.DataArray(
                stats['std'],
                attrs={
                    'long_name': f'Training standard deviation for {var_name}',
                    'computed_from': 'training_data_only',
                    'training_years': str(CONFIG['train_years'])
                }
            )
    
    # Add global attributes
    ds_stats.attrs = {
        'description': 'Normalization statistics computed from training data only',
        'created_at': datetime.now().isoformat(),
        'training_years': str(CONFIG['train_years']),
        'validation_years': str(CONFIG['val_years']),
        'test_years': str(CONFIG['test_years']),
        'note': 'Statistics computed to prevent data leakage in ML pipeline'
    }
    
    # Save to NetCDF
    Path(CONFIG['stats_file']).parent.mkdir(parents=True, exist_ok=True)
    ds_stats.to_netcdf(CONFIG['stats_file'], mode='w')
    logger.info(f"  Saved statistics for {len(stats_dict)} variables")


def process_normalization():
    """Main normalization process with proper train/val/test split."""
    
    logger.info("="*60)
    logger.info("Starting dataset normalization (No Data Leakage Version)")
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
    
    # Step 2: Split dataset BEFORE computing any statistics
    logger.info("\n2. Splitting dataset by time (BEFORE computing statistics)")
    ds_train, ds_val, ds_test = split_dataset_by_time(ds)
    
    # Step 3: Compute statistics from TRAINING DATA ONLY
    logger.info("\n3. Computing normalization statistics from TRAINING DATA ONLY")
    logger.info("="*40)
    
    training_stats = {}
    existing_predictors = []
    
    for var_name in CONFIG['predictor_vars']:
        if var_name in ds_train.variables:
            stats = compute_training_statistics(ds_train, var_name)
            if stats:
                training_stats[var_name] = stats
                existing_predictors.append(var_name)
        else:
            logger.warning(f"Variable {var_name} not found, skipping")
    
    # Save statistics
    save_normalization_stats(training_stats)
    
    # Step 4: Apply normalization to ALL splits using training statistics
    logger.info("\n4. Applying normalization to all splits using training statistics")
    logger.info("="*40)
    
    # Process each split
    normalized_splits = []
    
    for split_name, ds_split in [('train', ds_train), ('val', ds_val), ('test', ds_test)]:
        logger.info(f"\nNormalizing {split_name} split:")
        
        # Create normalized dataset for this split
        ds_split_norm = xr.Dataset(coords=ds_split.coords)
        
        # Normalize predictor variables
        for var_name in existing_predictors:
            normalized_var = normalize_with_training_stats(
                ds_split, var_name, training_stats[var_name]
            )
            if normalized_var is not None:
                ds_split_norm[var_name] = normalized_var
                logger.info(f"  ✓ {var_name}")
        
        # Copy target variable without normalization
        if CONFIG['target_var'] in ds_split.variables:
            ds_split_norm[CONFIG['target_var']] = ds_split[CONFIG['target_var']].copy()
            ds_split_norm[CONFIG['target_var']].attrs['normalization'] = 'none'
            ds_split_norm[CONFIG['target_var']].attrs['note'] = 'Target variable - not normalized'
        
        # Copy other variables
        other_vars = set(ds_split.data_vars) - set(existing_predictors) - {CONFIG['target_var']}
        for var_name in other_vars:
            if var_name not in ['expver', 'number', 'surface']:
                ds_split_norm[var_name] = ds_split[var_name].copy()
        
        # Add split identifier attribute
        ds_split_norm.attrs['split'] = split_name
        ds_split_norm.attrs['years'] = CONFIG[f'{split_name}_years']
        
        normalized_splits.append(ds_split_norm)
    
    # Step 5: Combine normalized splits
    logger.info("\n5. Combining normalized splits")
    logger.info("="*40)
    
    ds_normalized = xr.concat(normalized_splits, dim='time')
    ds_normalized = ds_normalized.sortby('time')
    
    # Add global attributes
    ds_normalized.attrs = {
        'normalization_method': 'z-score_training_only',
        'normalization_date': datetime.now().isoformat(),
        'normalization_stats_file': CONFIG['stats_file'],
        'preprocessing_step': '08_normalize_dataset',
        'description': 'Normalized unified FWI dataset (no data leakage)',
        'train_years': str(CONFIG['train_years']),
        'val_years': str(CONFIG['val_years']),
        'test_years': str(CONFIG['test_years']),
        'statistics_computed_from': 'training_data_only',
        'important_note': 'All normalization statistics computed from training data only to prevent data leakage'
    }
    
    # Step 6: Save normalized dataset
    logger.info("\n6. Saving normalized dataset")
    logger.info("="*40)
    
    # Set up encoding for efficient storage
    encoding = {}
    for var in ds_normalized.data_vars:
        encoding[var] = {
            'zlib': True,
            'complevel': 4,
            'dtype': 'float32',
            '_FillValue': np.nan
        }
    
    # Save
    logger.info(f"Writing to: {CONFIG['output_file']}")
    ds_normalized.to_netcdf(
        CONFIG['output_file'],
        encoding=encoding,
        engine='netcdf4',
        mode='w'
    )
    
    # Verify output
    output_size = Path(CONFIG['output_file']).stat().st_size / (1024**3)
    logger.info(f"Output file size: {output_size:.2f} GB")
    
    # Step 7: Verification of train/val/test statistics
    logger.info("\n7. Verification of normalization by split")
    logger.info("="*40)
    
    # Reload and check statistics by split
    ds_check = xr.open_dataset(CONFIG['output_file'])
    
    for split_name, years in [('train', CONFIG['train_years']), 
                               ('val', CONFIG['val_years']), 
                               ('test', CONFIG['test_years'])]:
        logger.info(f"\n{split_name.upper()} split statistics:")
        
        # Get time mask for this split
        time_index = pd.to_datetime(ds_check.time.values)
        mask = np.isin(time_index.year, years)
        ds_split_check = ds_check.isel(time=mask)
        
        # Check a few variables
        for var_name in existing_predictors[:3]:
            if var_name in ds_split_check:
                var_data = ds_split_check[var_name]
                mean_check = float(var_data.mean(skipna=True).compute())
                std_check = float(var_data.std(skipna=True).compute())
                logger.info(f"  {var_name}: mean={mean_check:.4f}, std={std_check:.4f}")
    
    # Note about expected statistics
    logger.info("\nNOTE: Only the TRAINING set should have mean≈0 and std≈1")
    logger.info("      Validation and test sets will have different statistics")
    logger.info("      This is CORRECT and prevents data leakage!")
    
    # Clean up
    ds.close()
    ds_normalized.close()
    ds_check.close()
    
    logger.info("\n" + "="*60)
    logger.info("Normalization complete (No Data Leakage Version)!")
    logger.info(f"Output saved to: {CONFIG['output_file']}")
    logger.info(f"Statistics saved to: {CONFIG['stats_file']}")
    logger.info("="*60)
    
    return True


if __name__ == "__main__":
    try:
        success = process_normalization()
        if success:
            logger.info("\n✅ Dataset normalization completed successfully!")
            logger.info("   Statistics computed from training data only")
            logger.info("   No data leakage in normalization process")
            
            # Print verification commands
            print("\n" + "="*60)
            print("VERIFICATION COMMANDS:")
            print("="*60)
            print("\n1. Check file size:")
            print(f"   ls -lh {CONFIG['output_file']}")
            print("\n2. Run verification script:")
            print("   python sandbox/verify_normalization.py")
            print("\n3. Check statistics file:")
            print(f"   ncdump -h {CONFIG['stats_file']}")
            print("="*60)
        else:
            logger.error("❌ Normalization failed")
            exit(1)
    except Exception as e:
        logger.error(f"❌ Error during normalization: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)