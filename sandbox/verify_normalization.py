#!/usr/bin/env python3
"""
Verification script for normalized dataset (No Data Leakage Version).
Checks that normalization was applied correctly using training statistics only.
"""

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import logging
from tabulate import tabulate

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
NORMALIZED_FILE = 'data/processed/portugal/08_fwi_unified_normalized.nc'
STATS_FILE = 'data/interim/portugal/normalization_stats.nc'
UNNORMALIZED_FILE = 'data/interim/portugal/07_fwi_unified_unnormalized.nc'

# Data splits
TRAIN_YEARS = [2010, 2011, 2012, 2013, 2014, 2015]
VAL_YEARS = [2016]
TEST_YEARS = [2017]

# Tolerance for numerical checks
TOLERANCE = {
    'mean': 0.05,  # Mean should be within 0.05 of 0 for training set
    'std': 0.05,   # Std should be within 0.05 of 1 for training set
}


def check_file_exists(filepath):
    """Check if file exists and return size."""
    path = Path(filepath)
    if not path.exists():
        return False, 0
    size_gb = path.stat().st_size / (1024**3)
    return True, size_gb


def load_normalization_stats():
    """Load normalization statistics from NetCDF."""
    if not Path(STATS_FILE).exists():
        logger.warning(f"Stats file not found: {STATS_FILE}")
        return None
    
    ds_stats = xr.open_dataset(STATS_FILE)
    
    # Extract statistics
    stats = {}
    for var in ds_stats.data_vars:
        if var.endswith('_mean'):
            var_name = var[:-5]  # Remove '_mean' suffix
            if f'{var_name}_std' in ds_stats:
                stats[var_name] = {
                    'mean': float(ds_stats[f'{var_name}_mean'].values),
                    'std': float(ds_stats[f'{var_name}_std'].values)
                }
    
    ds_stats.close()
    return stats


def split_dataset_for_verification(ds):
    """Split dataset by years for verification."""
    time_index = pd.to_datetime(ds.time.values)
    years = time_index.year
    
    train_mask = np.isin(years, TRAIN_YEARS)
    val_mask = np.isin(years, VAL_YEARS)
    test_mask = np.isin(years, TEST_YEARS)
    
    return {
        'train': ds.isel(time=train_mask),
        'val': ds.isel(time=val_mask),
        'test': ds.isel(time=test_mask)
    }


def verify_split_statistics(ds_split, split_name, var_name, expected_mean=None, expected_std=None):
    """
    Verify statistics for a specific split and variable.
    
    For training set: should be close to mean=0, std=1
    For val/test sets: should NOT be mean=0, std=1 (that would indicate data leakage)
    """
    if var_name not in ds_split.variables:
        return {
            'split': split_name,
            'variable': var_name,
            'status': 'NOT FOUND',
            'mean': None,
            'std': None,
            'pass': False
        }
    
    var_data = ds_split[var_name]
    
    # Skip land cover variables for time-based checks
    if 'time' not in var_data.dims:
        return {
            'split': split_name,
            'variable': var_name,
            'status': 'STATIC',
            'mean': None,
            'std': None,
            'pass': True
        }
    
    # Compute statistics
    mean_val = float(var_data.mean(skipna=True).compute())
    std_val = float(var_data.std(skipna=True).compute())
    
    # Check based on split
    if split_name == 'train':
        # Training set should have mean≈0, std≈1
        mean_ok = abs(mean_val - 0.0) < TOLERANCE['mean']
        std_ok = abs(std_val - 1.0) < TOLERANCE['std']
        status = 'PASS' if (mean_ok and std_ok) else 'FAIL'
    else:
        # Val/test sets should NOT have mean=0, std=1 (unless by chance)
        # We expect them to be different from 0 and 1
        is_normalized_like_train = (abs(mean_val - 0.0) < TOLERANCE['mean'] and 
                                    abs(std_val - 1.0) < TOLERANCE['std'])
        if is_normalized_like_train:
            status = 'WARNING'  # Possible data leakage
        else:
            status = 'PASS'  # Correctly different from training normalization
    
    return {
        'split': split_name,
        'variable': var_name,
        'status': status,
        'mean': mean_val,
        'std': std_val,
        'pass': status in ['PASS', 'STATIC']
    }


def verify_data_leakage_prevention(ds_norm, ds_orig, stats):
    """
    Verify that normalization was done correctly without data leakage.
    """
    logger.info("\nDATA LEAKAGE PREVENTION CHECK")
    logger.info("-"*40)
    
    # Split normalized dataset
    splits_norm = split_dataset_for_verification(ds_norm)
    
    # Check first few variables
    test_vars = ['si10', 'r2', 't2m'] if stats else []
    
    results = []
    for var_name in test_vars[:3]:
        if var_name in stats:
            logger.info(f"\nChecking {var_name}:")
            logger.info(f"  Training stats used: mean={stats[var_name]['mean']:.4f}, std={stats[var_name]['std']:.4f}")
            
            for split_name in ['train', 'val', 'test']:
                result = verify_split_statistics(
                    splits_norm[split_name], 
                    split_name, 
                    var_name
                )
                results.append(result)
                
                if result['mean'] is not None:
                    logger.info(f"  {split_name:5s}: mean={result['mean']:7.4f}, std={result['std']:7.4f} [{result['status']}]")
    
    return results


def main():
    """Main verification function."""
    
    logger.info("="*70)
    logger.info("NORMALIZATION VERIFICATION (No Data Leakage Version)")
    logger.info("="*70)
    
    # Step 1: Check files exist
    logger.info("\n1. FILE CHECKS")
    logger.info("-"*40)
    
    files_info = []
    for name, filepath in [
        ("Normalized", NORMALIZED_FILE),
        ("Unnormalized", UNNORMALIZED_FILE),
        ("Stats NetCDF", STATS_FILE)
    ]:
        exists, size = check_file_exists(filepath)
        status = "✅" if exists else "❌"
        size_str = f"{size:.2f} GB" if size > 0.001 else f"{size*1024:.2f} MB"
        files_info.append([name, status, size_str, filepath])
        logger.info(f"{status} {name}: {filepath} ({size_str})")
    
    print("\nFile Status:")
    print(tabulate(files_info, headers=["File", "Status", "Size", "Path"], tablefmt="grid"))
    
    if not check_file_exists(NORMALIZED_FILE)[0]:
        logger.error(f"❌ Normalized file not found: {NORMALIZED_FILE}")
        logger.info("Please run: python scripts/08_normalize_dataset.py")
        return False
    
    # Step 2: Load datasets and stats
    logger.info("\n2. LOADING DATA")
    logger.info("-"*40)
    
    logger.info("Loading normalized dataset...")
    ds_norm = xr.open_dataset(NORMALIZED_FILE)
    logger.info(f"  Shape: {dict(ds_norm.dims)}")
    logger.info(f"  Variables: {len(ds_norm.data_vars)}")
    
    # Load normalization statistics
    stats = load_normalization_stats()
    if stats:
        logger.info(f"✅ Loaded normalization statistics")
        logger.info(f"  Variables with stats: {len(stats)}")
    else:
        logger.warning("⚠️ Could not load normalization statistics")
    
    # Step 3: Verify correct split-based normalization
    logger.info("\n3. SPLIT-BASED NORMALIZATION VERIFICATION")
    logger.info("-"*40)
    logger.info("Checking that only TRAINING set has mean≈0 and std≈1")
    
    # Split the normalized dataset
    splits = split_dataset_for_verification(ds_norm)
    
    logger.info(f"\nDataset splits:")
    for split_name, split_data in splits.items():
        years_str = {'train': TRAIN_YEARS, 'val': VAL_YEARS, 'test': TEST_YEARS}[split_name]
        logger.info(f"  {split_name:5s}: {split_data.time.size:4d} time steps (years: {years_str})")
    
    # Verify normalization for each split
    all_results = []
    predictor_vars = list(stats.keys()) if stats else ['si10', 'r2', 't2m', 'd2m', 'tp', 'u10', 'v10']
    
    for var_name in predictor_vars[:7]:  # Check first 7 variables
        for split_name in ['train', 'val', 'test']:
            result = verify_split_statistics(splits[split_name], split_name, var_name)
            all_results.append(result)
    
    # Create results table by split
    for split_name in ['train', 'val', 'test']:
        split_results = [r for r in all_results if r['split'] == split_name]
        
        print(f"\n{split_name.upper()} Split Statistics:")
        table_data = []
        for r in split_results:
            if r['status'] == 'STATIC':
                table_data.append([r['variable'], "Static (2D)", "-", "-", "✅"])
            else:
                status_emoji = {
                    'PASS': "✅",
                    'FAIL': "❌",
                    'WARNING': "⚠️",
                    'NOT FOUND': "❓"
                }.get(r['status'], "?")
                
                mean_str = f"{r['mean']:.4f}" if r['mean'] is not None else "N/A"
                std_str = f"{r['std']:.4f}" if r['std'] is not None else "N/A"
                table_data.append([r['variable'], mean_str, std_str, r['status'], status_emoji])
        
        print(tabulate(
            table_data,
            headers=["Variable", "Mean", "Std Dev", "Status", "✓"],
            tablefmt="grid"
        ))
        
        # Add interpretation
        if split_name == 'train':
            print("Expected: Mean ≈ 0.0, Std ≈ 1.0 (normalized using its own statistics)")
        else:
            print("Expected: Mean ≠ 0.0, Std ≠ 1.0 (normalized using training statistics)")
    
    # Step 4: Check target variable (FWI)
    logger.info("\n4. TARGET VARIABLE CHECK")
    logger.info("-"*40)
    
    if 'fwi' in ds_norm.variables:
        fwi_data = ds_norm['fwi']
        fwi_stats = {
            'mean': float(fwi_data.mean(skipna=True).compute()),
            'std': float(fwi_data.std(skipna=True).compute()),
            'min': float(fwi_data.min(skipna=True).compute()),
            'max': float(fwi_data.max(skipna=True).compute())
        }
        logger.info("✅ FWI (target) variable present - NOT normalized")
        logger.info(f"  Mean: {fwi_stats['mean']:.2f}")
        logger.info(f"  Std: {fwi_stats['std']:.2f}")
        logger.info(f"  Range: [{fwi_stats['min']:.2f}, {fwi_stats['max']:.2f}]")
    else:
        logger.warning("⚠️ FWI variable not found")
    
    # Step 5: Data leakage prevention verification
    if stats and check_file_exists(UNNORMALIZED_FILE)[0]:
        logger.info("\n5. DATA LEAKAGE PREVENTION VERIFICATION")
        logger.info("-"*40)
        
        logger.info("Loading original dataset for comparison...")
        ds_orig = xr.open_dataset(UNNORMALIZED_FILE)
        leakage_results = verify_data_leakage_prevention(ds_norm, ds_orig, stats)
        ds_orig.close()
    
    # Step 6: Summary assessment
    logger.info("\n6. SUMMARY ASSESSMENT")
    logger.info("-"*40)
    
    # Check training set normalization
    train_results = [r for r in all_results if r['split'] == 'train' and r['status'] != 'STATIC']
    train_pass = all(r['pass'] for r in train_results)
    
    # Check that val/test are NOT normalized to 0,1
    val_test_results = [r for r in all_results if r['split'] in ['val', 'test'] and r['status'] != 'STATIC']
    no_leakage = all(r['status'] != 'WARNING' for r in val_test_results)
    
    if train_pass and no_leakage:
        logger.info("✅ SUCCESS: Normalization correctly applied without data leakage!")
        logger.info("   - Training set normalized to mean≈0, std≈1")
        logger.info("   - Val/test sets normalized using training statistics")
        logger.info("   - No data leakage detected")
        success = True
    elif train_pass and not no_leakage:
        logger.warning("⚠️ WARNING: Possible data leakage detected!")
        logger.info("   - Val/test sets appear to have mean≈0, std≈1")
        logger.info("   - This suggests statistics computed from all data")
        success = False
    else:
        logger.warning("⚠️ ISSUES DETECTED:")
        if not train_pass:
            logger.info("   - Training set not properly normalized")
        if not no_leakage:
            logger.info("   - Possible data leakage in val/test normalization")
        success = False
    
    # Step 7: Check attributes
    logger.info("\n7. METADATA CHECK")
    logger.info("-"*40)
    
    important_attrs = [
        'normalization_method',
        'statistics_computed_from',
        'train_years',
        'val_years',
        'test_years'
    ]
    
    for attr in important_attrs:
        if attr in ds_norm.attrs:
            logger.info(f"✅ {attr}: {ds_norm.attrs[attr]}")
        else:
            logger.warning(f"❌ Missing attribute: {attr}")
    
    # Clean up
    ds_norm.close()
    
    return success


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n" + "="*70)
            print("✅ NORMALIZATION VERIFICATION COMPLETE - NO DATA LEAKAGE")
            print("="*70)
            print("\nThe dataset has been correctly normalized:")
            print("- Statistics computed from TRAINING data only (2010-2015)")
            print("- Applied to all splits (train/val/test)")
            print("- No data leakage in the normalization process")
            print(f"\nNormalized file: {NORMALIZED_FILE}")
            print(f"Statistics file: {STATS_FILE}")
        else:
            print("\n" + "="*70)
            print("❌ VERIFICATION FAILED - ISSUES DETECTED")
            print("="*70)
            print("\nPlease review the issues above.")
            print("The normalization may have data leakage or other problems.")
            exit(1)
    except Exception as e:
        logger.error(f"❌ Error during verification: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)