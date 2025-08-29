#!/usr/bin/env python3
"""
Step 7: Create unified dataset by merging all processed datasets.
Follows the hierarchy: UERRA > ERA5-Land > ERA5 Atmospheric
Outputs an unnormalized unified dataset.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import xarray as xr
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration."""
    import yaml
    with open('configs/params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Resolve paths
    def resolve_vars(obj, root):
        if isinstance(obj, str) and '${' in obj:
            import re
            pattern = r'\$\{([^}]+)\}'
            def replacer(match):
                keys = match.group(1).split('.')
                value = root
                for key in keys:
                    value = value[key]
                return str(value)
            return re.sub(pattern, replacer, obj)
        elif isinstance(obj, dict):
            return {k: resolve_vars(v, root) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_vars(item, root) for item in obj]
        return obj
    
    return resolve_vars(config, config)


def load_dataset(path: Path, dataset_name: str) -> Optional[xr.Dataset]:
    """Load a dataset with error handling."""
    if not path.exists():
        logger.warning(f"  ⚠️  {dataset_name} not found at {path}")
        return None
    
    try:
        ds = xr.open_dataset(path)
        logger.info(f"  ✓ Loaded {dataset_name}: {dict(ds.sizes)}")
        logger.info(f"    Variables: {list(ds.data_vars)}")
        return ds
    except Exception as e:
        logger.error(f"  ✗ Failed to load {dataset_name}: {e}")
        return None


def standardize_coordinates(ds: xr.Dataset, master_grid: xr.Dataset) -> xr.Dataset:
    """Ensure dataset has same coordinates as master grid."""
    # Check if coordinates match
    lat_match = np.allclose(ds.latitude.values, master_grid.latitude.values, rtol=1e-5)
    lon_match = np.allclose(ds.longitude.values, master_grid.longitude.values, rtol=1e-5)
    
    if not (lat_match and lon_match):
        logger.warning("    Coordinates don't match master grid, reindexing...")
        ds = ds.reindex(
            latitude=master_grid.latitude,
            longitude=master_grid.longitude,
            method='nearest',
            tolerance=0.01  # 0.01 degrees tolerance
        )
        logger.info("    ✓ Reindexed to master grid")
    
    return ds


def merge_hierarchical(datasets: Dict[str, xr.Dataset], hierarchy: List[str]) -> xr.Dataset:
    """
    Merge datasets according to hierarchy.
    Higher priority datasets override lower priority ones.
    """
    logger.info("\nMerging datasets hierarchically...")
    logger.info(f"  Hierarchy: {' > '.join(hierarchy)}")
    
    # Start with the lowest priority dataset
    merged = None
    
    for priority_level, source_name in enumerate(reversed(hierarchy)):
        if source_name not in datasets or datasets[source_name] is None:
            logger.info(f"  Skipping {source_name} (not available)")
            continue
        
        ds = datasets[source_name]
        logger.info(f"\n  Processing {source_name} (priority {len(hierarchy) - priority_level}):")
        logger.info(f"    Variables: {list(ds.data_vars)}")
        
        if merged is None:
            # First dataset becomes the base
            merged = ds.copy()
            logger.info(f"    → Base dataset initialized with {len(ds.data_vars)} variables")
        else:
            # Merge with existing data
            # For overlapping variables, the new dataset takes precedence
            overlapping_vars = set(ds.data_vars) & set(merged.data_vars)
            new_vars = set(ds.data_vars) - set(merged.data_vars)
            
            if overlapping_vars:
                logger.info(f"    → Overriding {len(overlapping_vars)} variables: {sorted(overlapping_vars)}")
                for var in overlapping_vars:
                    merged[var] = ds[var]
            
            if new_vars:
                logger.info(f"    → Adding {len(new_vars)} new variables: {sorted(new_vars)}")
                for var in new_vars:
                    merged[var] = ds[var]
    
    return merged


def calculate_derived_variables(ds: xr.Dataset) -> xr.Dataset:
    """Calculate any derived variables needed."""
    logger.info("\nCalculating derived variables...")
    
    # Calculate wind speed if u10 and v10 are available
    if 'u10' in ds and 'v10' in ds:
        ds['wind_speed'] = np.sqrt(ds['u10']**2 + ds['v10']**2)
        ds['wind_speed'].attrs = {
            'long_name': '10 metre wind speed',
            'units': 'm/s',
            'description': 'Calculated from U and V wind components'
        }
        logger.info("  ✓ Calculated wind_speed from u10 and v10")
    
    # Calculate relative humidity if available
    if 't2m' in ds and 'd2m' in ds:
        # Magnus formula for saturation vapor pressure
        # es = 6.112 * exp((17.67 * T) / (T + 243.5))  where T is in Celsius
        t_celsius = ds['t2m'] - 273.15  # Convert from Kelvin to Celsius
        td_celsius = ds['d2m'] - 273.15
        
        es = 6.112 * np.exp((17.67 * t_celsius) / (t_celsius + 243.5))
        e = 6.112 * np.exp((17.67 * td_celsius) / (td_celsius + 243.5))
        
        ds['relative_humidity'] = (e / es) * 100
        ds['relative_humidity'] = ds['relative_humidity'].clip(0, 100)  # Ensure 0-100% range
        ds['relative_humidity'].attrs = {
            'long_name': '2 metre relative humidity',
            'units': '%',
            'description': 'Calculated from temperature and dewpoint'
        }
        logger.info("  ✓ Calculated relative_humidity from t2m and d2m")
    
    return ds


def add_global_metadata(ds: xr.Dataset, config: dict) -> xr.Dataset:
    """Add comprehensive global metadata."""
    geo = config['geography']
    
    ds.attrs = {
        'title': 'Unified FWI Dataset for Portugal (Unnormalized)',
        'description': 'Merged dataset from multiple sources for wildfire prediction',
        'creation_date': pd.Timestamp.now().isoformat(),
        'region': 'Portugal Continental',
        'lat_bounds': f"{geo['lat_min']} to {geo['lat_max']}",
        'lon_bounds': f"{geo['lon_min']} to {geo['lon_max']}",
        'spatial_resolution': '1km (0.01 degrees)',
        'data_hierarchy': 'UERRA > ERA5-Land > ERA5-Atmospheric',
        'processing_pipeline': 'FWI Resolution Enhancement Pipeline v1.0',
        'sources': 'UERRA, ERA5-Land, ERA5 Atmospheric, ESA WorldCover, FWI',
        'coordinate_system': 'WGS84 (EPSG:4326)'
    }
    
    # Add time coverage if available
    if 'time' in ds.dims:
        time_start = str(ds.time.values[0])[:19]
        time_end = str(ds.time.values[-1])[:19]
        ds.attrs['temporal_coverage'] = f"{time_start} to {time_end}"
        ds.attrs['temporal_resolution'] = 'Daily'
    
    return ds


def quality_check(ds: xr.Dataset) -> None:
    """Perform quality checks on the unified dataset."""
    logger.info("\n" + "="*60)
    logger.info("QUALITY CHECK")
    logger.info("="*60)
    
    # Dataset overview
    logger.info("\nDataset Overview:")
    logger.info(f"  Dimensions: {dict(ds.sizes)}")
    logger.info(f"  Total variables: {len(ds.data_vars)}")
    logger.info(f"  Coordinate variables: {list(ds.coords)}")
    
    # Variable categories
    categories = {
        'FWI indices': ['fwi', 'ffmc', 'dmc', 'dc', 'isi', 'bui'],
        'Temperature': ['t2m', 'd2m', 'skt'],
        'Precipitation': ['tp'],
        'Wind': ['u10', 'v10', 'wind_speed'],
        'Humidity': ['relative_humidity'],
        'Radiation': ['ssr', 'str', 'ssrd', 'strd'],
        'Surface': ['sp', 'blh', 'e', 'slhf', 'sshf'],
        'Vegetation': ['lai_hv', 'lai_lv', 'fal'],
        'Soil': ['swvl1', 'swvl2', 'swvl3', 'swvl4', 'stl1', 'stl2', 'stl3', 'stl4'],
        'Land cover': [var for var in ds.data_vars if 'lc_frac' in var]
    }
    
    logger.info("\nVariable Categories:")
    for category, vars in categories.items():
        present = [v for v in vars if v in ds.data_vars]
        if present:
            logger.info(f"  {category}: {len(present)} variables")
            for var in present[:5]:  # Show first 5
                logger.info(f"    - {var}")
            if len(present) > 5:
                logger.info(f"    ... and {len(present)-5} more")
    
    # Check for missing data
    logger.info("\nData Completeness:")
    for var in list(ds.data_vars)[:10]:  # Check first 10 variables
        data = ds[var].values
        nan_count = np.isnan(data).sum()
        nan_pct = (nan_count / data.size) * 100
        if nan_pct > 0:
            logger.info(f"  {var}: {nan_pct:.1f}% missing")
        else:
            logger.info(f"  {var}: ✓ Complete")
    
    if len(ds.data_vars) > 10:
        logger.info(f"  ... and {len(ds.data_vars)-10} more variables")
    
    # Memory usage
    total_size = sum(ds[var].nbytes for var in ds.data_vars) / (1024**3)  # GB
    logger.info(f"\nEstimated memory usage: {total_size:.2f} GB")


def main():
    """Create unified dataset from all processed components."""
    logger.info("="*60)
    logger.info("STEP 7: CREATE UNIFIED DATASET")
    logger.info("="*60)
    
    # Load configuration
    config = load_config()
    
    # Define input paths
    input_paths = {
        'master_grid': Path(config['paths']['master_grid']),
        'fwi': Path(config['paths']['processed_fwi']),
        'era5_land': Path(config['paths']['processed_era5_land']),
        'era5_atmos': Path(config['paths']['processed_era5_atmos']),
        'uerra': Path(config['paths']['processed_uerra']),
        'landcover': Path("data/interim/portugal/06_landcover_fractions_1km.nc")
    }
    
    # Load master grid
    logger.info("\nLoading master grid...")
    master_grid = load_dataset(input_paths['master_grid'], "Master Grid")
    if master_grid is None:
        logger.error("Cannot proceed without master grid")
        return None
    
    # Load all datasets
    logger.info("\nLoading processed datasets...")
    datasets = {}
    
    # Load each dataset
    for name, path in input_paths.items():
        if name == 'master_grid':
            continue
        
        ds = load_dataset(path, name.upper().replace('_', ' '))
        if ds is not None:
            # Standardize coordinates
            ds = standardize_coordinates(ds, master_grid)
            datasets[name] = ds
    
    # Check if we have at least some data
    if not datasets:
        logger.error("No datasets could be loaded!")
        return None
    
    logger.info(f"\nSuccessfully loaded {len(datasets)} datasets")
    
    # Define hierarchy (UERRA > ERA5-Land > ERA5 Atmospheric)
    hierarchy = ['era5_atmos', 'era5_land', 'uerra']
    
    # Merge datasets hierarchically
    unified = merge_hierarchical(datasets, hierarchy)
    
    # Add FWI data (highest priority for FWI variables)
    if 'fwi' in datasets:
        logger.info("\nAdding FWI data (highest priority for FWI indices)...")
        fwi_vars = ['fwi', 'ffmc', 'dmc', 'dc', 'isi', 'bui']
        for var in fwi_vars:
            if var in datasets['fwi']:
                unified[var] = datasets['fwi'][var]
                logger.info(f"  ✓ Added {var}")
    
    # Add land cover data (non-conflicting)
    if 'landcover' in datasets:
        logger.info("\nAdding land cover fractions...")
        lc_vars = [v for v in datasets['landcover'].data_vars if 'lc_frac' in v]
        for var in lc_vars:
            unified[var] = datasets['landcover'][var]
        logger.info(f"  ✓ Added {len(lc_vars)} land cover variables")
    
    # Calculate derived variables
    unified = calculate_derived_variables(unified)
    
    # Add global metadata
    unified = add_global_metadata(unified, config)
    
    # Perform quality check
    quality_check(unified)
    
    # Save the unified dataset
    output_path = Path("data/interim/portugal/07_fwi_unified_unnormalized.nc")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nSaving unified dataset to: {output_path}")
    
    # Set up encoding for efficient storage
    encoding = {}
    for var in unified.data_vars:
        encoding[var] = {
            'zlib': True,
            'complevel': 4,
            'dtype': 'float32'
        }
    
    # Save
    unified.to_netcdf(output_path, encoding=encoding)
    logger.info("  ✓ Saved successfully")
    
    # Final verification
    if output_path.exists():
        file_size_mb = output_path.stat().st_size / (1024**2)
        logger.info(f"\n✓ Output file created: {output_path}")
        logger.info(f"  File size: {file_size_mb:.1f} MB")
        logger.info(f"  Total variables: {len(unified.data_vars)}")
        logger.info(f"  Dimensions: {dict(unified.sizes)}")
    
    return output_path


if __name__ == "__main__":
    output = main()
    if output:
        print(f"\n✅ SUCCESS: Unified dataset created at {output}")
        print("\n" + "="*60)
        print("PREPROCESSING PIPELINE COMPLETE")
        print("="*60)
    else:
        print("\n❌ FAILED: Script did not complete successfully")
        sys.exit(1)