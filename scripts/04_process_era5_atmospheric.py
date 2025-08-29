#!/usr/bin/env python3
"""
Step 4: Process ERA5 Atmospheric data.
Handle mixed structure of ZIP archives and standalone NetCDF files.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import xarray as xr
import numpy as np
import pandas as pd
import logging
import zipfile
import shutil
from glob import glob

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


def main():
    """Process ERA5 Atmospheric data with proper ZIP extraction."""
    logger.info("="*60)
    logger.info("STEP 4: PROCESS ERA5 ATMOSPHERIC DATA")
    logger.info("="*60)
    
    # Load configuration
    config = load_config()
    geo = config['geography']
    
    # Check master grid
    master_grid_path = Path(config['paths']['master_grid'])
    if not master_grid_path.exists():
        logger.error(f"Master grid not found at {master_grid_path}")
        return None
    
    logger.info("Loading master grid...")
    master_grid = xr.open_dataset(master_grid_path)
    logger.info(f"  Master grid: {len(master_grid.latitude)} x {len(master_grid.longitude)}")
    
    # 1. Create temporary extraction directory
    temp_extract_dir = Path("data/interim/portugal/era5_atm_unzipped")
    if temp_extract_dir.exists():
        logger.info(f"Removing existing temp directory: {temp_extract_dir}")
        shutil.rmtree(temp_extract_dir)
    temp_extract_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created temp extraction directory: {temp_extract_dir}")
    
    try:
        # 2. Safe extraction loop for ZIP files
        era5_atm_dir = Path(config['paths']['era5_atmospheric'])
        zip_files = sorted(glob(str(era5_atm_dir / "era5_daily_mean_*.zip")))
        
        logger.info(f"\nFound {len(zip_files)} ZIP files to extract")
        
        for zip_path in zip_files:
            zip_name = Path(zip_path).stem
            # Extract year and month from filename (e.g., era5_daily_mean_2010_05.zip)
            parts = zip_name.split('_')
            if len(parts) >= 5:
                year_month = f"{parts[3]}_{parts[4]}"
            else:
                year_month = zip_name
            
            # Create unique subdirectory
            extract_subdir = temp_extract_dir / year_month
            extract_subdir.mkdir(exist_ok=True)
            
            logger.info(f"  Extracting {Path(zip_path).name} to {extract_subdir.name}/")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_subdir)
        
        # 3. File discovery and grouping
        logger.info("\nDiscovering and grouping files by variable...")
        
        # Initialize file lists for each variable
        file_lists = {
            't2m': [],   # Temperature from standalone files
            'tp': [],    # Precipitation from standalone files
            'u10': [],   # U wind from extracted ZIPs
            'v10': [],   # V wind from extracted ZIPs
            'd2m': []    # Dewpoint from extracted ZIPs
        }
        
        # Find standalone temperature files
        t2m_files = sorted(glob(str(era5_atm_dir / "era5_daily_max_temp_*.nc")))
        file_lists['t2m'] = t2m_files
        logger.info(f"  Found {len(t2m_files)} temperature files")
        
        # Find standalone precipitation files
        tp_files = sorted(glob(str(era5_atm_dir / "era5_daily_total_precipitation_*.nc")))
        file_lists['tp'] = tp_files
        logger.info(f"  Found {len(tp_files)} precipitation files")
        
        # Find extracted wind and dewpoint files
        for subdir in temp_extract_dir.iterdir():
            if subdir.is_dir():
                # U wind component
                u10_pattern = str(subdir / "*10m_u_component_of_wind*.nc")
                u10_files = glob(u10_pattern)
                file_lists['u10'].extend(u10_files)
                
                # V wind component
                v10_pattern = str(subdir / "*10m_v_component_of_wind*.nc")
                v10_files = glob(v10_pattern)
                file_lists['v10'].extend(v10_files)
                
                # Dewpoint temperature
                d2m_pattern = str(subdir / "*2m_dewpoint_temperature*.nc")
                d2m_files = glob(d2m_pattern)
                file_lists['d2m'].extend(d2m_files)
        
        # Sort all file lists
        for var in file_lists:
            file_lists[var] = sorted(file_lists[var])
            logger.info(f"  {var}: {len(file_lists[var])} files")
        
        # 4. Variable-by-variable consolidation
        logger.info("\nConsolidating variables...")
        
        data_arrays = {}
        
        for var_short, file_list in file_lists.items():
            if not file_list:
                logger.warning(f"  No files found for {var_short}")
                continue
            
            logger.info(f"  Processing {var_short}...")
            
            # Open all files for this variable
            try:
                # Use open_mfdataset to combine files
                ds_var = xr.open_mfdataset(
                    file_list, 
                    combine='by_coords',
                    parallel=False,
                    engine='netcdf4'
                )
                
                # Find the actual variable name in the dataset
                # ERA5 uses long names that we need to map
                var_mapping = {
                    't2m': ['t2m', '2m_temperature', 'mx2t'],
                    'tp': ['tp', 'total_precipitation'],
                    'u10': ['u10', '10m_u_component_of_wind'],
                    'v10': ['v10', '10m_v_component_of_wind'],
                    'd2m': ['d2m', '2m_dewpoint_temperature']
                }
                
                # Find which variable name is actually in the dataset
                actual_var = None
                for possible_name in var_mapping[var_short]:
                    if possible_name in ds_var.data_vars:
                        actual_var = possible_name
                        break
                
                if actual_var:
                    # Extract the data array and rename to standard name
                    data_arrays[var_short] = ds_var[actual_var].rename(var_short)
                    logger.info(f"    ✓ Loaded {var_short} (from {actual_var})")
                else:
                    logger.warning(f"    Could not find variable for {var_short}")
                    logger.warning(f"    Available vars: {list(ds_var.data_vars)}")
                
                ds_var.close()
                
            except Exception as e:
                logger.error(f"    Failed to process {var_short}: {e}")
                continue
        
        # 5. Merge and final processing
        logger.info("\nMerging variables into single dataset...")
        
        if not data_arrays:
            logger.error("No variables successfully loaded!")
            return None
        
        # Create dataset from data arrays
        ds_merged = xr.Dataset(data_arrays)
        
        # Standardize dimensions
        if 'valid_time' in ds_merged.dims:
            ds_merged = ds_merged.rename({'valid_time': 'time'})
        
        logger.info(f"  Merged dataset dimensions: {dict(ds_merged.sizes)}")
        logger.info(f"  Variables: {list(ds_merged.data_vars)}")
        
        # Remove duplicate times if any
        if 'time' in ds_merged.dims:
            _, unique_indices = np.unique(ds_merged.time.values, return_index=True)
            if len(unique_indices) < len(ds_merged.time):
                logger.info(f"  Removing {len(ds_merged.time) - len(unique_indices)} duplicate time steps")
                ds_merged = ds_merged.isel(time=unique_indices)
            ds_merged = ds_merged.sortby('time')
        
        # Handle coordinates (0-360 to -180-180)
        logger.info("\nProcessing coordinates...")
        if ds_merged.longitude.min() >= 0 and ds_merged.longitude.max() > 180:
            logger.info("  Converting longitude from 0-360 to -180-180...")
            lon_180 = xr.where(ds_merged.longitude > 180, ds_merged.longitude - 360, ds_merged.longitude)
            ds_merged = ds_merged.assign_coords(longitude=lon_180)
            ds_merged = ds_merged.sortby('longitude')
        
        # Subset to Portugal
        logger.info("\nSubsetting to Portugal region...")
        ds_portugal = ds_merged.sel(
            latitude=slice(geo['lat_max'], geo['lat_min']),
            longitude=slice(geo['lon_min'], geo['lon_max'])
        )
        
        logger.info(f"  Original grid: {dict(ds_merged.sizes)}")
        logger.info(f"  Portugal subset: {dict(ds_portugal.sizes)}")
        
        # Regrid to 1km
        logger.info("\nRegridding to 1km...")
        target_grid = master_grid[['latitude', 'longitude']]
        
        logger.info(f"  Source: {len(ds_portugal.latitude)} x {len(ds_portugal.longitude)} (~25km)")
        logger.info(f"  Target: {len(target_grid.latitude)} x {len(target_grid.longitude)} (~1km)")
        
        # Ensure coordinates are in increasing order
        if ds_portugal.latitude.values[0] > ds_portugal.latitude.values[-1]:
            ds_portugal = ds_portugal.reindex(latitude=ds_portugal.latitude[::-1])
        
        # Perform interpolation
        logger.info("  Performing interpolation...")
        # Debug: Check coordinates before interpolation
        logger.info(f"  Source longitude: {ds_portugal.longitude.values[:3]}...")
        logger.info(f"  Target longitude: {target_grid.longitude.values[:3]}...")
        
        ds_regridded = ds_portugal.interp_like(
            target_grid,
            method='linear',
            kwargs={'fill_value': np.nan, 'bounds_error': False}
        )
        
        # Debug: Check result
        logger.info(f"  Result longitude: {ds_regridded.longitude.values[:3]}...")
        logger.info(f"  NaN count in result longitude: {np.isnan(ds_regridded.longitude.values).sum()}")
        
        # Fix: Ensure coordinates are preserved
        if np.isnan(ds_regridded.longitude.values).any():
            logger.warning("  Fixing NaN longitude values...")
            ds_regridded = ds_regridded.assign_coords({
                'longitude': target_grid.longitude,
                'latitude': target_grid.latitude
            })
            logger.info("  ✓ Coordinates fixed")
        
        logger.info("  ✓ Regridding complete")
        
        # Add metadata
        logger.info("\nAdding metadata...")
        ds_regridded.attrs = {
            'title': 'Processed ERA5 Atmospheric Data for Portugal',
            'source': 'ERA5 reanalysis',
            'original_resolution': '25km (0.25 degrees)',
            'target_resolution': '1km (0.01 degrees)',
            'processing_date': pd.Timestamp.now().isoformat(),
            'region': 'Portugal Continental',
            'lat_bounds': f"{geo['lat_min']} to {geo['lat_max']}",
            'lon_bounds': f"{geo['lon_min']} to {geo['lon_max']}"
        }
        
        if 'time' in ds_regridded.dims:
            time_start = str(ds_regridded.time.values[0])[:19]
            time_end = str(ds_regridded.time.values[-1])[:19]
            ds_regridded.attrs['time_coverage'] = f"{time_start} to {time_end}"
        
        # Variable attributes
        var_attrs = {
            't2m': {'long_name': '2 metre temperature', 'units': 'K'},
            'd2m': {'long_name': '2 metre dewpoint temperature', 'units': 'K'},
            'u10': {'long_name': '10 metre U wind component', 'units': 'm/s'},
            'v10': {'long_name': '10 metre V wind component', 'units': 'm/s'},
            'tp': {'long_name': 'Total precipitation', 'units': 'm'}
        }
        
        for var, attrs in var_attrs.items():
            if var in ds_regridded:
                ds_regridded[var].attrs = attrs
        
        # 6. Save output
        output_path = Path(config['paths']['processed_era5_atmos'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nSaving to: {output_path}")
        
        encoding = {}
        for var in ds_regridded.data_vars:
            encoding[var] = {
                'zlib': True,
                'complevel': 4,
                'dtype': 'float32'
            }
        
        ds_regridded.to_netcdf(output_path, encoding=encoding)
        logger.info("  ✓ Saved successfully")
        
        # Verify output
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024**2)
            logger.info(f"\n✓ Output file created: {output_path}")
            logger.info(f"  File size: {file_size_mb:.1f} MB")
            logger.info(f"  Variables: {list(ds_regridded.data_vars)}")
        
    finally:
        # 7. Clean up temporary directory
        if temp_extract_dir.exists():
            logger.info(f"\nCleaning up temporary directory: {temp_extract_dir}")
            shutil.rmtree(temp_extract_dir)
            logger.info("  ✓ Cleanup complete")
    
    return output_path


if __name__ == "__main__":
    output = main()
    if output:
        print(f"\n✅ SUCCESS: Output saved to {output}")
    else:
        print("\n❌ FAILED: Script did not complete successfully")
        sys.exit(1)