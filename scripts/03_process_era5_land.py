#!/usr/bin/env python3
"""
Step 3: Process ERA5-Land data.
Extract from ZIP files, combine yearly data, and regrid to 1km resolution.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import xarray as xr
import numpy as np
import pandas as pd
import logging
import zipfile
import tempfile
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
    """Process ERA5-Land data from ZIP files."""
    logger.info("="*60)
    logger.info("STEP 3: PROCESS ERA5-LAND DATA")
    logger.info("="*60)
    
    # Load configuration
    config = load_config()
    geo = config['geography']
    
    # Check if master grid exists
    master_grid_path = Path(config['paths']['master_grid'])
    if not master_grid_path.exists():
        logger.error(f"Master grid not found at {master_grid_path}")
        logger.error("Please run 01_create_grid.py first")
        return None
    
    # Load master grid
    logger.info("Loading master grid...")
    master_grid = xr.open_dataset(master_grid_path)
    logger.info(f"  Master grid: {len(master_grid.latitude)} x {len(master_grid.longitude)}")
    
    # Find ERA5-Land ZIP files
    era5_land_dir = Path(config['paths']['era5_land'])
    zip_files = sorted(glob(str(era5_land_dir / "*.zip")))
    
    if not zip_files:
        logger.error(f"No ZIP files found in {era5_land_dir}")
        return None
    
    logger.info(f"\nFound {len(zip_files)} ERA5-Land ZIP files:")
    for zf in zip_files:
        logger.info(f"  - {Path(zf).name}")
    
    # ERA5-Land variable mapping
    variable_mapping = {
        't2m': '2m_temperature',           # 2m temperature
        'd2m': '2m_dewpoint_temperature',  # 2m dewpoint temperature
        'u10': '10m_u_component_of_wind',  # 10m wind U component
        'v10': '10m_v_component_of_wind',  # 10m wind V component
        'tp': 'total_precipitation'        # Total precipitation
    }
    
    # Process each ZIP file
    all_datasets = []
    
    for zip_path in zip_files:
        logger.info(f"\nProcessing: {Path(zip_path).name}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract ZIP
            logger.info("  Extracting ZIP file...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(tmpdir)
            
            # Find NetCDF files
            nc_files = sorted(glob(str(Path(tmpdir) / "*.nc")))
            if not nc_files:
                logger.warning(f"  No NetCDF files found in {Path(zip_path).name}")
                continue
            
            logger.info(f"  Found {len(nc_files)} NetCDF files")
            
            # Load and combine NetCDF files for this year
            year_data = []
            for nc_file in nc_files:
                try:
                    ds = xr.open_dataset(nc_file)
                    
                    # Standardize time dimension
                    if 'valid_time' in ds.dims:
                        ds = ds.rename({'valid_time': 'time'})
                    
                    # Check for ERA5-Land variables
                    found_vars = []
                    for short_name, long_name in variable_mapping.items():
                        if long_name in ds.data_vars:
                            ds = ds.rename({long_name: short_name})
                            found_vars.append(short_name)
                        elif short_name in ds.data_vars:
                            found_vars.append(short_name)
                    
                    if found_vars:
                        # Keep only the variables we need
                        ds = ds[found_vars]
                        year_data.append(ds)
                        logger.info(f"    Loaded: {Path(nc_file).name} with variables: {found_vars}")
                    
                except Exception as e:
                    logger.warning(f"    Failed to load {Path(nc_file).name}: {e}")
                    continue
            
            # Combine data for this year
            if year_data:
                if len(year_data) > 1:
                    year_ds = xr.merge(year_data)
                else:
                    year_ds = year_data[0]
                
                all_datasets.append(year_ds)
                logger.info(f"  Combined {len(year_data)} files for this year")
    
    if not all_datasets:
        logger.error("No valid ERA5-Land data found in ZIP files")
        return None
    
    # Combine all years
    logger.info("\nCombining all years...")
    if len(all_datasets) > 1:
        ds_combined = xr.concat(all_datasets, dim='time')
    else:
        ds_combined = all_datasets[0]
    
    # Sort by time
    ds_combined = ds_combined.sortby('time')
    
    logger.info(f"  Combined dataset dimensions: {dict(ds_combined.sizes)}")
    logger.info(f"  Variables: {list(ds_combined.data_vars)}")
    logger.info(f"  Time range: {str(ds_combined.time.values[0])[:19]} to {str(ds_combined.time.values[-1])[:19]}")
    
    # Handle coordinate system
    logger.info("\nProcessing coordinates...")
    
    # Convert longitude if needed (0-360 to -180-180)
    if ds_combined.longitude.min() >= 0 and ds_combined.longitude.max() > 180:
        logger.info("  Converting longitude from 0-360 to -180-180...")
        lon_180 = xr.where(ds_combined.longitude > 180, ds_combined.longitude - 360, ds_combined.longitude)
        ds_combined = ds_combined.assign_coords(longitude=lon_180)
        ds_combined = ds_combined.sortby('longitude')
    
    # Subset to Portugal region
    logger.info("\nSubsetting to Portugal region...")
    ds_portugal = ds_combined.sel(
        latitude=slice(geo['lat_max'], geo['lat_min']),  # Reversed for decreasing latitude
        longitude=slice(geo['lon_min'], geo['lon_max'])
    )
    
    logger.info(f"  Original grid: {dict(ds_combined.sizes)}")
    logger.info(f"  Portugal subset: {dict(ds_portugal.sizes)}")
    
    # Regrid to 1km
    logger.info("\nRegridding to 1km...")
    target_grid = master_grid[['latitude', 'longitude']]
    
    logger.info(f"  Source: {len(ds_portugal.latitude)} x {len(ds_portugal.longitude)} (~11km)")
    logger.info(f"  Target: {len(target_grid.latitude)} x {len(target_grid.longitude)} (~1km)")
    
    # Ensure coordinates are in increasing order for interpolation
    if ds_portugal.latitude.values[0] > ds_portugal.latitude.values[-1]:
        ds_portugal = ds_portugal.reindex(latitude=ds_portugal.latitude[::-1])
    
    # Perform interpolation
    logger.info("  Performing interpolation...")
    ds_regridded = ds_portugal.interp_like(
        target_grid,
        method='linear',
        kwargs={'fill_value': np.nan, 'bounds_error': False}
    )
    logger.info("  ✓ Regridding complete")
    
    # Add metadata
    logger.info("\nAdding metadata...")
    
    ds_regridded.attrs = {
        'title': 'Processed ERA5-Land Data for Portugal',
        'source': 'ERA5-Land reanalysis',
        'original_resolution': '11km (0.1 degrees)',
        'target_resolution': '1km (0.01 degrees)',
        'processing_date': pd.Timestamp.now().isoformat(),
        'region': 'Portugal Continental',
        'lat_bounds': f"{geo['lat_min']} to {geo['lat_max']}",
        'lon_bounds': f"{geo['lon_min']} to {geo['lon_max']}",
        'time_coverage': f"{str(ds_regridded.time.values[0])[:19]} to {str(ds_regridded.time.values[-1])[:19]}"
    }
    
    # Variable attributes
    var_attrs = {
        't2m': {
            'long_name': '2 metre temperature',
            'units': 'K',
            'description': 'Temperature at 2m above surface'
        },
        'd2m': {
            'long_name': '2 metre dewpoint temperature',
            'units': 'K',
            'description': 'Dewpoint temperature at 2m above surface'
        },
        'u10': {
            'long_name': '10 metre U wind component',
            'units': 'm/s',
            'description': 'Eastward wind component at 10m'
        },
        'v10': {
            'long_name': '10 metre V wind component',
            'units': 'm/s',
            'description': 'Northward wind component at 10m'
        },
        'tp': {
            'long_name': 'Total precipitation',
            'units': 'm',
            'description': 'Total precipitation'
        }
    }
    
    for var, attrs in var_attrs.items():
        if var in ds_regridded:
            ds_regridded[var].attrs = attrs
    
    # Save output
    output_path = Path("data/interim/portugal/03_processed_era5_land_1km.nc")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nSaving to: {output_path}")
    
    # Compression settings
    encoding = {}
    for var in ds_regridded.data_vars:
        encoding[var] = {
            'zlib': True,
            'complevel': 4,
            'dtype': 'float32'
        }
    
    ds_regridded.to_netcdf(output_path, encoding=encoding)
    logger.info("  ✓ Saved successfully")
    
    # Quality check
    logger.info("\n" + "="*60)
    logger.info("QUALITY CHECK")
    logger.info("="*60)
    
    logger.info("Variable statistics:")
    for var in ds_regridded.data_vars:
        values = ds_regridded[var].values
        logger.info(f"\n{var}:")
        logger.info(f"  Min: {np.nanmin(values):.2f}")
        logger.info(f"  Max: {np.nanmax(values):.2f}")
        logger.info(f"  Mean: {np.nanmean(values):.2f}")
        nan_pct = 100 * np.isnan(values).sum() / values.size
        logger.info(f"  NaN: {nan_pct:.1f}%")
    
    # Final verification
    if output_path.exists():
        file_size_mb = output_path.stat().st_size / (1024**2)
        logger.info(f"\n✓ Output file created: {output_path}")
        logger.info(f"  File size: {file_size_mb:.1f} MB")
        logger.info(f"  Variables: {list(ds_regridded.data_vars)}")
    else:
        logger.error("\n✗ Output file not created!")
        return None
    
    # Clean up
    ds_combined.close()
    master_grid.close()
    
    return output_path


if __name__ == "__main__":
    output = main()
    if output:
        print(f"\n✅ SUCCESS: Output saved to {output}")
    else:
        print("\n❌ FAILED: Script did not complete successfully")
        sys.exit(1)