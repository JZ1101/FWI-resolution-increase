#!/usr/bin/env python3
"""
Step 5: Process UERRA high-resolution reanalysis data.
Uses Dask chunking to handle large file sizes efficiently.
Properly handles UERRA's Lambert Conformal Conic projection.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import xarray as xr
import numpy as np
import pandas as pd
import logging
from glob import glob
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

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


def regrid_uerra_to_latlon(ds_portugal, target_grid):
    """
    Regrid UERRA data from Lambert Conformal Conic to regular lat/lon grid.
    Uses scipy.interpolate.griddata for irregular grid interpolation.
    """
    from scipy.interpolate import griddata
    
    logger.info("  Converting UERRA projection to regular lat/lon grid...")
    
    # Get the 2D lat/lon arrays from UERRA
    uerra_lat = ds_portugal.latitude.values
    uerra_lon = ds_portugal.longitude.values
    
    # Create target grid meshgrid
    target_lon_mesh, target_lat_mesh = np.meshgrid(
        target_grid.longitude.values,
        target_grid.latitude.values
    )
    
    # Initialize output arrays
    output_shape = (len(ds_portugal.time), len(target_grid.latitude), len(target_grid.longitude))
    regridded_data = {}
    
    # Process each variable
    for var in ds_portugal.data_vars:
        logger.info(f"    Regridding {var}...")
        
        # Initialize output array
        var_output = np.full(output_shape, np.nan, dtype=np.float32)
        
        # Process in time chunks for memory efficiency
        chunk_size = 50
        n_times = len(ds_portugal.time)
        
        for t_start in range(0, n_times, chunk_size):
            t_end = min(t_start + chunk_size, n_times)
            logger.info(f"      Processing time steps {t_start+1}-{t_end}/{n_times}")
            
            for t_idx in range(t_start, t_end):
                # Get data for this time step
                data_slice = ds_portugal[var].isel(time=t_idx).values
                
                # Flatten arrays for griddata
                points = np.column_stack((
                    uerra_lon.ravel(),
                    uerra_lat.ravel()
                ))
                values = data_slice.ravel()
                
                # Remove NaN points
                valid_mask = ~(np.isnan(points[:, 0]) | 
                              np.isnan(points[:, 1]) | 
                              np.isnan(values))
                
                if valid_mask.sum() > 10:  # Need at least 10 points
                    try:
                        # Interpolate to target grid
                        interpolated = griddata(
                            points[valid_mask],
                            values[valid_mask],
                            (target_lon_mesh, target_lat_mesh),
                            method='linear',
                            fill_value=np.nan
                        )
                        var_output[t_idx] = interpolated
                    except Exception as e:
                        logger.warning(f"        Interpolation failed for time {t_idx}: {e}")
        
        regridded_data[var] = (('time', 'latitude', 'longitude'), var_output)
    
    # Create new dataset with regridded data
    ds_regridded = xr.Dataset(
        regridded_data,
        coords={
            'time': ds_portugal.time,
            'latitude': target_grid.latitude,
            'longitude': target_grid.longitude
        }
    )
    
    return ds_regridded


def main():
    """Process UERRA data with Dask chunking for memory efficiency."""
    logger.info("="*60)
    logger.info("STEP 5: PROCESS UERRA DATA (FULL VERSION)")
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
    
    # Find UERRA files
    uerra_dir = Path(config['paths']['uerra'])
    uerra_files = sorted(glob(str(uerra_dir / "*.nc")))
    
    if not uerra_files:
        logger.error(f"No UERRA NetCDF files found in {uerra_dir}")
        return None
    
    logger.info(f"\nFound {len(uerra_files)} UERRA files:")
    total_size_gb = 0
    for f in uerra_files:
        file_size_gb = Path(f).stat().st_size / (1024**3)
        total_size_gb += file_size_gb
        logger.info(f"  - {Path(f).name} ({file_size_gb:.2f} GB)")
    logger.info(f"  Total size: {total_size_gb:.2f} GB")
    
    # Load UERRA data with Dask chunking
    logger.info("\nLoading UERRA data with Dask chunking...")
    
    # Define chunk sizes for memory efficiency
    chunks = {
        'time': 100,  # Process 100 time steps at a time
        'y': 200,     # Chunk spatial dimensions
        'x': 200
    }
    
    try:
        # Open all files with chunking
        ds = xr.open_mfdataset(
            uerra_files,
            combine='by_coords',
            parallel=False,
            chunks=chunks,
            engine='netcdf4'
        )
        
        logger.info(f"  Loaded dataset with Dask chunks")
        logger.info(f"  Original dimensions: {dict(ds.sizes)}")
        logger.info(f"  Variables: {list(ds.data_vars)}")
        logger.info(f"  Coordinates: {list(ds.coords)}")
        
    except Exception as e:
        logger.error(f"Failed to load UERRA files: {e}")
        return None
    
    # Standardize dimension names
    logger.info("\nStandardizing names...")
    
    if 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})
        logger.info("  ✓ Renamed: valid_time → time")
    
    # Handle time duplicates
    logger.info("\nChecking for duplicate times...")
    unique_times, unique_indices = np.unique(ds.time.values, return_index=True)
    
    if len(unique_times) < len(ds.time):
        logger.info(f"  Removing {len(ds.time) - len(unique_times)} duplicate time entries")
        ds = ds.isel(time=unique_indices)
    else:
        logger.info("  No duplicates found")
    
    ds = ds.sortby('time')
    logger.info(f"  Final time steps: {len(ds.time)}")
    
    # Check for required variables
    logger.info("\nChecking UERRA variables...")
    required_vars = ['si10', 'r2', 't2m']
    found_vars = [v for v in required_vars if v in ds.data_vars]
    
    if len(found_vars) != 3:
        logger.warning(f"  Found only {len(found_vars)}/3 required variables: {found_vars}")
        logger.info(f"  Available variables: {list(ds.data_vars)}")
    else:
        logger.info(f"  ✓ All required variables present: {found_vars}")
    
    # Keep only required variables
    ds = ds[found_vars]
    
    # Subset to Portugal region using 2D lat/lon
    logger.info("\nSubsetting to Portugal region...")
    
    if 'latitude' in ds.coords and 'longitude' in ds.coords:
        # UERRA uses 2D lat/lon arrays
        lat_2d = ds.latitude.values
        lon_2d = ds.longitude.values
        
        # Convert longitude to -180 to 180 if needed
        if lon_2d.max() > 180:
            logger.info("  Converting longitude from 0-360 to -180-180...")
            lon_2d = np.where(lon_2d > 180, lon_2d - 360, lon_2d)
            ds = ds.assign_coords(longitude=(ds.longitude.dims, lon_2d))
        
        # Find Portugal bounds in the 2D arrays
        portugal_mask = (
            (lat_2d >= geo['lat_min']) & 
            (lat_2d <= geo['lat_max']) & 
            (lon_2d >= geo['lon_min']) & 
            (lon_2d <= geo['lon_max'])
        )
        
        # Get bounding box
        y_indices, x_indices = np.where(portugal_mask)
        
        if len(y_indices) > 0:
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()
            
            # Add buffer
            buffer = 20
            y_min = max(0, y_min - buffer)
            y_max = min(ds.sizes['y'] - 1, y_max + buffer)
            x_min = max(0, x_min - buffer)
            x_max = min(ds.sizes['x'] - 1, x_max + buffer)
            
            # Subset data
            ds_portugal = ds.isel(
                y=slice(y_min, y_max + 1),
                x=slice(x_min, x_max + 1)
            )
            
            logger.info(f"  Original grid: y={ds.sizes['y']}, x={ds.sizes['x']}")
            logger.info(f"  Portugal subset: y={ds_portugal.sizes['y']}, x={ds_portugal.sizes['x']}")
            
            # Compute the subset (trigger Dask computation)
            logger.info("  Computing subset...")
            ds_portugal = ds_portugal.compute()
            logger.info("  ✓ Subset computed")
        else:
            logger.error("  Could not find Portugal in UERRA domain")
            return None
    else:
        logger.error("  No lat/lon coordinates found")
        return None
    
    # Regrid to 1km master grid
    logger.info("\nRegridding to 1km master grid...")
    logger.info(f"  Source: UERRA Lambert Conformal Conic (~5km)")
    logger.info(f"  Target: Regular lat/lon grid (1km)")
    
    target_grid = master_grid[['latitude', 'longitude']]
    
    # Perform regridding
    logger.info("  Note: This process may take 10-15 minutes due to the large data volume")
    ds_regridded = regrid_uerra_to_latlon(ds_portugal, target_grid)
    logger.info("  ✓ Regridding complete")
    
    # Add metadata
    logger.info("\nAdding metadata...")
    
    ds_regridded.attrs = {
        'title': 'Processed UERRA Data for Portugal',
        'source': 'UERRA-HARMONIE regional reanalysis',
        'original_resolution': '5.5km Lambert Conformal Conic',
        'target_resolution': '1km (0.01 degrees)',
        'processing_date': pd.Timestamp.now().isoformat(),
        'region': 'Portugal Continental',
        'lat_bounds': f"{geo['lat_min']} to {geo['lat_max']}",
        'lon_bounds': f"{geo['lon_min']} to {geo['lon_max']}",
        'time_coverage': f"{str(ds_regridded.time.values[0])[:19]} to {str(ds_regridded.time.values[-1])[:19]}"
    }
    
    # Variable attributes
    var_attrs = {
        'si10': {
            'long_name': '10 metre wind speed',
            'units': 'm/s',
            'description': 'Wind speed at 10m height'
        },
        'r2': {
            'long_name': '2 metre relative humidity',
            'units': '%',
            'description': 'Relative humidity at 2m height'
        },
        't2m': {
            'long_name': '2 metre temperature',
            'units': 'K',
            'description': 'Temperature at 2m height'
        }
    }
    
    for var, attrs in var_attrs.items():
        if var in ds_regridded:
            ds_regridded[var].attrs = attrs
    
    # Save output
    output_path = Path("data/interim/portugal/05_processed_uerra_1km.nc")
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
    
    return output_path


if __name__ == "__main__":
    output = main()
    if output:
        print(f"\n✅ SUCCESS: Output saved to {output}")
    else:
        print("\n❌ FAILED: Script did not complete successfully")
        sys.exit(1)