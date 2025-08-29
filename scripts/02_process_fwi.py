#!/usr/bin/env python3
"""
Step 2: Process ERA5 FWI data.
Process the single, pre-concatenated FWI file and regrid to 1km resolution.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import xarray as xr
import numpy as np
import pandas as pd
import logging

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
    """Process single ERA5 FWI file using efficient vectorized operations."""
    logger.info("="*60)
    logger.info("STEP 2: PROCESS ERA5 FWI DATA (SIMPLIFIED)")
    logger.info("="*60)
    
    # Load configuration
    config = load_config()
    geo = config['geography']
    
    # 1. Check if master grid exists
    master_grid_path = Path(config['paths']['master_grid'])
    if not master_grid_path.exists():
        logger.error(f"Master grid not found at {master_grid_path}")
        logger.error("Please run 01_create_grid.py first")
        return None
    
    # Load master grid
    logger.info("Loading master grid...")
    master_grid = xr.open_dataset(master_grid_path)
    logger.info(f"  Master grid: {len(master_grid.latitude)} x {len(master_grid.longitude)}")
    
    # 2. Load single FWI file
    logger.info("\n1. Loading single FWI file...")
    fwi_path = Path(config['paths']['era5_fwi'])
    
    if not fwi_path.exists():
        logger.error(f"FWI file not found at {fwi_path}")
        return None
    
    ds = xr.open_dataset(fwi_path)
    logger.info(f"  Loaded: {fwi_path.name}")
    logger.info(f"  Original dimensions: {dict(ds.sizes)}")
    logger.info(f"  Variables: {list(ds.data_vars)}")
    
    # 3. Standardize variable and coordinate names
    logger.info("\n2. Standardizing names...")
    
    # Rename time dimension if needed
    if 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})
        logger.info("  ✓ Renamed: valid_time → time")
    
    # Rename FWI variable if needed
    if 'fwinx' in ds.data_vars:
        ds = ds.rename({'fwinx': 'fwi'})
        logger.info("  ✓ Renamed: fwinx → fwi")
    
    # 4. Handle coordinate system (0-360 to -180-180)
    logger.info("\n3. Converting coordinates...")
    
    if ds.longitude.min() >= 0 and ds.longitude.max() > 180:
        logger.info("  Converting longitude from 0-360 to -180-180...")
        lon_180 = xr.where(ds.longitude > 180, ds.longitude - 360, ds.longitude)
        ds = ds.assign_coords(longitude=lon_180)
        ds = ds.sortby('longitude')
        logger.info("  ✓ Longitude converted")
    
    # 5. Subset to Portugal region
    logger.info("\n4. Subsetting to Portugal region...")
    
    ds_portugal = ds.sel(
        latitude=slice(geo['lat_max'], geo['lat_min']),  # Reversed for decreasing latitude
        longitude=slice(geo['lon_min'], geo['lon_max'])
    )
    
    logger.info(f"  Original grid: {dict(ds.sizes)}")
    logger.info(f"  Portugal subset: {dict(ds_portugal.sizes)}")
    
    # 6. Regrid to 1km using vectorized interpolation
    logger.info("\n5. Regridding to 1km...")
    
    # Prepare target grid
    target_grid = master_grid[['latitude', 'longitude']]
    
    logger.info(f"  Source: {len(ds_portugal.latitude)} x {len(ds_portugal.longitude)} (~25km)")
    logger.info(f"  Target: {len(target_grid.latitude)} x {len(target_grid.longitude)} (~1km)")
    
    # Ensure coordinates are in increasing order for interpolation
    if ds_portugal.latitude.values[0] > ds_portugal.latitude.values[-1]:
        ds_portugal = ds_portugal.reindex(latitude=ds_portugal.latitude[::-1])
    
    # Perform vectorized interpolation
    logger.info("  Performing interpolation...")
    ds_regridded = ds_portugal.interp_like(
        target_grid,
        method='linear',
        kwargs={'fill_value': np.nan, 'bounds_error': False}
    )
    logger.info("  ✓ Regridding complete")
    
    # 7. Add metadata
    logger.info("\n6. Adding metadata...")
    
    ds_regridded.attrs = {
        'title': 'Processed ERA5 Fire Weather Index for Portugal',
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
    if 'fwi' in ds_regridded:
        ds_regridded['fwi'].attrs = {
            'long_name': 'Fire Weather Index',
            'units': 'dimensionless',
            'valid_range': [0, 100],
            'description': 'Canadian Fire Weather Index'
        }
    
    # 8. Save output
    output_path = Path("data/interim/portugal/02_processed_fwi_1km.nc")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n7. Saving to: {output_path}")
    
    encoding = {}
    if 'fwi' in ds_regridded:
        encoding['fwi'] = {
            'zlib': True,
            'complevel': 4,
            'dtype': 'float32'
        }
    
    ds_regridded.to_netcdf(output_path, encoding=encoding)
    logger.info("  ✓ Saved successfully")
    
    # 9. Quality check
    logger.info("\n" + "="*60)
    logger.info("QUALITY CHECK")
    logger.info("="*60)
    
    if 'fwi' in ds_regridded:
        fwi_values = ds_regridded.fwi.values
        logger.info("FWI statistics:")
        logger.info(f"  Min: {np.nanmin(fwi_values):.2f}")
        logger.info(f"  Max: {np.nanmax(fwi_values):.2f}")
        logger.info(f"  Mean: {np.nanmean(fwi_values):.2f}")
        logger.info(f"  Std: {np.nanstd(fwi_values):.2f}")
        
        nan_count = np.isnan(fwi_values).sum()
        total_count = fwi_values.size
        logger.info(f"  NaN values: {nan_count:,} / {total_count:,} ({100*nan_count/total_count:.1f}%)")
    
    # 10. Final verification
    if output_path.exists():
        file_size_mb = output_path.stat().st_size / (1024**2)
        logger.info(f"\n✓ Output file created: {output_path}")
        logger.info(f"  File size: {file_size_mb:.1f} MB")
    else:
        logger.error("\n✗ Output file not created!")
        return None
    
    # Clean up
    ds.close()
    master_grid.close()
    
    return output_path


if __name__ == "__main__":
    output = main()
    if output:
        print(f"\n✅ SUCCESS: Output saved to {output}")
    else:
        print("\n❌ FAILED: Script did not complete successfully")
        sys.exit(1)