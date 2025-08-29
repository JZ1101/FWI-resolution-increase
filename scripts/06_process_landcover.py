#!/usr/bin/env python3
"""
Step 6: Process ESA WorldCover land cover data with memory-efficient chunking.
Calculate fractional coverage for each land cover class at 1km resolution.
Uses windowed reading to avoid memory issues with large GeoTIFF files.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import xarray as xr
import numpy as np
import pandas as pd
import logging
from glob import glob
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.warp import reproject, Resampling as WarpResampling
from rasterio.crs import CRS
import zipfile
import shutil
import warnings
from affine import Affine

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


def process_tile_chunked(tif_path, master_grid, lc_classes, chunk_size=2048):
    """
    Process a single GeoTIFF tile in chunks to avoid memory issues.
    
    Args:
        tif_path: Path to the GeoTIFF file
        master_grid: Target 1km grid
        lc_classes: Dictionary of land cover classes
        chunk_size: Size of chunks to process (default 2048x2048)
    
    Returns:
        Dictionary of accumulated fractional coverage arrays
    """
    logger.info(f"  Opening {Path(tif_path).name} for chunked processing...")
    
    # Initialize output arrays for each land cover class
    output_shape = (len(master_grid.latitude), len(master_grid.longitude))
    fractions = {}
    counts = {}  # Track how many chunks contributed to each cell
    
    for class_name in lc_classes.keys():
        var_name = f'lc_frac_{class_name}'
        fractions[var_name] = np.zeros(output_shape, dtype=np.float32)
        counts[var_name] = np.zeros(output_shape, dtype=np.float32)
    
    # Get master grid bounds and resolution
    master_lons = master_grid.longitude.values
    master_lats = master_grid.latitude.values
    lon_res = abs(master_lons[1] - master_lons[0])
    lat_res = abs(master_lats[1] - master_lats[0])
    
    # Master grid bounds
    master_west = float(master_lons.min() - lon_res/2)
    master_east = float(master_lons.max() + lon_res/2)
    master_south = float(master_lats.min() - lat_res/2)
    master_north = float(master_lats.max() + lat_res/2)
    
    with rasterio.open(tif_path) as src:
        # Get source metadata
        src_height = src.height
        src_width = src.width
        src_crs = src.crs or CRS.from_epsg(4326)
        
        logger.info(f"    Source: {src_width}x{src_height} pixels")
        logger.info(f"    Processing in {chunk_size}x{chunk_size} chunks...")
        
        # Calculate number of chunks
        n_chunks_y = (src_height + chunk_size - 1) // chunk_size
        n_chunks_x = (src_width + chunk_size - 1) // chunk_size
        total_chunks = n_chunks_y * n_chunks_x
        logger.info(f"    Total chunks to process: {total_chunks}")
        
        chunk_count = 0
        
        # Process in chunks
        for row_off in range(0, src_height, chunk_size):
            for col_off in range(0, src_width, chunk_size):
                chunk_count += 1
                
                # Calculate actual chunk size (handle edges)
                actual_height = min(chunk_size, src_height - row_off)
                actual_width = min(chunk_size, src_width - col_off)
                
                # Create window for this chunk
                window = Window(col_off, row_off, actual_width, actual_height)
                
                # Read chunk data
                chunk_data = src.read(1, window=window)
                
                # Get chunk transform
                chunk_transform = src.window_transform(window)
                
                # Calculate chunk bounds in geographic coordinates
                chunk_west, chunk_north = chunk_transform * (0, 0)
                chunk_east, chunk_south = chunk_transform * (actual_width, actual_height)
                
                # Check if chunk overlaps with master grid
                if (chunk_east < master_west or chunk_west > master_east or
                    chunk_north < master_south or chunk_south > master_north):
                    continue  # Skip chunks outside our area of interest
                
                # Process each land cover class
                for class_name, class_info in lc_classes.items():
                    class_value = class_info['value']
                    var_name = f'lc_frac_{class_name}'
                    
                    # Create binary mask for this class
                    binary_mask = (chunk_data == class_value).astype(np.float32)
                    
                    # Calculate target window in master grid coordinates
                    # Find which master grid cells this chunk overlaps
                    lon_idx_start = max(0, int((chunk_west - master_west) / lon_res))
                    lon_idx_end = min(len(master_lons), int((chunk_east - master_west) / lon_res) + 1)
                    lat_idx_start = max(0, int((master_north - chunk_north) / lat_res))
                    lat_idx_end = min(len(master_lats), int((master_north - chunk_south) / lat_res) + 1)
                    
                    if lon_idx_end <= lon_idx_start or lat_idx_end <= lat_idx_start:
                        continue  # No overlap
                    
                    # Create destination array for this chunk's contribution
                    dst_height = lat_idx_end - lat_idx_start
                    dst_width = lon_idx_end - lon_idx_start
                    dst_array = np.zeros((dst_height, dst_width), dtype=np.float32)
                    
                    # Calculate destination transform
                    dst_west = master_west + lon_idx_start * lon_res
                    dst_north = master_north - lat_idx_start * lat_res
                    dst_transform = Affine(lon_res, 0, dst_west, 0, -lat_res, dst_north)
                    
                    # Reproject binary mask to destination grid using average resampling
                    reproject(
                        source=binary_mask,
                        destination=dst_array,
                        src_transform=chunk_transform,
                        src_crs=src_crs,
                        dst_transform=dst_transform,
                        dst_crs=CRS.from_epsg(4326),
                        resampling=WarpResampling.average
                    )
                    
                    # Add to accumulated fractions
                    fractions[var_name][lat_idx_start:lat_idx_end, lon_idx_start:lon_idx_end] += dst_array
                    counts[var_name][lat_idx_start:lat_idx_end, lon_idx_start:lon_idx_end] += (dst_array > 0).astype(np.float32)
                
                # Log progress every 10% of chunks
                if chunk_count % max(1, total_chunks // 10) == 0:
                    progress = (chunk_count / total_chunks) * 100
                    logger.info(f"      Progress: {progress:.1f}% ({chunk_count}/{total_chunks} chunks)")
    
    # Average the accumulated fractions by the counts (for overlapping areas)
    for var_name in fractions.keys():
        mask = counts[var_name] > 0
        fractions[var_name][mask] = fractions[var_name][mask] / counts[var_name][mask]
        # Ensure values are between 0 and 1
        fractions[var_name] = np.clip(fractions[var_name], 0, 1)
    
    logger.info(f"    ✓ Tile processed successfully")
    return fractions


def main():
    """Process ESA WorldCover land cover data using memory-efficient chunked processing."""
    logger.info("="*60)
    logger.info("STEP 6: PROCESS ESA LAND COVER DATA (CHUNKED)")
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
    
    # Find land cover files
    lc_dir = Path(config['paths']['esa_landcover'])
    
    # Check for ZIP files that need extraction
    zip_files = sorted(glob(str(lc_dir / "*.zip")))
    temp_dir = None
    
    if zip_files:
        logger.info(f"Found {len(zip_files)} ZIP files to extract")
        
        # Create temporary directory for extraction
        temp_dir = Path("data/interim/portugal/temp_landcover")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        for zip_file in zip_files:
            logger.info(f"  Extracting {Path(zip_file).name}...")
            try:
                with zipfile.ZipFile(zip_file, 'r') as zf:
                    zf.extractall(temp_dir)
                logger.info("    ✓ Extracted successfully")
            except Exception as e:
                logger.error(f"    Failed to extract: {e}")
                return None
        
        # Look for GeoTIFF files in the extracted directory
        tif_files = sorted(glob(str(temp_dir / "**/*.tif"), recursive=True))
    else:
        # Look for GeoTIFF files directly
        tif_files = sorted(glob(str(lc_dir / "*.tif"))) + sorted(glob(str(lc_dir / "*.tiff")))
    
    if not tif_files:
        logger.error("No GeoTIFF files found. Cannot proceed without source data.")
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)
        return None
    
    logger.info(f"Found {len(tif_files)} GeoTIFF files total")
    
    # Filter to only Portugal-relevant tiles
    portugal_tiles = []
    for tif_file in tif_files:
        filename = Path(tif_file).name
        # ESA tiles covering Portugal: N36-N42, W006-W009
        if any(tile in filename for tile in ['N36W006', 'N36W009', 'N39W006', 'N39W009', 'N42W006', 'N42W009']):
            portugal_tiles.append(tif_file)
    
    if not portugal_tiles:
        logger.error("No Portugal-relevant tiles found")
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)
        return None
    
    logger.info(f"  Selected {len(portugal_tiles)} Portugal-relevant tiles")
    
    # Define ESA WorldCover land cover classes
    lc_classes = {
        'tree_cover': {'value': 10, 'description': 'Tree cover'},
        'shrubland': {'value': 20, 'description': 'Shrubland'},
        'grassland': {'value': 30, 'description': 'Grassland'},
        'cropland': {'value': 40, 'description': 'Cropland'},
        'built_up': {'value': 50, 'description': 'Built-up'},
        'bare_sparse': {'value': 60, 'description': 'Bare/sparse vegetation'},
        'snow_ice': {'value': 70, 'description': 'Snow and ice'},
        'water': {'value': 80, 'description': 'Permanent water bodies'},
        'wetland': {'value': 90, 'description': 'Herbaceous wetland'},
        'mangroves': {'value': 95, 'description': 'Mangroves'},
        'moss_lichen': {'value': 100, 'description': 'Moss and lichen'}
    }
    
    # Process each tile with chunking
    logger.info("\nProcessing land cover tiles with chunking...")
    
    # Initialize accumulated fractions
    output_shape = (len(master_grid.latitude), len(master_grid.longitude))
    accumulated_fractions = {}
    for class_name in lc_classes.keys():
        var_name = f'lc_frac_{class_name}'
        accumulated_fractions[var_name] = np.zeros(output_shape, dtype=np.float32)
    
    tiles_processed = 0
    
    for tile_idx, tif_file in enumerate(portugal_tiles):
        logger.info(f"\nProcessing tile {tile_idx+1}/{len(portugal_tiles)}: {Path(tif_file).name}")
        
        try:
            # Process this tile with chunking
            tile_fractions = process_tile_chunked(tif_file, master_grid, lc_classes, chunk_size=2048)
            
            # Accumulate results (take maximum for overlapping areas)
            for var_name, fraction_data in tile_fractions.items():
                accumulated_fractions[var_name] = np.maximum(
                    accumulated_fractions[var_name],
                    fraction_data
                )
            
            tiles_processed += 1
            
        except Exception as e:
            logger.warning(f"  Failed to process tile: {e}")
            continue
    
    if tiles_processed == 0:
        logger.error("No tiles were successfully processed")
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)
        return None
    
    logger.info(f"\n✓ Processed {tiles_processed} tiles successfully")
    
    # Create output dataset
    logger.info("\nCreating output dataset...")
    
    # Convert accumulated fractions to xarray format
    data_vars = {}
    for var_name, fraction_data in accumulated_fractions.items():
        # Replace NaN with 0 (no coverage)
        fraction_data = np.nan_to_num(fraction_data, nan=0.0)
        data_vars[var_name] = (('latitude', 'longitude'), fraction_data)
    
    ds_landcover = xr.Dataset(
        data_vars,
        coords={
            'latitude': master_grid.latitude,
            'longitude': master_grid.longitude
        }
    )
    
    # Add metadata
    ds_landcover.attrs = {
        'title': 'ESA WorldCover Land Cover Fractions for Portugal',
        'source': 'ESA WorldCover 10m v100 (2020)',
        'method': 'Chunked binary mask processing with average resampling',
        'chunk_size': '2048x2048 pixels',
        'original_resolution': '10m',
        'target_resolution': '1km (0.01 degrees)',
        'processing_date': pd.Timestamp.now().isoformat(),
        'region': 'Portugal Continental',
        'lat_bounds': f"{geo['lat_min']} to {geo['lat_max']}",
        'lon_bounds': f"{geo['lon_min']} to {geo['lon_max']}",
        'tiles_processed': tiles_processed,
        'description': 'Fractional coverage of each land cover class per 1km grid cell'
    }
    
    # Add variable attributes
    for class_name, class_info in lc_classes.items():
        var_name = f'lc_frac_{class_name}'
        if var_name in ds_landcover:
            ds_landcover[var_name].attrs = {
                'long_name': f'Fraction of {class_info["description"]}',
                'units': 'fraction',
                'valid_range': [0.0, 1.0],
                'class_value': class_info['value'],
                'description': f'Fractional coverage of {class_info["description"]} in 1km grid cell',
                'method': 'Average of binary mask from 10m to 1km using chunked processing'
            }
    
    # Save output
    output_path = Path("data/interim/portugal/06_landcover_fractions_1km.nc")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nSaving to: {output_path}")
    
    # Compression settings
    encoding = {}
    for var in ds_landcover.data_vars:
        encoding[var] = {
            'zlib': True,
            'complevel': 4,
            'dtype': 'float32'
        }
    
    ds_landcover.to_netcdf(output_path, encoding=encoding)
    logger.info("  ✓ Saved successfully")
    
    # Quality check
    logger.info("\n" + "="*60)
    logger.info("QUALITY CHECK")
    logger.info("="*60)
    
    logger.info("Land cover fraction statistics:")
    total_coverage = np.zeros((len(master_grid.latitude), len(master_grid.longitude)))
    
    for var in ds_landcover.data_vars:
        values = ds_landcover[var].values
        total_coverage += values
        non_zero = np.sum(values > 0.001)  # Count cells with >0.1% coverage
        logger.info(f"\n{var}:")
        logger.info(f"  Min: {np.min(values):.3f}")
        logger.info(f"  Max: {np.max(values):.3f}")
        logger.info(f"  Mean: {np.mean(values):.3f}")
        logger.info(f"  Std: {np.std(values):.3f}")
        logger.info(f"  Cells with >0.1% coverage: {non_zero:,} / {values.size:,}")
    
    # Check total coverage
    logger.info("\nTotal coverage check:")
    logger.info(f"  Mean total: {np.mean(total_coverage):.3f}")
    logger.info(f"  Max total: {np.max(total_coverage):.3f}")
    logger.info(f"  Cells with coverage: {np.sum(total_coverage > 0):,}")
    logger.info(f"  Cells fully covered (>0.95): {np.sum(total_coverage > 0.95):,}")
    
    # Final verification
    if output_path.exists():
        file_size_mb = output_path.stat().st_size / (1024**2)
        logger.info(f"\n✓ Output file created: {output_path}")
        logger.info(f"  File size: {file_size_mb:.1f} MB")
        logger.info(f"  Variables: {len(list(ds_landcover.data_vars))} land cover fractions")
    
    # Clean up temporary directory
    if temp_dir and temp_dir.exists():
        logger.info(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)
        logger.info("  ✓ Cleanup complete")
    
    return output_path


if __name__ == "__main__":
    output = main()
    if output:
        print(f"\n✅ SUCCESS: Output saved to {output}")
        print("\nReady for next step: 07_create_unified_dataset.py")
    else:
        print("\n❌ FAILED: Script did not complete successfully")
        sys.exit(1)