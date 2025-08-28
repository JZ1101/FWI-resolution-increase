#!/usr/bin/env python3
"""
Step 4: Process ESA Land Cover 10m data
- Load ESA WorldCover GeoTIFFs covering Portugal  
- Aggregate 10m pixels to 1km grid by calculating fractional coverage
- Create variables for each land cover class fraction
- Save to data/01_processed/processed_landcover_1km.nc
"""

import xarray as xr
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ESA WorldCover class definitions
LANDCOVER_CLASSES = {
    10: 'lc_frac_10_tree_cover',
    20: 'lc_frac_20_shrubland', 
    30: 'lc_frac_30_grassland',
    40: 'lc_frac_40_cropland',
    50: 'lc_frac_50_built_up',
    60: 'lc_frac_60_bare',
    70: 'lc_frac_70_snow_ice',
    80: 'lc_frac_80_water',
    90: 'lc_frac_90_herbaceous_wetland',
    95: 'lc_frac_95_mangroves',
    100: 'lc_frac_100_moss_lichen'
}

def process_esa_landcover():
    """Process ESA WorldCover data to 1km fractional coverage"""
    print("=" * 70)
    print("STEP 4: PROCESS ESA LAND COVER DATA")
    print("=" * 70)
    
    # Load master grid
    master_path = Path("data/01_processed/master_grid_1km_2010_2017.nc")
    print(f"\nLoading master grid from: {master_path}")
    master = xr.open_dataset(master_path)
    print(f"  Master grid: {len(master.latitude)} x {len(master.longitude)} (1km)")
    print(f"  Bounds: lat=[{master.latitude.min().values:.2f}, {master.latitude.max().values:.2f}]")
    print(f"  Bounds: lon=[{master.longitude.min().values:.2f}, {master.longitude.max().values:.2f}]")
    
    # Find ESA tiles covering Portugal
    esa_dir = Path("data/00_raw/portugal_2010_2017/ESA_land_cover/sample_extract")
    
    # Portugal tiles: N36-N42, W006-W012
    portugal_tiles = []
    for lat in [36, 39, 42]:
        for lon in [6, 9, 12]:
            pattern = f"*N{lat:02d}W{lon:03d}*.tif"
            tiles = list(esa_dir.glob(pattern))
            portugal_tiles.extend(tiles)
    
    portugal_tiles = sorted(set(portugal_tiles))
    print(f"\nFound {len(portugal_tiles)} ESA WorldCover tiles for Portugal:")
    for tile in portugal_tiles:
        size_mb = tile.stat().st_size / (1024**2)
        print(f"  - {tile.name} ({size_mb:.1f} MB)")
    
    if not portugal_tiles:
        print("ERROR: No ESA tiles found for Portugal region!")
        return None
    
    # Initialize fractional coverage arrays
    print("\nInitializing fractional coverage arrays...")
    fraction_arrays = {}
    for class_val, class_name in LANDCOVER_CLASSES.items():
        fraction_arrays[class_name] = np.zeros(
            (len(master.latitude), len(master.longitude)), 
            dtype=np.float32
        )
    
    # Also track total pixel count for normalization
    pixel_counts = np.zeros((len(master.latitude), len(master.longitude)), dtype=np.float32)
    
    # Process each tile
    print("\n" + "=" * 60)
    print("Processing tiles...")
    print("=" * 60)
    
    for tile_idx, tile_path in enumerate(portugal_tiles):
        print(f"\n[{tile_idx+1}/{len(portugal_tiles)}] Processing: {tile_path.name}")
        
        try:
            with rasterio.open(tile_path) as src:
                # Get tile bounds
                bounds = src.bounds
                print(f"  Bounds: lat=[{bounds.bottom:.2f}, {bounds.top:.2f}], lon=[{bounds.left:.2f}, {bounds.right:.2f}]")
                
                # Check if tile overlaps with master grid
                if (bounds.top < master.latitude.min().values or 
                    bounds.bottom > master.latitude.max().values or
                    bounds.right < master.longitude.min().values or
                    bounds.left > master.longitude.max().values):
                    print("  Skipping: No overlap with master grid")
                    continue
                
                # Read the land cover data
                lc_data = src.read(1)
                print(f"  Shape: {lc_data.shape} (10m resolution)")
                
                # Get unique classes in this tile
                unique_classes = np.unique(lc_data)
                valid_classes = [c for c in unique_classes if c in LANDCOVER_CLASSES]
                print(f"  Land cover classes present: {valid_classes}")
                
                # Get transform for pixel to coordinate conversion
                transform = src.transform
                
                # Process each 1km grid cell that overlaps with this tile
                for lat_idx in range(len(master.latitude)):
                    lat = master.latitude.values[lat_idx]
                    
                    # Skip if outside tile bounds
                    if lat < bounds.bottom or lat > bounds.top:
                        continue
                    
                    for lon_idx in range(len(master.longitude)):
                        lon = master.longitude.values[lon_idx]
                        
                        # Skip if outside tile bounds
                        if lon < bounds.left or lon > bounds.right:
                            continue
                        
                        # Define 1km cell bounds (approximately 0.01 degrees)
                        cell_lat_min = lat - 0.005
                        cell_lat_max = lat + 0.005
                        cell_lon_min = lon - 0.005
                        cell_lon_max = lon + 0.005
                        
                        # Convert to pixel coordinates
                        col_min, row_max = ~transform * (cell_lon_min, cell_lat_min)
                        col_max, row_min = ~transform * (cell_lon_max, cell_lat_max)
                        
                        # Convert to integers and clip to tile bounds
                        row_min = max(0, int(row_min))
                        row_max = min(lc_data.shape[0], int(row_max) + 1)
                        col_min = max(0, int(col_min))
                        col_max = min(lc_data.shape[1], int(col_max) + 1)
                        
                        # Skip if no valid pixels
                        if row_min >= row_max or col_min >= col_max:
                            continue
                        
                        # Extract the subset for this 1km cell
                        cell_data = lc_data[row_min:row_max, col_min:col_max]
                        
                        # Count pixels for each class
                        total_pixels = cell_data.size
                        if total_pixels == 0:
                            continue
                        
                        pixel_counts[lat_idx, lon_idx] += total_pixels
                        
                        # Calculate fractions for each class
                        for class_val in valid_classes:
                            class_pixels = np.sum(cell_data == class_val)
                            if class_pixels > 0:
                                class_name = LANDCOVER_CLASSES[class_val]
                                fraction_arrays[class_name][lat_idx, lon_idx] += class_pixels
                
                print(f"  ✓ Processed successfully")
                
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Normalize to get fractions
    print("\n" + "=" * 60)
    print("Normalizing to fractional coverage...")
    print("=" * 60)
    
    for class_name in fraction_arrays:
        # Avoid division by zero
        mask = pixel_counts > 0
        fraction_arrays[class_name][mask] /= pixel_counts[mask]
        
        # Set areas with no data to NaN
        fraction_arrays[class_name][pixel_counts == 0] = np.nan
        
        # Print statistics
        valid_data = fraction_arrays[class_name][~np.isnan(fraction_arrays[class_name])]
        if len(valid_data) > 0:
            coverage_pct = (len(valid_data) / fraction_arrays[class_name].size) * 100
            mean_frac = np.mean(valid_data) * 100
            max_frac = np.max(valid_data) * 100
            print(f"  {class_name}:")
            print(f"    Coverage: {coverage_pct:.1f}% of grid cells")
            print(f"    Mean fraction: {mean_frac:.1f}%")
            print(f"    Max fraction: {max_frac:.1f}%")
    
    # Create xarray dataset
    print("\n" + "=" * 60)
    print("Creating output dataset...")
    print("=" * 60)
    
    data_vars = {}
    for class_name, frac_array in fraction_arrays.items():
        # Only include classes that have data
        if np.any(~np.isnan(frac_array)):
            data_vars[class_name] = xr.DataArray(
                frac_array,
                dims=['latitude', 'longitude'],
                coords={'latitude': master.latitude, 'longitude': master.longitude},
                attrs={
                    'long_name': f'Fractional coverage of {class_name.replace("lc_frac_", "").replace("_", " ")}',
                    'units': 'fraction (0-1)',
                    'source': 'ESA WorldCover 10m 2020'
                }
            )
    
    ds = xr.Dataset(data_vars)
    
    # Add global attributes
    ds.attrs['title'] = 'ESA WorldCover Land Cover Fractions - Portugal'
    ds.attrs['description'] = '10m land cover aggregated to 1km fractional coverage'
    ds.attrs['source'] = 'ESA WorldCover 10m 2020 v100'
    ds.attrs['processing'] = 'Aggregated from 10m to 1km by calculating fractional coverage per class'
    ds.attrs['classes_included'] = ', '.join(sorted(data_vars.keys()))
    
    # Save
    output_path = Path("data/01_processed/processed_landcover_1km.nc")
    print(f"\nSaving to: {output_path}")
    
    encoding = {var: {'zlib': True, 'complevel': 4} for var in ds.data_vars}
    ds.to_netcdf(output_path, encoding=encoding)
    
    file_size_mb = output_path.stat().st_size / (1024**2)
    print(f"  Saved! File size: {file_size_mb:.1f} MB")
    
    print(f"\nFinal dataset summary:")
    print(f"  Variables: {len(ds.data_vars)} land cover fractions")
    print(f"  Grid: {len(ds.latitude)} x {len(ds.longitude)} (1km)")
    print(f"  Variable names:")
    for var in sorted(ds.data_vars):
        print(f"    - {var}")
    
    return output_path

def main():
    """Main workflow for Step 4"""
    try:
        import rasterio
    except ImportError:
        print("ERROR: rasterio not installed!")
        print("Installing rasterio...")
        import subprocess
        subprocess.run(["uv", "add", "rasterio"], check=True)
        print("Please run the script again.")
        return
    
    output_path = process_esa_landcover()
    
    if output_path and output_path.exists():
        print("\n" + "=" * 70)
        print("DELIVERABLE VERIFICATION")
        print("=" * 70)
        print("\nRun this command to verify:")
        print(f'ncdump -h {output_path} | grep "float\\|double"')
        print("\nExpected: Multiple lc_frac_... variables")

if __name__ == "__main__":
    main()