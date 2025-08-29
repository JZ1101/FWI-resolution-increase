#!/usr/bin/env python3
"""
Step 3: Process UERRA high-resolution data
- Load UERRA NetCDF files using Dask chunking for memory management
- Handle any overlapping time entries between files
- Regrid all variables (si10, r2, t2m) to 1km master grid
- Save to data/01_processed/processed_uerra_1km.nc
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import dask.array as da
warnings.filterwarnings('ignore')

def load_master_grid():
    """Load the 1km master grid"""
    master_path = Path("data/01_processed/master_grid_1km_2010_2017.nc")
    if master_path.exists():
        print(f"Loading master grid from: {master_path}")
        ds = xr.open_dataset(master_path)
        return ds
    else:
        print(f"ERROR: Master grid not found at {master_path}")
        return None

def process_uerra_data():
    """Process UERRA high-resolution data with Dask chunking"""
    print("=" * 70)
    print("STEP 3: PROCESS UERRA HIGH-RESOLUTION DATA")
    print("=" * 70)
    
    # Define paths
    source_dir = Path("data/00_raw/portugal_2010_2017/UERRA_reanalysis_atmospheric parameters")
    output_dir = Path("data/01_processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load master grid
    print("\nLoading master grid...")
    master = load_master_grid()
    if master is None:
        print("ERROR: Cannot proceed without master grid")
        return
    print(f"  Master grid shape: lat={len(master.latitude)}, lon={len(master.longitude)}")
    print(f"  Master grid bounds: lat=[{master.latitude.min().values:.2f}, {master.latitude.max().values:.2f}]")
    print(f"  Master grid bounds: lon=[{master.longitude.min().values:.2f}, {master.longitude.max().values:.2f}]")
    
    # List all UERRA NetCDF files
    uerra_files = sorted(source_dir.glob("uerra_mescan_*.nc"))
    print(f"\nFound {len(uerra_files)} UERRA files:")
    
    total_size_gb = 0
    for file_path in uerra_files:
        size_gb = file_path.stat().st_size / (1024**3)
        total_size_gb += size_gb
        print(f"  - {file_path.name} ({size_gb:.1f} GB)")
    print(f"  Total size: {total_size_gb:.1f} GB")
    
    # Variables to process
    target_variables = ['si10', 'r2', 't2m']
    print(f"\nTarget variables to process: {target_variables}")
    
    # Load all files with Dask chunking to manage memory
    print("\n" + "=" * 60)
    print("Loading UERRA files with Dask chunking...")
    print("=" * 60)
    
    datasets = []
    for file_path in uerra_files:
        print(f"\nLoading: {file_path.name}")
        try:
            # Open with chunks to avoid loading everything into memory
            # Chunk by time to process each timestep separately
            ds = xr.open_dataset(file_path, chunks={'valid_time': 10})
            
            # Rename valid_time to time for consistency
            if 'valid_time' in ds.dims:
                ds = ds.rename({'valid_time': 'time'})
            
            # Print dataset info
            print(f"  Dimensions: {dict(ds.dims)}")
            print(f"  Variables: {list(ds.data_vars)}")
            
            if 'time' in ds.dims:
                time_range = f"{ds.time.min().values} to {ds.time.max().values}"
                print(f"  Time range: {time_range}")
                print(f"  Time steps: {len(ds.time)}")
            
            # Check that we have the expected variables
            missing_vars = [v for v in target_variables if v not in ds.data_vars]
            if missing_vars:
                print(f"  WARNING: Missing variables: {missing_vars}")
            
            datasets.append(ds)
            
        except Exception as e:
            print(f"  ERROR loading {file_path.name}: {e}")
    
    if not datasets:
        print("ERROR: No UERRA datasets loaded!")
        return None
    
    print(f"\nSuccessfully loaded {len(datasets)} UERRA files")
    
    # Concatenate all datasets along time dimension
    print("\n" + "=" * 60)
    print("Concatenating datasets along time dimension...")
    print("=" * 60)
    
    try:
        # Concatenate with Dask
        combined = xr.concat(datasets, dim='time')
        print(f"  Combined dimensions: {dict(combined.dims)}")
        
        # Sort by time
        print("  Sorting by time...")
        combined = combined.sortby('time')
        
        # Check for duplicate times
        print("  Checking for duplicate timestamps...")
        time_values = pd.to_datetime(combined.time.values)
        unique_times = np.unique(time_values)
        n_duplicates = len(time_values) - len(unique_times)
        
        if n_duplicates > 0:
            print(f"  WARNING: Found {n_duplicates} duplicate timestamps")
            print(f"  Removing duplicates (keeping first occurrence)...")
            
            # Remove duplicates
            _, unique_indices = np.unique(time_values, return_index=True)
            combined = combined.isel(time=sorted(unique_indices))
            print(f"  Shape after deduplication: {dict(combined.dims)}")
        else:
            print(f"  No duplicates found")
        
        print(f"\n  Final time range: {combined.time.min().values} to {combined.time.max().values}")
        print(f"  Final time steps: {len(combined.time)}")
        
    except Exception as e:
        print(f"ERROR concatenating datasets: {e}")
        return None
    
    # Process each variable and regrid to master grid
    print("\n" + "=" * 60)
    print("Regridding variables to 1km master grid...")
    print("=" * 60)
    
    processed_variables = {}
    
    for var_name in target_variables:
        if var_name not in combined.data_vars:
            print(f"\n  WARNING: Variable '{var_name}' not found in combined dataset")
            continue
        
        print(f"\n  Processing variable: {var_name}")
        var_data = combined[var_name]
        
        # Get original grid info
        print(f"    Original shape: {var_data.shape}")
        print(f"    Original dims: {var_data.dims}")
        
        # UERRA uses 2D lat/lon arrays (curvilinear grid)
        # We need to flatten and interpolate to regular grid
        print("    Handling curvilinear grid...")
        
        # Get the 2D lat/lon arrays
        lat_2d = combined.latitude.values
        lon_2d = combined.longitude.values
        
        # Convert longitude from 0-360 to -180 to 180 if needed
        if lon_2d.max() > 180:
            print("    Converting longitude from 0-360 to -180 to 180...")
            lon_2d = np.where(lon_2d > 180, lon_2d - 360, lon_2d)
        
        # Create bounds for subsetting
        lat_min, lat_max = master.latitude.min().values, master.latitude.max().values
        lon_min, lon_max = master.longitude.min().values, master.longitude.max().values
        
        # Find indices that fall within our domain
        mask = (lat_2d >= lat_min - 1) & (lat_2d <= lat_max + 1) & \
               (lon_2d >= lon_min - 1) & (lon_2d <= lon_max + 1)
        
        # Check if we have data in our region
        if not mask.any():
            print(f"    WARNING: No data points within target region!")
            print(f"    UERRA lat range: [{lat_2d.min():.2f}, {lat_2d.max():.2f}]")
            print(f"    UERRA lon range: [{lon_2d.min():.2f}, {lon_2d.max():.2f}]")
            continue
        
        print(f"    Found {mask.sum()} points within target region")
        
        # Process time step by time step to manage memory
        print("    Regridding to regular 1km grid (this may take a while)...")
        
        # Initialize output array
        regridded_shape = (len(combined.time), len(master.latitude), len(master.longitude))
        regridded_data = np.full(regridded_shape, np.nan, dtype=np.float32)
        
        # Process in chunks of 10 time steps
        chunk_size = 10
        n_chunks = int(np.ceil(len(combined.time) / chunk_size))
        
        from scipy.interpolate import griddata
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(combined.time))
            
            if chunk_idx % 10 == 0:
                progress = (chunk_idx / n_chunks) * 100
                print(f"      Progress: {progress:.0f}%")
            
            # Get chunk of data
            chunk_data = var_data.isel(time=slice(start_idx, end_idx))
            
            # Convert to numpy array if it's a Dask array
            if hasattr(chunk_data, 'compute'):
                chunk_values = chunk_data.compute().values
            else:
                chunk_values = chunk_data.values
            
            # Process each time step in the chunk
            for t_idx, t_local in enumerate(range(start_idx, end_idx)):
                # Get data for this time step
                data_slice = chunk_values[t_idx]
                
                # Skip if all NaN
                if np.all(np.isnan(data_slice)):
                    continue
                
                # Get valid points
                valid_mask = ~np.isnan(data_slice) & mask
                if not valid_mask.any():
                    continue
                
                # Extract valid points and coordinates
                points = np.column_stack([
                    lat_2d[valid_mask].flatten(),
                    lon_2d[valid_mask].flatten()
                ])
                values = data_slice[valid_mask].flatten()
                
                # Create target grid
                target_lats, target_lons = np.meshgrid(
                    master.latitude.values,
                    master.longitude.values,
                    indexing='ij'
                )
                target_points = np.column_stack([
                    target_lats.flatten(),
                    target_lons.flatten()
                ])
                
                # Interpolate
                try:
                    interpolated = griddata(
                        points, values, target_points,
                        method='linear', fill_value=np.nan
                    )
                    regridded_data[t_local] = interpolated.reshape(
                        len(master.latitude), len(master.longitude)
                    )
                except Exception as e:
                    # Skip this time step if interpolation fails
                    continue
        
        print(f"      Progress: 100%")
        
        # Create xarray DataArray with proper coordinates
        regridded_da = xr.DataArray(
            regridded_data,
            dims=['time', 'latitude', 'longitude'],
            coords={
                'time': combined.time,
                'latitude': master.latitude,
                'longitude': master.longitude
            },
            name=var_name
        )
        
        # Store processed variable
        processed_variables[var_name] = regridded_da
        
        # Print statistics
        valid_data = regridded_data[~np.isnan(regridded_data)]
        if len(valid_data) > 0:
            print(f"    ✓ Successfully regridded '{var_name}'")
            print(f"      Valid points: {len(valid_data):,} / {regridded_data.size:,}")
            print(f"      Range: [{valid_data.min():.2f}, {valid_data.max():.2f}]")
            print(f"      Mean: {valid_data.mean():.2f}")
    
    # Create final dataset
    if not processed_variables:
        print("\nERROR: No variables were successfully processed!")
        return None
    
    print("\n" + "=" * 60)
    print(f"Creating final dataset with {len(processed_variables)} variables...")
    print("=" * 60)
    
    # Combine all variables into a dataset
    final_dataset = xr.Dataset(processed_variables)
    
    # Add attributes
    final_dataset.attrs['title'] = 'UERRA High-Resolution Reanalysis - Portugal'
    final_dataset.attrs['description'] = 'UERRA-HARMONIE reanalysis regridded to 1km'
    final_dataset.attrs['source'] = 'UERRA regional reanalysis'
    final_dataset.attrs['processing'] = 'Concatenated, deduplicated, and regridded from curvilinear to regular 1km grid'
    final_dataset.attrs['variables_processed'] = ', '.join(sorted(processed_variables.keys()))
    final_dataset.attrs['original_resolution'] = '~5.5km (UERRA-HARMONIE)'
    final_dataset.attrs['target_resolution'] = '1km'
    
    # Print summary
    print(f"\nFinal dataset summary:")
    print(f"  Variables: {list(final_dataset.data_vars)}")
    print(f"  Dimensions: {dict(final_dataset.dims)}")
    print(f"  Time range: {final_dataset.time.min().values} to {final_dataset.time.max().values}")
    print(f"  Time steps: {len(final_dataset.time)}")
    
    # Save processed file
    output_path = output_dir / "processed_uerra_1km.nc"
    print(f"\nSaving to: {output_path}")
    
    # Encode with compression
    encoding = {}
    for var in final_dataset.data_vars:
        encoding[var] = {'zlib': True, 'complevel': 4, 'dtype': 'float32'}
    
    print("  Saving with compression...")
    final_dataset.to_netcdf(output_path, encoding=encoding)
    
    # Verify file
    file_size_gb = output_path.stat().st_size / (1024**3)
    print(f"  ✓ Saved successfully!")
    print(f"  File size: {file_size_gb:.2f} GB")
    
    print("\n" + "=" * 70)
    print("STEP 3 COMPLETE")
    print("=" * 70)
    
    return output_path

def main():
    """Main workflow for Step 3"""
    output_path = process_uerra_data()
    
    if output_path and output_path.exists():
        print("\n" + "=" * 70)
        print("DELIVERABLE VERIFICATION")
        print("=" * 70)
        print("\nRun this command to verify the output:")
        print(f'ncdump -h {output_path} | grep "float\\|double"')
        print("\nExpected variables: si10, r2, t2m")

if __name__ == "__main__":
    main()