#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nc_to_csv.py

Convert UERRA MESCAN-SURFEX netCDF files to a single CSV table by day,
keeping only 2015–2018 data for Portugal sub-region (36°–43°N, –10°–-6°E).
Rounds coordinates to 1 decimal place and averages duplicate values.
"""
import os
import glob
import xarray as xr
import pandas as pd
import numpy as np

# Input/output paths - current folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NC_DIR     = SCRIPT_DIR
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "uerra_2017_PT_averaged.csv")

# Variables to export (will be filtered based on availability)
DESIRED_VARS = ["si10", "r2", "t2m", "tp"]

# Common alternative variable names in UERRA datasets
VAR_ALIASES = {
    "si10": ["si10", "10m_wind_speed", "ws10"],
    "r2": ["r2", "2m_relative_humidity", "rh2m"],
    "t2m": ["t2m", "2m_temperature", "temp2m"],
    "tp": ["tp", "total_precipitation", "precip", "pr"]
}

# Portugal sub-region boundaries
LAT_MIN, LAT_MAX = 36.8, 42.2
LON_MIN, LON_MAX = -9.6, -6.2

# Only process these years
YEARS = {"2017"}

def nc_files_list(nc_dir):
    """List all .nc files in directory"""
    return sorted(glob.glob(os.path.join(nc_dir, "*.nc")))

def find_available_vars(ds):
    """Find which variables are available in the dataset"""
    available_vars = {}
    dataset_vars = list(ds.data_vars.keys())
    
    print(f"  Available variables in dataset: {dataset_vars}")
    
    for desired_var in DESIRED_VARS:
        found_var = None
        # Check if exact match exists
        if desired_var in dataset_vars:
            found_var = desired_var
        else:
            # Check aliases
            for alias in VAR_ALIASES.get(desired_var, []):
                if alias in dataset_vars:
                    found_var = alias
                    break
        
        if found_var:
            available_vars[desired_var] = found_var
            print(f"  ✓ {desired_var} -> {found_var}")
        else:
            print(f"  ✗ {desired_var} not found")
    
    return available_vars

def process_and_append(nc_file, csv_path, first_write):
    """Process single netCDF file and append to CSV"""
    # Open dataset
    ds = xr.open_dataset(nc_file, engine="netcdf4")
    
    print(f"Processing {os.path.basename(nc_file)}...")
    print(f"  Dataset dimensions: {dict(ds.dims)}")
    
    # Find available variables
    available_vars = find_available_vars(ds)
    
    if not available_vars:
        print(f"  ⚠ No required variables found, skipping")
        ds.close()
        return first_write

    # Convert longitude from [0,360) to [-180,180) if needed
    if ds.longitude.max() > 180:
        ds = ds.assign_coords(
            longitude = (((ds.longitude + 180) % 360) - 180)
        )
        print(f"  ✓ Converted longitude coordinates")

    # Find time coordinate
    time_dim = next((c for c, vals in ds.coords.items()
                     if np.issubdtype(vals.dtype, np.datetime64)), None)
    if time_dim is None:
        print("  ⚠ No time coordinate found, skipping")
        ds.close()
        return first_write

    print(f"  ✓ Time coordinate: {time_dim}")
    print(f"  ✓ Time range: {ds[time_dim].values[0]} to {ds[time_dim].values[-1]}")

    # Show original coordinate precision
    print(f"  ✓ Original latitude range: {ds.latitude.values.min():.6f} to {ds.latitude.values.max():.6f}")
    print(f"  ✓ Original longitude range: {ds.longitude.values.min():.6f} to {ds.longitude.values.max():.6f}")

    # Process each time step
    time_count = 0
    for t in ds[time_dim].values:
        ds_t = ds.sel({time_dim: t})
        
        # Get lat/lon coordinates
        if ds_t.latitude.ndim == 1 and ds_t.longitude.ndim == 1:
            # 1D coordinates - create meshgrid
            lon_1d = ds_t.longitude.values
            lat_1d = ds_t.latitude.values
            lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
            lat_flat = lat_2d.flatten()
            lon_flat = lon_2d.flatten()
        else:
            # 2D coordinates
            lat_flat = ds_t["latitude"].values.flatten()
            lon_flat = ds_t["longitude"].values.flatten()
        
        # Round coordinates to 1 decimal place
        lat_rounded = np.round(lat_flat, 1)
        lon_rounded = np.round(lon_flat, 1)
        
        # Extract variable data
        data = {}
        for desired_var, actual_var in available_vars.items():
            data[desired_var] = ds_t[actual_var].values.flatten()

        df = pd.DataFrame({
            "time":      [t] * lat_flat.size,
            "latitude":  lat_rounded,
            "longitude": lon_rounded,
            **data
        })
        
        # Filter Portugal sub-region
        df = df[
            (df.latitude  >= LAT_MIN) & (df.latitude  <= LAT_MAX) &
            (df.longitude >= LON_MIN)   & (df.longitude <= LON_MAX)
        ]
        
        # Remove NaN values
        df = df.dropna()
        
        if df.empty:
            continue

        # Group by time, latitude, longitude and take average of duplicate coordinates
        groupby_cols = ['time', 'latitude', 'longitude']
        value_cols = [col for col in df.columns if col not in groupby_cols]
        
        if value_cols:
            df_averaged = df.groupby(groupby_cols)[value_cols].mean().reset_index()
            
            # Show how many duplicates were found
            original_count = len(df)
            averaged_count = len(df_averaged)
            if original_count > averaged_count:
                print(f"  ✓ Averaged {original_count} points to {averaged_count} points (removed {original_count - averaged_count} duplicates)")
        else:
            df_averaged = df.drop_duplicates(subset=groupby_cols)
        
        df_averaged.to_csv(
            csv_path,
            mode='w' if first_write else 'a',
            header=first_write,
            index=False
        )
        first_write = False
        time_count += 1

    print(f"  ✓ Processed {time_count} time steps")
    print(f"  ✓ Rounded coordinates to 1 decimal place and averaged duplicates")
    ds.close()
    return first_write

def main():
    files = nc_files_list(NC_DIR)
    # Only keep files that contain target years in filename
    files = [f for f in files if any(year in os.path.basename(f) for year in YEARS)]
    
    print(f"Found {len(files)} files for years {sorted(YEARS)}. Start processing...")
    print(f"Rounding coordinates to 1 decimal place and averaging duplicates")
    
    first = True
    for nc in files:
        first = process_and_append(nc, OUTPUT_CSV, first)
    print(f"✅ Done. Subset CSV saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()