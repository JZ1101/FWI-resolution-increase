#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convertfwi2017.py

Convert ERA5 Fire Weather Index GRIB files to CSV format
for Portugal sub-region (36.8°–42.2°N, -9.6°–-6.2°E).

Input: era5_fwi_YYYY.grib files
Output: era5_fwi_2017_portugal.csv (combined CSV)
"""

import os
import glob
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = SCRIPT_DIR
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "era5_fwi_2017_portugal.csv")

# Portugal sub-region boundaries
LAT_MIN, LAT_MAX = 36.8, 42.2
LON_MIN, LON_MAX = -9.6, -6.2

# Years to process
YEARS = {"2017"}

def list_fwi_files(directory):
    """List all FWI GRIB files in directory"""
    pattern = os.path.join(directory, "era5_fwi_*.grib")
    files = glob.glob(pattern)
    # Filter by year
    filtered_files = []
    for f in files:
        filename = os.path.basename(f)
        if any(year in filename for year in YEARS):
            filtered_files.append(f)
    return sorted(filtered_files)

def convert_fwi_file(grib_file):
    """Convert single FWI GRIB file to DataFrame"""
    print(f"Processing {os.path.basename(grib_file)}...")
    
    try:
        # Try different engines for GRIB files
        ds = None
        for engine in ["cfgrib", None]:
            try:
                if engine:
                    ds = xr.open_dataset(grib_file, engine=engine)
                else:
                    ds = xr.open_dataset(grib_file)
                print(f"  Successfully opened with engine: {engine or 'default'}")
                break
            except Exception as e:
                print(f"  Failed with engine {engine or 'default'}: {str(e)}")
                continue
        
        if ds is None:
            print(f"  Could not open {grib_file} with any engine")
            return pd.DataFrame()
        
        with ds:
            print(f"  Dataset dimensions: {dict(ds.dims)}")
            print(f"  Variables: {list(ds.data_vars.keys())}")
            print(f"  Coordinates: {list(ds.coords.keys())}")
            
            # Find FWI variable
            fwi_var = None
            possible_fwi_vars = ['fwi', 'fire_weather_index', 'fireweatherindex', 'FWI', 'fwinx']
            
            for var_name in ds.data_vars:
                if any(fwi_key in var_name.lower() for fwi_key in ['fwi', 'fire_weather_index', 'fireweather', 'fwinx']):
                    fwi_var = var_name
                    break
            
            if fwi_var is None:
                # Try exact matches
                for var_name in possible_fwi_vars:
                    if var_name in ds.data_vars:
                        fwi_var = var_name
                        break
            
            if fwi_var is None:
                print(f"  Warning: No FWI variable found. Available variables: {list(ds.data_vars.keys())}")
                return pd.DataFrame()
            
            print(f"  FWI variable: {fwi_var}")
            
            # Find coordinate names
            lat_coord = None
            lon_coord = None
            time_coord = None
            
            for coord_name in ds.coords:
                coord_lower = coord_name.lower()
                if 'lat' in coord_lower:
                    lat_coord = coord_name
                elif 'lon' in coord_lower:
                    lon_coord = coord_name
                elif 'time' in coord_lower or 'valid_time' in coord_lower:
                    time_coord = coord_name
            
            print(f"  Coordinates - Lat: {lat_coord}, Lon: {lon_coord}, Time: {time_coord}")
            
            if not all([lat_coord, lon_coord, time_coord]):
                print(f"  Warning: Missing coordinates. Available: {list(ds.coords.keys())}")
                return pd.DataFrame()
            
            # Check coordinate ranges
            print(f"  Latitude range: {ds[lat_coord].min().values:.3f} to {ds[lat_coord].max().values:.3f}")
            print(f"  Longitude range: {ds[lon_coord].min().values:.3f} to {ds[lon_coord].max().values:.3f}")
            
            # Convert longitude from [0,360) to [-180,180) if needed
            if ds[lon_coord].max() > 180:
                print("  Converting longitude from [0,360) to [-180,180)...")
                ds = ds.assign_coords({lon_coord: (((ds[lon_coord] + 180) % 360) - 180)})
                # Sort longitude coordinate
                ds = ds.sortby(lon_coord)
                print(f"  New longitude range: {ds[lon_coord].min().values:.3f} to {ds[lon_coord].max().values:.3f}")
            
            print(f"  Time range: {ds[time_coord].values[0]} to {ds[time_coord].values[-1]}")
            
            # Filter by Portugal region BEFORE converting to DataFrame
            print("  Filtering by Portugal region...")
            
            # Convert Portugal bounds to match coordinate system if needed
            if ds[lon_coord].min() >= 0:  # Still in [0,360) system
                # Convert Portugal bounds to [0,360) system
                lon_min_360 = LON_MIN + 360 if LON_MIN < 0 else LON_MIN
                lon_max_360 = LON_MAX + 360 if LON_MAX < 0 else LON_MAX
                print(f"  Using longitude bounds in [0,360) system: {lon_min_360:.1f} to {lon_max_360:.1f}")
                
                ds_filtered = ds.sel(
                    latitude=slice(LAT_MAX, LAT_MIN),  # Note: reverse order for ERA5
                    longitude=slice(lon_min_360, lon_max_360)
                )
            else:
                # Use original bounds
                print(f"  Using original longitude bounds: {LON_MIN:.1f} to {LON_MAX:.1f}")
                ds_filtered = ds.sel(
                    latitude=slice(LAT_MAX, LAT_MIN),  # Note: reverse order for ERA5
                    longitude=slice(LON_MIN, LON_MAX)
                )
            
            print(f"  Filtered dimensions: {dict(ds_filtered.dims)}")
            
            if ds_filtered.dims.get('latitude', 0) == 0 or ds_filtered.dims.get('longitude', 0) == 0:
                print("  Warning: No data points in filtered region!")
                return pd.DataFrame()
            
            # Convert the filtered dataset to DataFrame
            print("  Converting filtered dataset to DataFrame...")
            df = ds_filtered.to_dataframe().reset_index()
            
            # Use the time coordinate as the time column
            df['time'] = df[time_coord]
            
            # Rename columns to standard names
            df = df.rename(columns={
                lat_coord: 'latitude',
                lon_coord: 'longitude',
                fwi_var: 'fwi'
            })
            
            # Keep only the columns we need
            df = df[['time', 'latitude', 'longitude', 'fwi']]
            
            # Convert longitude back to [-180,180) if needed
            if df['longitude'].max() > 180:
                df['longitude'] = df['longitude'].apply(lambda x: x - 360 if x > 180 else x)
            
            # Additional filtering (in case slice didn't work perfectly)
            df = df[
                (df.latitude >= LAT_MIN) & (df.latitude <= LAT_MAX) &
                (df.longitude >= LON_MIN) & (df.longitude <= LON_MAX)
            ]
            
            # Remove NaN values
            df = df.dropna()
            
            print(f"  Processed {len(df)} data points")
            return df
                
    except Exception as e:
        print(f"  Error processing {grib_file}: {str(e)}")
        return pd.DataFrame()

def main():
    """Main conversion function"""
    print("ERA5 Fire Weather Index GRIB to CSV Converter")
    print("=" * 60)
    print(f"Processing directory: {INPUT_DIR}")
    print(f"Output file: {OUTPUT_CSV}")
    print(f"Portugal bounds: Lat[{LAT_MIN}, {LAT_MAX}], Lon[{LON_MIN}, {LON_MAX}]")
    print(f"Years: {sorted(YEARS)}")
    print("-" * 60)
    
    # Find all FWI files
    fwi_files = list_fwi_files(INPUT_DIR)
    
    if not fwi_files:
        print("No FWI GRIB files found!")
        return
    
    print(f"Found {len(fwi_files)} FWI files:")
    for f in fwi_files:
        print(f"  - {os.path.basename(f)}")
    
    print("\nStarting conversion...")
    
    # Process all files
    all_dataframes = []
    
    for grib_file in fwi_files:
        df = convert_fwi_file(grib_file)
        if not df.empty:
            all_dataframes.append(df)
    
    # Combine all data
    if all_dataframes:
        print(f"\nCombining data from {len(all_dataframes)} files...")
        final_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Sort by time and coordinates
        final_df = final_df.sort_values(['time', 'latitude', 'longitude'])
        
        # Round coordinates to reasonable precision
        final_df['latitude'] = final_df['latitude'].round(3)
        final_df['longitude'] = final_df['longitude'].round(3)
        
        # Round FWI values to reasonable precision
        final_df['fwi'] = final_df['fwi'].round(3)
        
        # Save to CSV
        final_df.to_csv(OUTPUT_CSV, index=False)
        
        print(f"\nConversion completed successfully!")
        print(f"Total data points: {len(final_df):,}")
        print(f"Date range: {final_df['time'].min()} to {final_df['time'].max()}")
        print(f"Latitude range: {final_df['latitude'].min():.3f} to {final_df['latitude'].max():.3f}")
        print(f"Longitude range: {final_df['longitude'].min():.3f} to {final_df['longitude'].max():.3f}")
        print(f"FWI range: {final_df['fwi'].min():.3f} to {final_df['fwi'].max():.3f}")
        print(f"CSV saved to: {OUTPUT_CSV}")
        
        # Show sample data
        print(f"\nSample data (first 5 rows):")
        print(final_df.head())
        
    else:
        print("No valid data found in any files!")

if __name__ == "__main__":
    main()