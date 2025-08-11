#!/usr/bin/env python3
"""
Analyze current downloaded data to understand what features we have
"""

import os
import zipfile
import xarray as xr
import pandas as pd

def analyze_data():
    data_dir = 'data'
    print("=== Current Data Analysis ===\n")
    
    # Check all .nc files
    for filename in os.listdir(data_dir):
        if filename.endswith('.nc'):
            filepath = os.path.join(data_dir, filename)
            size = os.path.getsize(filepath)
            
            print(f"File: {filename}")
            print(f"Size: {size/1024:.1f} KB")
            
            # Check if it's a ZIP file
            try:
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    print("Type: ZIP archive")
                    print("Contents:")
                    for name in zip_ref.namelist():
                        print(f"  - {name}")
                    
                    # Extract first file to examine
                    extract_dir = os.path.join(data_dir, 'temp_extract')
                    os.makedirs(extract_dir, exist_ok=True)
                    zip_ref.extractall(extract_dir)
                    
                    # Examine first NetCDF file inside
                    for extracted_file in os.listdir(extract_dir):
                        if extracted_file.endswith('.nc'):
                            try:
                                ds = xr.open_dataset(os.path.join(extract_dir, extracted_file))
                                print(f"Variables in {extracted_file}:")
                                for var in ds.data_vars:
                                    print(f"  - {var}: {ds[var].dims} {ds[var].shape}")
                                print(f"Time range: {len(ds.time)} time steps")
                                ds.close()
                                break
                            except Exception as e:
                                print(f"  Error reading {extracted_file}: {e}")
                    
                    # Clean up
                    import shutil
                    shutil.rmtree(extract_dir)
                    
            except zipfile.BadZipFile:
                # Not a ZIP, try reading directly as NetCDF
                print("Type: NetCDF file")
                try:
                    ds = xr.open_dataset(filepath)
                    print("Variables:")
                    for var in ds.data_vars:
                        print(f"  - {var}: {ds[var].dims} {ds[var].shape}")
                    print(f"Time range: {len(ds.time)} time steps")
                    if hasattr(ds, 'time'):
                        print(f"Date range: {ds.time.values[0]} to {ds.time.values[-1]}")
                    ds.close()
                except Exception as e:
                    print(f"Error reading NetCDF: {e}")
            
            print("-" * 50)

if __name__ == "__main__":
    analyze_data()