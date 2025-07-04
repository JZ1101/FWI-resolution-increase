# Experiment Data Download Scripts

This folder contains data download scripts for preliminary testing purposes. All scripts are configured to download data for the year **2017 only** to facilitate initial testing and validation of the download pipeline.

## Contents

### ERA5_reanalysis_atmospheric_land-surface_parameters/
- **daily.py**: Downloads ERA5-Land reanalysis data (~10 km resolution) for 2017
  - Variables: 2m dewpoint temperature, 2m temperature, 10m wind components (u/v), total precipitation
  - Dataset: `reanalysis-era5-land`
  - Time: Daily at 00:00 UTC
  - Area: Portugal [42.2, -9.6, 36.8, -6.2]

### UERRA_reanalysis_atmospheric_parameters/
- **daily.py**: Downloads UERRA MESCAN-SURFEX reanalysis data (~5 km resolution) for 2017
  - Variables: 10m wind speed, 2m relative humidity, 2m temperature, total precipitation
  - Dataset: `reanalysis-uerra-europe-single-levels`
  - Time: Daily at 00:00 UTC
  - Area: Portugal [42.2, -9.6, 36.8, -6.2]

## Purpose

These scripts are designed for:
- **Initial testing** of the data download pipeline
- **Validation** of API connectivity and configuration
- **Format verification** of downloaded netCDF files
- **System testing** with a smaller dataset before full-scale downloads

## Configuration

Both scripts are configured with:
- **Year range**: 2017 (single year for testing)
- **Output directory**: `C:\Personal_Files\UCL_FT\Finanl_Project\dataset`
- **Format**: NetCDF (.nc files)
- **Logging**: INFO level with timestamps

## Usage

1. Ensure your CDS API credentials are properly configured in `~/.cdsapirc`
2. Run either script to download 2017 data for testing:
   ```bash
   python daily.py
   ```

## Output Files

- ERA5-Land: `era5_land_2017.nc`
- UERRA: `uerra_mescan_2017.nc`

## Next Steps

After successful testing with 2017 data, modify the `YEARS` variable in the production scripts to download the full time series (2015-2018 for ERA5-Land, 2000-2019 for UERRA).