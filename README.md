# Fire Weather Index Resolution Enhancement

## Project Overview

This project aims to develop a methodology to increase the resolution of ERA5 Fire Weather Index (FWI) data from its native ~25km resolution to ≤1km resolution, with a focus on Continental Portugal as a test case.

## Geographic Focus

**Study Area**: Continental Portugal
- Latitude: 36°N to 43°N  
- Longitude: -10°E to -6°E

## Project Goals

### Primary Objective
Develop and implement a method to successfully increase ERA5 FWI resolution from ~25km to ≤1km that:
- Maintains positive-definite FWI values (no negative outputs)
- Demonstrates reasonable higher resolution data through qualitative validation
- Remains transferable to other geographic regions with minimal modifications

### Secondary Objectives
- **Financial Impact Analysis**: Demonstrate material impact of increased FWI resolution using provided damage models and demo portfolio
- **Performance**: Achieve faster processing than standard bi-linear interpolation
- **Global Scalability**: Indicate potential for rapid global implementation

## Data Sources

### Reanalysis Data (2000-2019 recommended)

#### ERA5 Reanalysis (~25km, Atmospheric Parameters)
**Source**: [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/datasets/derived-era5-single-levels-daily-statistics?tab=overview)
- 10m u-component of wind (daily mean)
- 10m v-component of wind (daily mean)  
- 2m dewpoint temperature (daily mean)
- 2m temperature (daily maximum)
- Total precipitation (daily sum)

#### ERA5 Fire Weather Index (~25km)
**Source**: [CEMS Fire Historical](https://ewds.climate.copernicus.eu/datasets/cems-fire-historical-v1?tab=overview)
- Canadian Forest Service Fire Weather Index Rating System
- Grid: 0.25° × 0.25° (interpolated)

#### ERA5-Land Reanalysis (~10km, Atmospheric and Land-Surface)
**Source**: [ERA5-Land Reanalysis](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download)
- 2m dewpoint temperature
- 2m temperature
- 10m u-component of wind
- 10m v-component of wind
- Total precipitation

#### UERRA Reanalysis (~5km, Atmospheric Parameters)
**Source**: [UERRA Europe Single Levels](https://cds.climate.copernicus.eu/datasets/reanalysis-uerra-europe-single-levels?tab=overview)
- MESCAN-SURFEX analysis
- 10m wind speed
- 2m relative humidity
- 2m temperature
- Total precipitation

### Land Cover Data

#### ESA WorldCover (10m, 11 land cover types)
**Source**: [ESA WorldCover 2020](https://worldcover2020.esa.int/download)
- Access Portugal data in 3° × 3° lat/lon tiles
- 11 distinct land cover classifications

## Technical Requirements

- **Data Format**: NetCDF preferred for Python/R compatibility
- **License**: Creative Commons (unlimited open access)
- **Accounts**: Required for data access and download
- **Processing**: Transform to common tabular format for analysis

## Project Structure

```
FWI-resolution-increase/
├── download_scripts/     # Production data download scripts
├── experiment/          # Test scripts (2017 data only)
└── README.md           # This file
```

## Deliverables

1. **Primary Model(s)**: Resolution enhancement methodology
2. **Input Documentation**: Complete list of all data inputs
3. **Future Work**: Statements on next steps and recommendations
4. **Dissertation**: Academic documentation of methodology and results

## Next Steps

1. Set up data acquisition pipeline
2. Divide data structuring tasks
3. Develop resolution enhancement methodology
4. Validate results against quality criteria
5. Test transferability to other regions
