# PROJECT GOALS AND GUIDELINES

## Main Project Goal
**"Overall goal is to decide and implement a method that successfully increase the resolution of the ERA5 Fire Weather Index (from its native ~25km to <=1km) for a test case in e.g., Portugal."**

## Success Criteria
1. **Positive-Definite Constraint**: The resulting higher resolution fire weather index data should remain positive-definite (i.e., not produce negative values)
2. **Quality Demonstration**: Any other qualitative arguments to demonstrate that the higher resolution data is reasonable:
   - Comparison to other datasets
   - Demonstration of smoothness
   - Comment on correlation to weather parameters
3. **Transferability**: The resulting model should be transferrable from the initial training region (e.g., designed for Portugal but can be used anywhere with limited/no modifications)

## Data Requirements
### Target Region: Continental Portugal
- Latitude: 36°N to 43°N
- Longitude: -10°E to -6°E

### Required Datasets
1. **ERA5 reanalysis (~25km)**: 
   - Fire Weather Index (fwinx)
   - Atmospheric parameters (wind, temperature, precipitation)
   
2. **ERA5-Land reanalysis (~10km)**:
   - Higher resolution atmospheric and land-surface parameters
   
3. **UERRA reanalysis (~5km)**:
   - Highest resolution atmospheric parameters
   
4. **ESA WorldCover (10m)**:
   - Land cover types for auxiliary features

### Time Period
- Recommended: Daily data for 2000-2019
- Testing: Using 2017 data

## Technical Requirements
1. **Data Format**: NetCDF format
2. **Resolution Target**: From ~25km to ≤1km
3. **Programming**: Python implementation
4. **Validation**: Must demonstrate quality of downscaled results

## Method Evaluation Checklist
- [ ] Uses real ERA5 FWI data (not synthetic)
- [ ] Achieves resolution increase (25km → ≤1km)
- [ ] Maintains positive-definite values (FWI ≥ 0)
- [ ] Shows reasonable spatial patterns (smooth, realistic)
- [ ] Demonstrates correlation with input data
- [ ] Provides validation metrics
- [ ] Model is transferable to other regions
- [ ] Processing time is practical