# June 16, 2017 Pedrógão Grande Fire - Final FWI Outputs

## Fire Event Details
- **Date**: June 16, 2017
- **Location**: Pedrógão Grande, Portugal
- **Actual Coordinates**: 39.920°N, -8.150°W
- **Nearest ERA5 Grid Point**: 40.000°N, -8.250°W
- **Distance**: 14.2 km

## Model FWI Predictions

| Model | FWI Value | Risk Level | Difference from ERA5 |
|-------|-----------|------------|---------------------|
| ERA5 | 26.168 | HIGH | - |
| XGBoost | 20.226 | MODERATE | -5.942 |
| ANN | 14.680 | MODERATE | -11.488 |
| CNN | 18.558 | MODERATE | -7.610 |
| Ensemble | 19.392 | MODERATE | -6.776 |

## Key Findings
- **ALL models underestimated fire risk**
- Fire occurred under HIGH risk conditions (FWI > 21.3)
- All models predicted MODERATE risk (11.2 ≤ FWI < 21.3)
- XGBoost performed best: -5.9 FWI units error
- ANN performed worst: -11.5 FWI units error

## Files in this folder
- `june16_fwi_outputs.json` - Complete results in JSON format
- `fire_location_comparison.png` - Visual comparison graphs
- `simple_fire_visual.py` - Script to generate visualizations
- `final_fwi_outputs.py` - Script to extract FWI values
- `fire_location_comparison.py` - Full comparison script

## Risk Classification
- Very Low: 0-5.2
- Low: 5.2-11.2  
- Moderate: 11.2-21.3
- High: 21.3-38
- Extreme: 38+