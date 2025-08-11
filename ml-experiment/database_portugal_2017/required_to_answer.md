# FWI Resolution Enhancement Results

## Problem
Increase FWI resolution from 25km to 1km using Portugal 2017 data.

## Data Sources
- ERA5 FWI (25km)
- ERA5 Temperature (25km) 
- ERA5-Land (10km)
- UERRA (5km)
- WorldCover (10m)

## Models Trained
1. **ANN** - 3 layers (128-64-32 neurons)
2. **XGBoost** - 200 trees, depth 8
3. **CNN** - 1D convolution 
4. **Ensemble** - Average of 2 best models

## Performance Results
Best model: Ensemble
- Test R² = 0.904
- Test RMSE = 3.661

## Validation Methods

### Back-Aggregation Validation
Aggregate 1km predictions back to 25km and compare with original.
- Correlation = 0.755
- RMSE = 4.865

### Spatial Correlation Analysis
Compare variance at different scales.
- 25km variance: 197.73
- 1km variance: 2.24
- Variance ratio: 0.011

### Cross-Scale Validation
Test model across different regions.
- Average R² = 0.949

## Key Finding
Low variance ratio (0.011) indicates model does spatial interpolation, not true resolution enhancement. Without ground truth 1km data, cannot prove actual fine-scale accuracy.

## Files Generated
- `model_results.csv` - Performance metrics
- `predictions_1km.csv` - High resolution predictions
- `validation_results.json` - Validation metrics

## Conclusion
Method produces consistent 1km predictions (correlation = 0.755 with 25km) but essentially smooths data rather than revealing true fine-scale patterns.