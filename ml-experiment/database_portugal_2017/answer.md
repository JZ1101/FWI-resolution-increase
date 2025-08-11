# FWI Resolution Enhancement - Portugal 2017

## 1. General Analysis of the Problem [10 marks]

### Problem Statement
Increase ERA5 Fire Weather Index (FWI) resolution from native ~25km to ≤1km for Portugal using machine learning downscaling techniques. Test case focuses on the catastrophic Pedrógão Grande fire event of June 17, 2017.

### a. Discussion of Features

**Data Sources Integration:**
- **ERA5 FWI (25km)**: Target variable, fire weather index values
- **ERA5 Temperature (25km)**: Daily maximum temperature 
- **ERA5-Land (10km)**: Higher resolution surface atmospheric data
- **UERRA (5km)**: Regional European reanalysis data
- **ESA WorldCover (10m)**: Land cover classification

**Feature Categories:**
1. **Temporal Features (13)**: Cyclical encoding (sin/cos), fire seasons, dates
2. **Spatial Features (10)**: Normalized coordinates, distance metrics, regional zones
3. **Land Cover Features (11)**: One-hot encoded vegetation types
4. **Lag Features (5)**: Previous day FWI values and rolling statistics

### b. Discussion of Number of Features
- **Total Features**: 44 engineered features from 5 data sources
- **Usable Features**: 37 after removing >50% missing data
- **Feature Importance**: Temporal features dominate (>95% importance)
- **Curse of Dimensionality**: Moderate risk with 37 features for 98,550 samples

### c. Discussion of Feasibility 
**Challenges:**
- No ground truth 1km FWI data for validation
- Multi-resolution data integration complexity
- Temporal dependencies in fire weather patterns
- Extreme weather events (fire days) are rare

**Opportunities:**
- Rich multi-source dataset available
- Strong temporal correlations in FWI
- Proven ML techniques for spatial downscaling
- Real fire event for validation

### d. Approach
1. **Multi-resolution data fusion** using coordinate interpolation
2. **Feature engineering** with temporal and spatial patterns
3. **Ensemble modeling** combining ANN, XGBoost, CNN
4. **Custom validation** including back-aggregation and fire event testing

## 2. Data Exploration and Feature Engineering [10 marks]

### a. Distributions of Data
- **Target Variable (FWI)**: Range 0-93.03, mean=19.08, heavily right-skewed
- **Temporal Coverage**: Complete 365 days (Jan-Dec 2017)
- **Spatial Coverage**: 270 coordinates across Portugal
- **Total Samples**: 98,550 (270 × 365)

### b. Outliers
- **High FWI Values**: >50 during extreme fire weather (summer months)
- **Spatial Outliers**: Coastal vs inland differences
- **Seasonal Patterns**: Summer peak, winter minimum
- **Missing Data**: Handled via mean imputation

### c. Correlations
- **Temporal Correlation**: Previous day FWI (85.6% feature importance)
- **Seasonal Correlation**: Strong fire season effects (May-October)
- **Spatial Correlation**: Regional patterns (North/Center/South Portugal)
- **Land Cover Impact**: Minimal correlation with vegetation types

### d. Scaling
- **StandardScaler**: Applied for neural networks (ANN, CNN)
- **Raw Features**: Used for tree-based models (XGBoost)
- **Feature Range**: Normalized coordinates [0,1], standardized meteorological variables

### e. Transformations
- **Cyclical Encoding**: Month/day as sin/cos pairs
- **Binary Indicators**: Fire season, regional zones
- **Lag Features**: 1, 3, 7-day FWI lags
- **Rolling Statistics**: 7-day mean and maximum FWI

### f. Data Balance
- **Temporal Balance**: Uniform daily sampling across year
- **Spatial Balance**: Regular grid coverage
- **Class Imbalance**: High FWI days (>30) are rare (~5% of data)
- **Seasonal Bias**: Summer months overrepresented in high FWI values

### g. New Features Generated
- **Distance Metrics**: Coast distance, Spain border distance
- **Regional Zones**: North/Center/South classification
- **Fire Season Indicators**: Peak fire season (Jun-Sep)
- **Land Cover One-hot**: 11 vegetation type categories

## 3. Model Training [10 marks]

### Training/Testing Split
- **Training**: Jan 1 - May 1 (32,400 samples)
- **Validation**: May 1 - June 1 (8,370 samples)  
- **Test**: June 1 - July 1 (8,100 samples)
- **Rationale**: Fire season testing, temporal validation

### Model Performance

| Model | Test R² | Architecture |
|-------|---------|--------------|
| ANN | -0.719 | 3 layers (128-64-32) |
| XGBoost | 0.588 | 200 trees, depth 8 |
| CNN | 0.558 | 1D convolution |
| **Ensemble** | **0.685** | **XGBoost + CNN** |

### a. Performance Measures

**Regression Metrics:**
- **RMSE**: 10.216 (Ensemble best)
- **R²**: 0.543 (Ensemble best)
- **MAE**: Calculated for all models

**Custom Validation Methods - ALL MODELS TESTED:**

**1. Back-Aggregation Validation** (Aggregate 1km → 25km vs original ERA5)
| Model | Correlation | RMSE | Interpretation |
|-------|-------------|------|----------------|
| ANN | -0.632 | 27.080 | Poor - negative correlation |
| XGBoost | 0.385 | 25.230 | Limited consistency |
| CNN | 0.006 | 16.077 | No consistency |
| **Ensemble** | **0.442** | **20.212** | **Best but still poor** |

**2. Spatial Correlation Analysis** (Variance at different scales)
| Model | Variance Ratio (1km/25km) | Over-smoothing Level |
|-------|---------------------------|---------------------|
| ANN | 0.001 | Extreme |
| XGBoost | 0.009 | High |
| CNN | 0.008 | High |
| **Ensemble** | **0.001** | **Extreme** |

**3. Cross-Scale Validation** (Leave-one-region-out)
| Model | Average R² | Spatial Generalization |
|-------|-----------|----------------------|
| All Models | 0.949 | Excellent across all regions |

4. **June 16 Fire Event Validation**
   - Test on Pedrógão Grande fire location (39.92°N, 8.15°W)
   - Compare ERA5 vs Enhanced FWI predictions
   - **Result**: [To be completed in experiment run]

### b. Cross-Validation
Regional cross-validation across 5 latitude bands (37°-42°N) shows consistent performance (R² = 0.949 average), indicating good spatial generalization.

### c. Bias and Variance
- **High Bias**: Temporal splits show performance degradation in summer months
- **Low Variance**: Ensemble reduces overfitting compared to individual models
- **Variance Ratio**: 0.001 indicates extreme over-smoothing

## 4. Model Optimization [10 marks]

### a. Hyperparameter Tuning

**ANN Optimization:**
- Hidden layers: (128, 64, 32) neurons
- Activation: ReLU 
- Solver: Adam with adaptive learning rate
- Regularization: α = 0.001
- Early stopping: 20 iterations no improvement

**XGBoost Optimization:**
- Trees: 200 estimators
- Depth: 8 maximum
- Learning rate: 0.05
- Subsample: 0.8
- Column sampling: 0.8

**CNN Optimization:**
- Architecture: 1D Conv → MaxPool → Conv → GlobalAvgPool → Dense
- Filters: 64 → 32
- Kernel size: 3
- Dropout: 0.3
- Early stopping with patience=10

**Ensemble Strategy:**
- Simple averaging of two best models
- Model selection based on validation R²
- XGBoost + CNN combination selected

## 5. Conclusions [10 marks]

### a. Algorithm Performance Comparison

**Best Model: Ensemble (XGBoost + CNN)**
- **Test R²**: 0.685 (best overall performance)
- **1km Validation Performance**: Mixed results across validation methods
- **Advantages**: Combines tree-based learning with spatial pattern detection
- **Critical Issue**: Still shows extreme over-smoothing (variance ratio = 0.001)

**Complete Performance Ranking (25km Test → 1km Validation):**
1. **Ensemble**: R²=0.685 → Back-agg=0.442, Variance ratio=0.001
2. **XGBoost**: R²=0.588 → Back-agg=0.385, Variance ratio=0.009 ⭐
3. **CNN**: R²=0.558 → Back-agg=0.006, Variance ratio=0.008
4. **ANN**: R²=-0.719 → Back-agg=-0.632, Variance ratio=0.001

**Key Finding**: XGBoost shows best 1km validation despite lower 25km performance - highest variance ratio (0.009) indicates less over-smoothing.

### b. Critical Findings

**Resolution Enhancement Assessment:**
- **Best Variance Ratio**: 0.009 (XGBoost) - still indicates severe over-smoothing
- **Best Back-aggregation**: 0.442 (Ensemble) - poor consistency with original data
- **All Models**: Show spatial interpolation rather than true resolution enhancement
- **Critical Limitation**: Without ground truth 1km data, cannot prove fine-scale accuracy

**Comprehensive Model Analysis:**
- **ANN**: Complete failure at both 25km (R²=-0.719) and 1km validation
- **CNN**: Decent 25km performance but no consistency in 1km back-aggregation (0.006)
- **Ensemble**: Best overall but extreme over-smoothing negates benefits
- **XGBoost**: Most promising for 1km - highest variance preservation

**Feature Importance:**
- **Temporal dominance**: Previous day FWI = 85.6% importance
- **Spatial features**: <5% combined importance
- **Implication**: Model relies on temporal persistence rather than spatial enhancement

### Fire Event Validation (June 16, 2017)
**Pedrógão Grande Fire Event Analysis:**
- **Fire Location**: 39.92°N, -8.15°W
- **Closest Model Coordinate**: 40.000°N, -8.250°W (14.2 km distance)
- **ERA5 FWI (25km)**: 26.17 → **HIGH risk**
- **Enhanced FWI (1km)**: 17.12 → **MODERATE risk**  
- **Model Underestimation**: -9.05 FWI units

**Critical Finding**: Model significantly underestimated fire risk on the day before the catastrophic fire, downgrading from HIGH to MODERATE risk level. This indicates the resolution enhancement method may smooth extreme weather conditions that are crucial for fire risk assessment.

### What Additional Steps Could Improve Models?

1. **Data Enhancement:**
   - Include more meteorological variables (wind, humidity)
   - Add topographical features (elevation, slope)
   - Incorporate satellite-derived fire indices

2. **Model Architecture:**
   - Attention mechanisms for spatial-temporal modeling
   - Graph neural networks for spatial relationships
   - Physics-informed neural networks with FWI equations

3. **Training Strategy:**
   - Transfer learning from other fire-prone regions
   - Multi-task learning with actual fire occurrence data
   - Adversarial training for extreme weather events

4. **Validation Enhancement:**
   - Validate against satellite fire detections
   - Use historical fire perimeter data
   - Cross-validation with different years

5. **Resolution Enhancement:**
   - Progressive upsampling techniques
   - Super-resolution deep learning approaches
   - Incorporate high-resolution static features (topography, land use)

### Final Assessment
The methodology successfully integrates multi-resolution data but reveals critical limitations for operational fire risk assessment. Key findings:

1. **Spatial Interpolation vs Enhancement**: Variance ratio of 0.002 and back-aggregation correlation of 0.361 indicate the model performs spatial smoothing rather than true resolution enhancement.

2. **Fire Event Underestimation**: On June 16, 2017 (day before Pedrógão Grande fire), the model underestimated fire risk by 9.05 FWI units, downgrading from HIGH to MODERATE risk level.

3. **Temporal Dominance**: 85.6% feature importance on previous day FWI shows the model relies on temporal persistence rather than spatial enhancement.

4. **Limited Spatial Enhancement**: Despite using 5 multi-resolution datasets, spatial features contribute <5% to model decisions.

**Recommendation**: While the method demonstrates technical feasibility for multi-source data integration, it fails to achieve true resolution enhancement and may be dangerous for operational fire management due to systematic underestimation of extreme fire weather conditions. Further research is needed with ground truth high-resolution fire weather observations and physics-informed modeling approaches that preserve extreme weather patterns rather than smoothing them.