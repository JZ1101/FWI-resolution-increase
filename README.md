# FWI Resolution Increase - Master's Thesis Research

Fire Weather Index (FWI) super-resolution from 25km to 1km for Portugal using deep learning.

## Project Structure

```
├── configs/
│   └── params.yaml              # Single configuration file
├── data/
│   ├── 00_raw/                 # Raw downloaded data
│   ├── 01_processed/           # Intermediate processed data  
│   └── 02_final_unified/       # Final unified dataset
├── notebooks/
│   └── 01_data_exploration.ipynb  # Data exploration
├── reports/
│   └── figures/                # Generated plots and figures
├── src/
│   ├── data_processing.py      # Data loading and preprocessing
│   ├── model.py               # U-Net and baseline models
│   ├── train.py               # Training script
│   └── evaluate.py            # Evaluation and metrics
└── README.md
```

## Quick Start

1. **Setup Environment:**
```bash
# Install dependencies
pip install torch torchvision xarray matplotlib pandas pyyaml scikit-learn scipy seaborn tqdm
```

2. **Load Data:**
The unified dataset should be available at `data/02_final_unified/unified_fwi_dataset_2016_2017.nc` (2.28 GB).

3. **Explore Data:**
```bash
jupyter lab notebooks/01_data_exploration.ipynb
```

4. **Train Model:**
```bash
python src/train.py
```

5. **View Results:**
Results and plots will be saved in `reports/` directory.

## Configuration

All settings are centralized in `configs/params.yaml`:
- Model architecture (U-Net parameters)
- Training parameters (batch size, learning rate, epochs)
- Data paths and processing settings
- Evaluation metrics and thresholds

## Model Architecture

**Primary Model:** U-Net for image super-resolution
- Input: Low-resolution FWI + auxiliary weather data
- Output: High-resolution FWI at 1km resolution
- Architecture: Encoder-decoder with skip connections

**Baselines:** 
- Bilinear interpolation
- Simple feedforward network

## Evaluation

Key validation approaches:
- **Back-aggregation testing:** High-res predictions should aggregate back to original low-res
- **Fire event validation:** Pedrógão Grande fire detection (June 17, 2017)
- **Standard metrics:** RMSE, MAE, correlation, conservation error

## Data Processing

The project uses a preprocessed unified dataset combining:
- ERA5 FWI (25km resolution)
- ERA5 atmospheric parameters  
- ERA5-Land surface parameters
- ESA WorldCover land classification

Dataset covers Portugal (2016-2017) with ~70M observations over 165K land pixels.

## Key Files

- `src/data_processing.py` - Unified data loading and preprocessing
- `src/model.py` - U-Net and baseline model implementations  
- `src/train.py` - Complete training pipeline
- `src/evaluate.py` - Comprehensive evaluation suite
- `configs/params.yaml` - Single source of configuration

This simplified structure focuses on the core thesis contribution while maintaining research rigor.