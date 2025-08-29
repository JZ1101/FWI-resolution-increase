# High-Resolution Fire Weather Index Downscaling using Self-Supervised Deep Learning

This repository contains the official code and data pipeline for the Master's thesis titled "[Your Thesis Title Here]". The project develops and validates a self-supervised U-Net model to downscale the 25km ERA5 Fire Weather Index (FWI) to a 1km resolution for Portugal.

## 🎯 Key Features

* **Self-Supervised Learning:** Trains a deep learning model without high-resolution ground truth.
* **Physics-Informed Loss:** Utilizes a composite loss function with a conservation constraint to ensure physical plausibility.
* **U-Net Architecture:** Employs a standard U-Net for a robust image-to-image translation task.
* **Reproducible Pipeline:** A fully containerized and scripted workflow from raw data ingestion to final model evaluation.
* **Comprehensive Validation:** Includes ablation studies, a critical case study on the 2017 Pedrógão Grande fire, and transferability tests.

## 📁 Project Structure

The project follows a standardized research structure to ensure clarity and reproducibility.

```
├── configs/
│   └── params.yaml              # Single configuration file
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   └── 01_data_exploration.ipynb
├── reports/
│   └── figures/
├── src/
│   ├── data_processing.py
│   ├── model.py
│   ├── loss.py
│   ├── train.py
│   └── evaluate.py
└── README.md
```

## ⚙️ Installation

This project uses `uv` for package management to ensure a reproducible environment.

1.  **Clone the repository:**
    ```bash
    git clone [your-repo-url]
    cd FWI-resolution-increase
    ```

2.  **Create the virtual environment and install dependencies:**
    ```bash
    # This command creates the virtual environment and installs all packages from pyproject.toml
    uv sync
    ```

## 🚀 Usage: End-to-End Workflow

The entire project can be run as a sequence of scripts.

### Step 1: Data Preprocessing
This step takes the raw data from `data/raw/` and generates the final, normalized, model-ready dataset in `data/processed/`.

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run the entire preprocessing pipeline
python src/data_processing.py
```

### Step 2: Model Training
This step runs the baseline models and then trains the primary U-Net model and its ablation variants using 4-fold spatial cross-validation.

```bash
# Run the baseline models first
python src/evaluate.py --mode=baselines

# Train the primary U-Net model
python src/train.py --experiment=primary_unet

# (Optional) Train the ablation models
python src/train.py --experiment=ablation_feature
python src/train.py --experiment=ablation_loss
```

### Step 3: Evaluation
This step takes the trained models and generates the final performance tables and figures for the thesis.

```bash
# Evaluate the trained U-Net and generate final figures/tables
python src/evaluate.py --mode=final
```

## 📊 Example Result

The goal of the U-Net is to produce a high-resolution FWI map that is more detailed and physically plausible than simple interpolation, especially during critical fire events.


*(A sample figure from `reports/figures/pedrogao_comparison.png` will be displayed here once generated)*


## 📄 License

This project is licensed under the [MIT License](LICENSE).