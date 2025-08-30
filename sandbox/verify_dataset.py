#!/usr/bin/env python3
"""
Verification script for FWIDataset class.
Tests the dataset loading and tensor generation functionality.
"""

import sys
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.modeling.dataset import FWIDataset


def main():
    """Verify the FWIDataset implementation."""
    
    print("=" * 80)
    print("FWIDataset Class Verification")
    print("=" * 80)
    
    # Initialize the dataset
    print("\n1. Initializing FWIDataset...")
    dataset = FWIDataset(config_path="configs/params.yaml")
    print(f"   Dataset initialized successfully")
    print(f"   Total samples: {len(dataset)}")
    
    # Get the first sample
    print("\n2. Retrieving first sample (dataset[0])...")
    input_tensor, target_tensor = dataset[0]
    print(f"   Sample retrieved successfully")
    
    # Print input tensor statistics
    print("\n3. INPUT TENSOR STATISTICS:")
    print(f"   Shape: {tuple(input_tensor.shape)}")
    print(f"   Data type: {input_tensor.dtype}")
    print(f"   Min value: {input_tensor.min().item():.6f}")
    print(f"   Max value: {input_tensor.max().item():.6f}")
    print(f"   Mean value: {input_tensor.mean().item():.6f}")
    print(f"   Std deviation: {input_tensor.std().item():.6f}")
    print(f"   Contains NaN: {torch.isnan(input_tensor).any().item()}")
    print(f"   Contains Inf: {torch.isinf(input_tensor).any().item()}")
    
    # Print target tensor statistics
    print("\n4. TARGET TENSOR STATISTICS:")
    print(f"   Shape: {tuple(target_tensor.shape)}")
    print(f"   Data type: {target_tensor.dtype}")
    print(f"   Min value: {target_tensor.min().item():.6f}")
    print(f"   Max value: {target_tensor.max().item():.6f}")
    print(f"   Mean value: {target_tensor.mean().item():.6f}")
    print(f"   Std deviation: {target_tensor.std().item():.6f}")
    print(f"   Contains NaN: {torch.isnan(target_tensor).any().item()}")
    print(f"   Contains Inf: {torch.isinf(target_tensor).any().item()}")
    
    # Additional verification details
    print("\n5. DATASET CONFIGURATION:")
    print(f"   Number of predictor variables: {dataset.n_channels}")
    print(f"   Spatial dimensions: {dataset.height} x {dataset.width}")
    print(f"   Low-res factor: {dataset.low_res_factor}")
    print(f"   Target variable: {dataset.target_var}")
    
    # Show first few predictor variable names
    print(f"\n   Predictor variables (first 5):")
    for i, var in enumerate(dataset.predictor_vars[:5]):
        print(f"     {i+1}. {var}")
    if len(dataset.predictor_vars) > 5:
        print(f"     ... and {len(dataset.predictor_vars) - 5} more")
    
    print("\n" + "=" * 80)
    print("Verification Complete: Dataset is functional")
    print("=" * 80)


if __name__ == "__main__":
    main()