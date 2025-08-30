#!/usr/bin/env python3
"""
FWI Dataset for PyTorch training.
Loads normalized multi-channel predictor data and FWI target.
"""

import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
from scipy.ndimage import zoom
from pathlib import Path
from typing import Tuple, Optional, List, Union
import yaml


class FWIDataset(Dataset):
    """PyTorch Dataset for FWI super-resolution training."""
    
    def __init__(
        self, 
        data_path: Optional[str] = None,
        config_path: str = "configs/params.yaml",
        predictor_vars: Optional[List[str]] = None,
        target_var: str = "fwi",
        low_res_factor: int = 5,  # 25km / 5km = 5
        transform=None
    ):
        """
        Initialize the FWI dataset.
        
        Args:
            data_path: Path to the normalized NetCDF file. If None, reads from config.
            config_path: Path to configuration file (default: 'configs/params.yaml')
            predictor_vars: List of predictor variable names. If None, uses default set.
            target_var: Name of the target variable (default: 'fwi')
            low_res_factor: Factor for creating low-resolution FWI (default: 5 for 25km from 5km)
            transform: Optional transform to apply to the data
        """
        # Load config if data_path not provided
        if data_path is None:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            data_path = config['paths']['normalized_file']
            
        self.data_path = Path(data_path)
        self.target_var = target_var
        self.low_res_factor = low_res_factor
        self.transform = transform
        
        # Load the dataset
        self.ds = xr.open_dataset(self.data_path)
        
        # Define predictor variables if not specified
        if predictor_vars is None:
            # Use all variables except FWI and coordinate variables
            self.predictor_vars = [
                'si10', 'r2', 't2m', 'd2m', 'tp', 'u10', 'v10',
                'wind_speed', 'relative_humidity',
                'lc_frac_tree_cover', 'lc_frac_shrubland', 'lc_frac_grassland',
                'lc_frac_cropland', 'lc_frac_bare_sparse', 'lc_frac_snow_ice',
                'lc_frac_water', 'lc_frac_wetland', 'lc_frac_mangroves',
                'lc_frac_built_up', 'lc_frac_moss_lichen'
            ]
        else:
            self.predictor_vars = predictor_vars
            
        # Verify all variables exist
        for var in self.predictor_vars:
            if var not in self.ds.data_vars:
                raise ValueError(f"Variable {var} not found in dataset")
        
        if target_var not in self.ds.data_vars:
            raise ValueError(f"Target variable {target_var} not found in dataset")
            
        # Get dimensions
        self.n_times = len(self.ds.time)
        self.n_channels = len(self.predictor_vars)
        self.height = len(self.ds.latitude)
        self.width = len(self.ds.longitude)
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.n_times
    
    def _create_low_res_fwi(self, fwi_high_res: np.ndarray) -> np.ndarray:
        """
        Create low-resolution FWI by downsampling and then upsampling with bicubic interpolation.
        
        Args:
            fwi_high_res: High-resolution FWI array (height, width)
            
        Returns:
            Bicubic interpolated FWI at original resolution
        """
        # Downsample to low resolution
        low_res_shape = (
            self.height // self.low_res_factor,
            self.width // self.low_res_factor
        )
        
        # Use area-based downsampling (average pooling)
        fwi_low_res = zoom(
            fwi_high_res,
            (1/self.low_res_factor, 1/self.low_res_factor),
            order=1  # Bilinear for downsampling
        )
        
        # Upsample back to original resolution using bicubic interpolation
        fwi_bicubic = zoom(
            fwi_low_res,
            (self.low_res_factor, self.low_res_factor),
            order=3  # Bicubic interpolation
        )
        
        # Ensure output shape matches input shape
        if fwi_bicubic.shape != fwi_high_res.shape:
            # Crop or pad if needed due to rounding
            h_diff = self.height - fwi_bicubic.shape[0]
            w_diff = self.width - fwi_bicubic.shape[1]
            
            if h_diff > 0 or w_diff > 0:
                # Pad if upsampled is smaller
                pad_h = max(0, h_diff)
                pad_w = max(0, w_diff)
                fwi_bicubic = np.pad(
                    fwi_bicubic,
                    ((0, pad_h), (0, pad_w)),
                    mode='edge'
                )
            elif h_diff < 0 or w_diff < 0:
                # Crop if upsampled is larger
                fwi_bicubic = fwi_bicubic[:self.height, :self.width]
                
        return fwi_bicubic
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Time index
            
        Returns:
            Tuple of (input_tensor, target_tensor)
            - input_tensor: Multi-channel predictor data (C, H, W)
            - target_tensor: Single-channel bicubic interpolated FWI (1, H, W)
        """
        # Extract predictor variables for this time step
        predictors_list = []
        for var in self.predictor_vars:
            # Get the variable data
            var_array = self.ds[var].values
            
            # Handle variables with and without time dimension
            if var_array.ndim == 3:  # Has time dimension
                var_data = var_array[idx, :, :]
            elif var_array.ndim == 2:  # No time dimension (static variables like land cover)
                var_data = var_array[:, :]
            else:
                raise ValueError(f"Unexpected dimensions for variable {var}: {var_array.ndim}")
                
            # Handle NaN values by replacing with 0 (since data is normalized)
            var_data = np.nan_to_num(var_data, nan=0.0)
            predictors_list.append(var_data)
        
        # Stack to create multi-channel array
        input_array = np.stack(predictors_list, axis=0)  # (C, H, W)
        
        # Get high-resolution FWI target
        fwi_high_res = self.ds[self.target_var].values[idx, :, :]  # Direct numpy indexing
        fwi_high_res = np.nan_to_num(fwi_high_res, nan=0.0)
        
        # Create bicubic interpolated version as target
        fwi_bicubic = self._create_low_res_fwi(fwi_high_res)
        
        # Add channel dimension to target
        target_array = fwi_bicubic[np.newaxis, :, :]  # (1, H, W)
        
        # Convert to PyTorch tensors
        input_tensor = torch.from_numpy(input_array).float()
        target_tensor = torch.from_numpy(target_array).float()
        
        # Apply transforms if any
        if self.transform:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)
        
        return input_tensor, target_tensor
    
    def get_sample_metadata(self, idx: int) -> dict:
        """
        Get metadata for a specific sample.
        
        Args:
            idx: Time index
            
        Returns:
            Dictionary with metadata about the sample
        """
        time_val = self.ds.time.isel(time=idx).values
        return {
            'time': str(time_val),
            'index': idx,
            'n_channels': self.n_channels,
            'height': self.height,
            'width': self.width,
            'predictor_vars': self.predictor_vars,
            'target_var': self.target_var
        }