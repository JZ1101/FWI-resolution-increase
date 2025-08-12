"""Utility functions for FWI project"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None
) -> None:
    """Setup logging configuration"""
    
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
        
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    logger.info(f"Loaded config from {config_path}")
    return config


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """Save configuration to YAML file"""
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    logger.info(f"Saved config to {save_path}")


def ensure_positive_definite(data: np.ndarray) -> np.ndarray:
    """Ensure FWI values remain positive (physical constraint)"""
    return np.maximum(data, 0)


def normalize_data(
    data: np.ndarray,
    method: str = 'minmax',
    axis: Optional[int] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalize data
    
    Returns:
        normalized_data, normalization_params (for denormalization)
    """
    
    if method == 'minmax':
        data_min = np.min(data, axis=axis, keepdims=True)
        data_max = np.max(data, axis=axis, keepdims=True)
        
        normalized = (data - data_min) / (data_max - data_min + 1e-8)
        
        params = {
            'method': 'minmax',
            'min': data_min,
            'max': data_max
        }
        
    elif method == 'standard':
        data_mean = np.mean(data, axis=axis, keepdims=True)
        data_std = np.std(data, axis=axis, keepdims=True)
        
        normalized = (data - data_mean) / (data_std + 1e-8)
        
        params = {
            'method': 'standard',
            'mean': data_mean,
            'std': data_std
        }
    else:
        raise ValueError(f"Unknown normalization method: {method}")
        
    return normalized, params


def denormalize_data(
    data: np.ndarray,
    params: Dict[str, Any]
) -> np.ndarray:
    """Denormalize data using saved parameters"""
    
    if params['method'] == 'minmax':
        return data * (params['max'] - params['min']) + params['min']
        
    elif params['method'] == 'standard':
        return data * params['std'] + params['mean']
        
    else:
        raise ValueError(f"Unknown normalization method: {params['method']}")


def get_portugal_bounds() -> Dict[str, float]:
    """Get standard Portugal geographic bounds"""
    return {
        'lat_min': 36.0,
        'lat_max': 43.0,
        'lon_min': -10.0,
        'lon_max': -6.0
    }


def create_grid(
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    resolution_km: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create latitude/longitude grid at specified resolution
    
    Args:
        lat_range: (min_lat, max_lat)
        lon_range: (min_lon, max_lon)
        resolution_km: Target resolution in kilometers
        
    Returns:
        lat_grid, lon_grid
    """
    
    lat_resolution = resolution_km / 111.0
    lon_resolution = resolution_km / (111.0 * np.cos(np.radians(np.mean(lat_range))))
    
    lats = np.arange(lat_range[0], lat_range[1] + lat_resolution, lat_resolution)
    lons = np.arange(lon_range[0], lon_range[1] + lon_resolution, lon_resolution)
    
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    return lat_grid, lon_grid


def calculate_grid_size(
    original_shape: Tuple[int, int],
    upscale_factor: int
) -> Tuple[int, int]:
    """Calculate target grid size after upscaling"""
    
    return (
        original_shape[0] * upscale_factor,
        original_shape[1] * upscale_factor
    )


def save_netcdf(
    data: np.ndarray,
    save_path: Union[str, Path],
    var_name: str = 'fwinx',
    coords: Optional[Dict] = None
) -> None:
    """Save data as NetCDF file"""
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if coords is None:
        if data.ndim == 3:
            coords = {
                'time': np.arange(data.shape[0]),
                'latitude': np.arange(data.shape[1]),
                'longitude': np.arange(data.shape[2])
            }
        else:
            coords = {
                'latitude': np.arange(data.shape[0]),
                'longitude': np.arange(data.shape[1])
            }
            
    ds = xr.Dataset(
        {var_name: (list(coords.keys()), data)},
        coords=coords
    )
    
    ds.to_netcdf(save_path)
    logger.info(f"Saved NetCDF to {save_path}")


class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None
        
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        import time
        self.elapsed = time.time() - self.start_time
        logger.info(f"{self.name} took {self.elapsed:.2f} seconds")