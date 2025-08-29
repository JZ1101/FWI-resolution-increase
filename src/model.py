#!/usr/bin/env python3
"""
Model architectures for FWI super-resolution

Contains U-Net and baseline model implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Optional, Tuple


class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return x


class UNet(nn.Module):
    """U-Net architecture for image super-resolution"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Extract U-Net parameters from config
        unet_config = config['model']['unet']
        in_channels = unet_config.get('in_channels', 1)
        out_channels = unet_config.get('out_channels', 1)
        base_channels = unet_config.get('base_channels', 64)
        depth = unet_config.get('depth', 4)
        dropout = unet_config.get('dropout', 0.1)
        
        self.depth = depth
        
        # Encoder (downsampling path)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        channels = in_channels
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.encoders.append(ConvBlock(channels, out_ch, dropout))
            if i < depth - 1:
                self.pools.append(nn.MaxPool2d(2))
            channels = out_ch
        
        # Bottleneck
        bottleneck_channels = base_channels * (2 ** depth)
        self.bottleneck = ConvBlock(channels, bottleneck_channels, dropout)
        
        # Decoder (upsampling path)
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        
        for i in range(depth - 1, -1, -1):
            in_ch = bottleneck_channels if i == depth - 1 else base_channels * (2 ** (i + 1))
            out_ch = base_channels * (2 ** i)
            
            self.upconvs.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            # Account for skip connection concatenation
            self.decoders.append(ConvBlock(out_ch * 2, out_ch, dropout))
            
            bottleneck_channels = out_ch
        
        # Final output layer
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder path with skip connections
        skip_connections = []
        
        for i in range(self.depth):
            x = self.encoders[i](x)
            skip_connections.append(x)
            if i < self.depth - 1:
                x = self.pools[i](x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        for i in range(self.depth):
            x = self.upconvs[i](x)
            skip = skip_connections[self.depth - 1 - i]
            
            # Handle size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, skip], dim=1)
            x = self.decoders[i](x)
        
        # Final output
        x = self.final_conv(x)
        
        return x


class SimpleCNN(nn.Module):
    """Simple CNN baseline for super-resolution"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        in_channels = config['model'].get('in_channels', 1)
        out_channels = config['model'].get('out_channels', 1)
        hidden_channels = 64
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(hidden_channels, out_channels, kernel_size=5, padding=2)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x


class BilinearBaseline(nn.Module):
    """Bilinear interpolation baseline (non-trainable)"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.scale_factor = config['data'].get('upscale_factor', 4)
        
    def forward(self, x):
        """Simple bilinear upsampling"""
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
    
    def parameters(self):
        """Return empty parameters (non-trainable model)"""
        return []


class RandomForestBaseline:
    """Random Forest baseline for comparison"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=config['training'].get('random_state', 42),
            n_jobs=-1
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the Random Forest model"""
        # Flatten spatial dimensions
        X_flat = X.reshape(X.shape[0], -1)
        y_flat = y.reshape(y.shape[0], -1)
        
        self.model.fit(X_flat, y_flat)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        original_shape = X.shape
        X_flat = X.reshape(X.shape[0], -1)
        
        y_flat = self.model.predict(X_flat)
        
        # Reshape back to original spatial dimensions
        if len(original_shape) == 4:  # (batch, channels, height, width)
            batch_size = original_shape[0]
            height = original_shape[2]
            width = original_shape[3]
            y = y_flat.reshape(batch_size, 1, height, width)
        else:
            y = y_flat.reshape(original_shape)
            
        return y


def create_model(config: Dict, model_type: Optional[str] = None) -> nn.Module:
    """
    Factory function to create models based on configuration.
    
    Args:
        config: Configuration dictionary
        model_type: Optional model type override
        
    Returns:
        Model instance
    """
    if model_type is None:
        model_type = config['model']['architecture']
    
    if model_type == 'unet':
        return UNet(config)
    elif model_type == 'simple_cnn':
        return SimpleCNN(config)
    elif model_type == 'bilinear':
        return BilinearBaseline(config)
    elif model_type == 'random_forest':
        return RandomForestBaseline(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a PyTorch model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model_shapes(config: Dict):
    """Test that models produce correct output shapes"""
    # Create test input
    batch_size = 2
    in_channels = config['model']['unet']['in_channels']
    height, width = 256, 256
    
    test_input = torch.randn(batch_size, in_channels, height, width)
    
    # Test U-Net
    unet = create_model(config, 'unet')
    unet.eval()
    with torch.no_grad():
        output = unet(test_input)
    
    print(f"✅ U-Net test passed:")
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Parameters: {count_parameters(unet):,}")
    
    # Test Simple CNN
    cnn = create_model(config, 'simple_cnn')
    cnn.eval()
    with torch.no_grad():
        output = cnn(test_input)
    
    print(f"✅ Simple CNN test passed:")
    print(f"   Output shape: {output.shape}")
    print(f"   Parameters: {count_parameters(cnn):,}")
    
    return True


if __name__ == "__main__":
    # Test models with sample config
    test_config = {
        'model': {
            'architecture': 'unet',
            'unet': {
                'in_channels': 1,
                'out_channels': 1,
                'base_channels': 64,
                'depth': 4,
                'dropout': 0.1
            },
            'in_channels': 1,
            'out_channels': 1
        },
        'data': {
            'upscale_factor': 4
        },
        'training': {
            'random_state': 42
        }
    }
    
    test_model_shapes(test_config)