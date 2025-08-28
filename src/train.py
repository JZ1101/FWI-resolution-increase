#!/usr/bin/env python3
"""
Training module for FWI super-resolution models

Handles training loops, optimization, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.model import create_model, count_parameters


def setup_logging(log_dir: Path, level: str = "INFO"):
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_data_loaders(
    X: np.ndarray, 
    y: np.ndarray,
    split_indices: Dict[str, slice],
    batch_size: int,
    device: torch.device
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for training, validation, and test sets.
    
    Args:
        X: Input features
        y: Target values  
        split_indices: Dictionary with train/val/test slices
        batch_size: Batch size for training
        device: PyTorch device
        
    Returns:
        Dictionary of DataLoaders
    """
    dataloaders = {}
    
    for split_name, indices in split_indices.items():
        X_split = X[indices]
        y_split = y[indices]
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_split)
        y_tensor = torch.FloatTensor(y_split)
        
        # Add channel dimension if needed
        if len(X_tensor.shape) == 3:  # (batch, height, width)
            X_tensor = X_tensor.unsqueeze(1)  # (batch, 1, height, width)
        if len(y_tensor.shape) == 3:
            y_tensor = y_tensor.unsqueeze(1)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        shuffle = (split_name == 'train')
        
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=(device.type == 'cuda')
        )
    
    return dataloaders


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    with tqdm(dataloader, desc="Training") as pbar:
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Handle shape mismatch if needed
            if outputs.shape != targets.shape:
                outputs = outputs.view_as(targets)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        with tqdm(dataloader, desc="Validating") as pbar:
            for inputs, targets in pbar:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                
                if outputs.shape != targets.shape:
                    outputs = outputs.view_as(targets)
                
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                # Calculate additional metrics
                mse = torch.mean((outputs - targets) ** 2).item()
                mae = torch.mean(torch.abs(outputs - targets)).item()
                
                total_mse += mse
                total_mae += mae
                
                pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches
    metrics = {
        'mse': total_mse / num_batches,
        'mae': total_mae / num_batches,
        'rmse': np.sqrt(total_mse / num_batches)
    }
    
    return avg_loss, metrics


def train_model(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    config: Dict,
    device: torch.device,
    logger: logging.Logger
) -> Dict:
    """
    Main training loop with validation and checkpointing.
    
    Args:
        model: PyTorch model to train
        dataloaders: Dictionary of DataLoaders
        config: Training configuration
        device: PyTorch device
        logger: Logger instance
        
    Returns:
        Dictionary with training history and best model state
    """
    # Setup training parameters
    epochs = config['training']['epochs']
    learning_rate = float(config['training']['learning_rate'])
    weight_decay = float(config['training'].get('weight_decay', 1e-4))
    patience = config['training'].get('patience', 10)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=patience // 2,
        factor=0.5
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training history
    history = {
        'train_losses': [],
        'val_losses': [],
        'val_metrics': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    logger.info(f"Starting training for {epochs} epochs...")
    logger.info(f"Model: {config['model']['architecture']}")
    logger.info(f"Parameters: {count_parameters(model):,}")
    
    # Training loop
    for epoch in range(1, epochs + 1):
        logger.info(f"\nEpoch {epoch}/{epochs}")
        
        # Train
        train_loss = train_epoch(
            model, dataloaders['train'], 
            criterion, optimizer, device
        )
        history['train_losses'].append(train_loss)
        
        # Validate
        val_loss, val_metrics = validate_epoch(
            model, dataloaders['val'],
            criterion, device
        )
        history['val_losses'].append(val_loss)
        history['val_metrics'].append(val_metrics)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            logger.info(f"New best validation loss: {val_loss:.6f}")
        else:
            patience_counter += 1
        
        # Log progress
        logger.info(f"Train Loss: {train_loss:.6f}")
        logger.info(f"Val Loss: {val_loss:.6f}, RMSE: {val_metrics['rmse']:.4f}, MAE: {val_metrics['mae']:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final test evaluation
    if 'test' in dataloaders:
        test_loss, test_metrics = validate_epoch(
            model, dataloaders['test'],
            criterion, device
        )
        history['test_loss'] = test_loss
        history['test_metrics'] = test_metrics
        logger.info(f"\nFinal test loss: {test_loss:.6f}")
        logger.info(f"Test RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}")
    
    return history


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: Dict,
    config: Dict,
    filepath: Path
):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'config': config
    }
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"✅ Saved checkpoint: {filepath}")


def load_checkpoint(filepath: Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
    """Load training checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint.get('history', {})


def plot_training_history(history: Dict, output_dir: Path):
    """Plot training and validation loss curves"""
    if not history.get('train_losses') or not history.get('val_losses'):
        return
    
    plt.figure(figsize=(12, 5))
    
    # Loss curve
    plt.subplot(1, 2, 1)
    epochs = range(1, len(history['train_losses']) + 1)
    plt.plot(epochs, history['train_losses'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_losses'], 'r-', label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Metrics curve
    if history.get('val_metrics'):
        plt.subplot(1, 2, 2)
        rmse_values = [m['rmse'] for m in history['val_metrics']]
        mae_values = [m['mae'] for m in history['val_metrics']]
        plt.plot(epochs, rmse_values, 'g-', label='RMSE')
        plt.plot(epochs, mae_values, 'm-', label='MAE')
        plt.title('Validation Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved training history plot: {output_dir / 'training_history.png'}")


def spatial_cross_validation(
    model_class,
    X: np.ndarray,
    y: np.ndarray,
    config: Dict,
    n_folds: int = 5
) -> Dict:
    """
    Perform spatial cross-validation for robust evaluation.
    
    Args:
        model_class: Model class to instantiate
        X: Input features
        y: Target values
        config: Configuration dictionary
        n_folds: Number of spatial folds
        
    Returns:
        Dictionary with cross-validation results
    """
    # This is a placeholder for spatial CV
    # In full implementation, would divide region into spatial blocks
    
    results = {
        'fold_scores': [],
        'mean_rmse': 0.0,
        'std_rmse': 0.0
    }
    
    print(f"Spatial cross-validation with {n_folds} folds")
    # Implementation would go here
    
    return results


if __name__ == "__main__":
    # Test the training module
    print("Training module loaded successfully")
    print("Functions available:")
    print("  - train_model()")
    print("  - train_epoch()")
    print("  - validate_epoch()")
    print("  - save_checkpoint()")
    print("  - load_checkpoint()")
    print("  - spatial_cross_validation()")