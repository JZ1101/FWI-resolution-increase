"""Training utilities for FWI downscaling models"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pickle
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class Trainer:
    """Training manager for downscaling models"""
    
    def __init__(
        self,
        model,
        output_dir: str = "outputs/models",
        log_dir: str = "outputs/logs"
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': {}
        }
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 1,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model
        
        For traditional ML models, epochs=1 since they don't iterate
        For deep learning models, this would handle multiple epochs
        """
        
        logger.info(f"Starting training with {X_train.shape[0]} samples")
        start_time = datetime.now()
        
        if hasattr(self.model, 'fit'):
            self.model.fit(X_train, y_train, **kwargs)
            
            if X_val is not None:
                val_pred = self.model.predict(X_val)
                if y_val is not None:
                    val_loss = self.calculate_loss(y_val, val_pred)
                    self.history['val_loss'].append(val_loss)
                    logger.info(f"Validation loss: {val_loss:.4f}")
                    
        else:
            logger.warning("Model does not have a fit method")
            
        training_time = (datetime.now() - start_time).total_seconds()
        
        self.history['training_time'] = training_time
        self.history['n_samples'] = X_train.shape[0]
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return self.history
    
    def calculate_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate MSE loss"""
        return np.mean((y_true - y_pred) ** 2)
    
    def save_model(self, name: str = None) -> Path:
        """Save trained model"""
        
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"model_{timestamp}"
            
        model_path = self.output_dir / f"{name}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
            
        history_path = self.output_dir / f"{name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
            
        logger.info(f"Model saved to {model_path}")
        return model_path
    
    @classmethod
    def load_model(cls, model_path: str):
        """Load a saved model"""
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        return model


class EarlyStopping:
    """Early stopping callback"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
        return self.should_stop


def cross_validate(
    model_class,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    **model_kwargs
) -> Dict[str, Any]:
    """Perform k-fold cross-validation"""
    
    n_samples = X.shape[0]
    fold_size = n_samples // n_folds
    scores = []
    
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n_samples
        
        val_idx = list(range(val_start, val_end))
        train_idx = list(range(0, val_start)) + list(range(val_end, n_samples))
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = model_class(**model_kwargs)
        trainer = Trainer(model)
        
        trainer.train(X_train, y_train, X_val, y_val)
        
        val_pred = model.predict(X_val)
        score = np.mean((y_val - val_pred) ** 2)
        scores.append(score)
        
        logger.info(f"Fold {fold + 1}/{n_folds} - MSE: {score:.4f}")
        
    return {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'scores': scores
    }