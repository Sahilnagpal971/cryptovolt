"""
ML Model Management and Training
"""
import logging
import joblib
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Model registry for version management and metadata tracking
    All models are versioned for reproducibility (per SRS requirement)
    """
    
    def __init__(self, registry_path: str = "./models"):
        """Initialize model registry"""
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        self.metadata_file = self.registry_path / "registry.json"
        self.registry = self._load_registry()
    
    def register_model(
        self,
        model: Any,
        model_name: str,
        model_type: str,
        version: str,
        metrics: Dict[str, float],
        config: Dict[str, Any] = None,
    ) -> str:
        """
        Register and save a trained model
        
        Returns:
            Model ID for reference
        """
        
        model_id = f"{model_name}_{version}_{datetime.utcnow().timestamp()}"
        model_dir = self.registry_path / model_id
        model_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            "model_id": model_id,
            "name": model_name,
            "type": model_type,
            "version": version,
            "metrics": metrics,
            "config": config or {},
            "created_at": datetime.utcnow().isoformat(),
            "path": str(model_path),
        }
        
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Update registry
        self.registry[model_id] = metadata
        self._save_registry()
        
        logger.info(f"Registered model: {model_id}")
        return model_id
    
    def get_model(self, model_id: str) -> Tuple[Any, Dict[str, Any]]:
        """Load model and metadata by ID"""
        metadata = self.registry.get(model_id)
        if not metadata:
            raise ValueError(f"Model not found: {model_id}")
        
        model = joblib.load(metadata["path"])
        return model, metadata
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models"""
        return list(self.registry.values())
    
    def get_latest_model(self, model_name: str) -> Tuple[Any, Dict[str, Any]]:
        """Get latest version of a model by name"""
        matching = [
            (id, meta) for id, meta in self.registry.items()
            if meta["name"] == model_name
        ]
        
        if not matching:
            raise ValueError(f"No models found: {model_name}")
        
        # Sort by creation date, get latest
        latest_id, latest_meta = sorted(
            matching,
            key=lambda x: x[1]["created_at"],
            reverse=True
        )[0]
        
        model = joblib.load(latest_meta["path"])
        return model, latest_meta
    
    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load registry from file"""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """Save registry to file"""
        with open(self.metadata_file, "w") as f:
            json.dump(self.registry, f, indent=2)


class XGBoostClassifier:
    """
    XGBoost classifier for signal generation
    Binary classification: Buy (1) vs Sell/Hold (0)
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize XGBoost classifier"""
        try:
            import xgboost as xgb
            self.xgb = xgb
        except ImportError:
            raise ImportError("XGBoost not installed")
        
        default_params = {
            "max_depth": 6,
            "eta": 0.1,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        }
        
        self.params = {**default_params, **(params or {})}
        self.model = None
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 100,
    ) -> Dict[str, float]:
        """
        Train the classifier
        
        Returns:
            Training metrics
        """
        
        dtrain = self.xgb.DMatrix(X_train, label=y_train)
        
        eval_list = [(dtrain, "train")]
        if X_val is not None and y_val is not None:
            dval = self.xgb.DMatrix(X_val, label=y_val)
            eval_list.append((dval, "val"))
        
        evals_result = {}
        self.model = self.xgb.train(
            self.params,
            dtrain,
            num_boost_round=epochs,
            evals=eval_list,
            evals_result=evals_result,
            verbose_eval=False,
        )
        
        # Calculate metrics
        metrics = self._calculate_metrics(self.model, X_val, y_val)
        
        logger.info(f"XGBoost training complete. Metrics: {metrics}")
        return metrics
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Make predictions
        
        Returns:
            {'signal': str, 'confidence': float, 'probability': float}
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        dtest = self.xgb.DMatrix(X)
        probs = self.model.predict(dtest)
        
        # Average probability if batch
        prob = float(np.mean(probs))
        confidence = abs(prob - 0.5) * 2  # 0 to 1
        
        if prob >= threshold:
            signal = "BUY"
        else:
            signal = "SELL"
        
        return {
            "signal": signal,
            "confidence": confidence,
            "probability": prob,
        }
    
    def _calculate_metrics(self, model, X_val, y_val) -> Dict[str, float]:
        """Calculate training metrics"""
        if X_val is None or y_val is None:
            return {"status": "training_complete"}
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
        
        dval = self.xgb.DMatrix(X_val)
        y_pred = model.predict(dval)
        y_pred_binary = (y_pred >= 0.5).astype(int)
        
        return {
            "accuracy": float(accuracy_score(y_val, y_pred_binary)),
            "precision": float(precision_score(y_val, y_pred_binary, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred_binary, zero_division=0)),
            "auc": float(roc_auc_score(y_val, y_pred)),
        }


class LSTMForecaster:
    """
    LSTM forecaster for price prediction
    Sequence-to-sequence model for time series forecasting
    """
    
    def __init__(self, sequence_length: int = 60, params: Dict[str, Any] = None):
        """Initialize LSTM forecaster"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            
            self.tf = tf
            self.Sequential = Sequential
            self.LSTM = LSTM
            self.Dense = Dense
            self.Dropout = Dropout
        except ImportError:
            raise ImportError("TensorFlow not installed")
        
        self.sequence_length = sequence_length
        self.params = params or {}
        self.model = None
        self.scaler = StandardScaler()
    
    def build_model(self, input_shape: Tuple[int, int]):
        """Build LSTM model architecture"""
        model = self.Sequential([
            self.LSTM(64, input_shape=input_shape, return_sequences=True),
            self.Dropout(0.2),
            self.LSTM(32, return_sequences=False),
            self.Dropout(0.2),
            self.Dense(16, activation='relu'),
            self.Dense(1, activation='linear'),
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae'],
        )
        
        self.model = model
        logger.info(f"LSTM model built with input shape: {input_shape}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """
        Train the LSTM model
        
        Returns:
            Training metrics
        """
        
        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))
        
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=0,
        )
        
        metrics = {
            "final_loss": float(history.history["loss"][-1]),
            "final_mae": float(history.history["mae"][-1]),
        }
        
        if validation_data:
            metrics["val_loss"] = float(history.history["val_loss"][-1])
            metrics["val_mae"] = float(history.history["val_mae"][-1])
        
        logger.info(f"LSTM training complete. Metrics: {metrics}")
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make price predictions"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        return self.model.predict(X, verbose=0)
