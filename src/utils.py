"""
Utility functions for data handling and model serialization.
"""

import joblib
import os
from pathlib import Path

def get_model_path(model_name):
    """Get the path for a model file."""
    models_dir = Path(__file__).parent.parent / "models"
    return os.path.join(models_dir, f"{model_name}.pkl")

def save_model(model, model_name):
    """Save a trained model using joblib."""
    model_path = get_model_path(model_name)
    joblib.dump(model, model_path)
    print(f"Model saved: {model_path}")

def load_model(model_name):
    """Load a trained model using joblib."""
    model_path = get_model_path(model_name)
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")

def get_data_path(filename):
    """Get the path for a data file."""
    data_dir = Path(__file__).parent.parent / "data"
    return os.path.join(data_dir, filename)
