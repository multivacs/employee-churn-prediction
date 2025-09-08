import joblib
from pathlib import Path

def save_model(model, path: Path):
    """Save trained model to disk."""
    joblib.dump(model, path)

def load_model(path: Path):
    """Load trained model from disk."""
    return joblib.load(path)
