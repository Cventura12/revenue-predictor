import pickle
import os
from model import LinearRegressionGD


# Use absolute paths based on file location for consistency with api.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "trained_model.pkl")


def save_model(model: LinearRegressionGD) -> str:
    """
    Save a trained LinearRegressionGD model to disk using pickle.
    
    Args:
        model: Trained LinearRegressionGD instance to save
    
    Returns:
        str: Path where the model was saved
    
    Raises:
        IOError: If model directory cannot be created or file cannot be written
    """
    # Create models directory if it doesn't exist (using absolute path)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save the model using pickle
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    return str(MODEL_PATH)

def load_model() -> LinearRegressionGD:
    """
    Load a trained LinearRegressionGD model from disk.
    
    Returns:
        LinearRegressionGD: Loaded model instance
    
    Raises:
        FileNotFoundError: If no saved model exists
        pickle.UnpicklingError: If the file cannot be unpickled
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. "
            "Please train the model first using POST /train"
        )
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    return model


def model_exists() -> bool:
    """
    Check if a trained model exists on disk.
    
    Returns:
        bool: True if model file exists, False otherwise
    """
    return os.path.exists(MODEL_PATH)

