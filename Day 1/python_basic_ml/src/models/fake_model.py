import numpy as np 
from typing import Dict, Any

def predict(image: np.ndarray) -> Dict[str, Any]:
    """ Make a dummy prediction on a fake image.

    Args:
        image (np.ndarray): Input 2D array representing an image.

    Returns:
         Dict[str, Any]: Contains predicted class and confidence.
    """
    
    mean_value = float(np.mean(image))

    if mean_value > 0.5:
        pred_class = "bright"
    else:
        pred_class = "dark"

    return {
        "class" : pred_class,
        "confidence" : round(mean_value, 3),
    }