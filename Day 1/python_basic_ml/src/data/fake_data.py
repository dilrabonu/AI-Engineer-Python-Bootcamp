import numpy as np 
from typing import Tuple

def generate_sample(shape: Tuple[int, int] = (28, 28)) -> np.ndarray:
    """Generate a random grayscale image for demo ML pipeline.

    Args:
        shape: (Tuple[int, int]): Height and width of the fake image.

    Returns:
        np.ndarray: A 2D array (H, W) with values in range [0,1].
    """ 
    image = np.random.rand(*shape).astype("float32")
    return image 