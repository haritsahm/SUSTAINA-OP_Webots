from functools import lru_cache
from typing import Union
import numpy as np

def sign(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Return the sign of the input value or array.
    
    This function returns -1 for negative values, 1 for positive values,
    and 0 for zero. It handles both scalar values and numpy arrays.
    
    Parameters
    ----------
    x : Union[float, np.ndarray]
        Input value or array
        
    Returns
    -------
    Union[float, np.ndarray]
        -1 if x < 0, 1 if x > 0, 0 if x == 0
    """
    if isinstance(x, np.ndarray):
        return np.sign(x)
    return -1.0 if x < 0.0 else 1.0 if x > 0.0 else 0.0


@lru_cache(maxsize=None)
def combination(n: int, r: int) -> int:
    """
    Calculate binomial coefficient (n choose r) using memoization.
    
    This function computes the number of ways to choose r items from a set of n items,
    using a recursive approach with memoization for efficiency.
    
    Parameters
    ----------
    n : int
        Total number of items
    r : int
        Number of items to choose
        
    Returns
    -------
    int
        Number of combinations (n choose r)
    """
    if n == r or r == 0:
        return 1
    return combination(n - 1, r - 1) + combination(n - 1, r)


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Constrain a value between minimum and maximum bounds.
    
    This function ensures that the input value does not exceed the specified
    minimum and maximum bounds.
    
    Parameters
    ----------
    value : float
        Value to constrain
    min_value : float
        Minimum allowed value
    max_value : float
        Maximum allowed value
        
    Returns
    -------
    float
        Constrained value between min_value and max_value
    """
    return max(min_value, min(value, max_value))


def normalize_angle(angle: float) -> float:
    """
    Normalize an angle to the range [-π, π].
    
    This function takes any angle value and returns an equivalent angle
    within the range of -π to π radians.
    
    Parameters
    ----------
    angle : float
        Angle to normalize (in radians)
        
    Returns
    -------
    float
        Normalized angle in the range [-π, π]
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi
