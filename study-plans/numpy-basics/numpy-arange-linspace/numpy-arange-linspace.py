import numpy as np

def create_sequence(start: float, stop: float, param: float, kind: str) -> np.ndarray:
    """
    Returns a 1D float64 array containing the requested sequence.
    """
    if kind == "arange": #Param is step size here. 
        return np.arange(start, stop, param, dtype = np.float64)
    else:
        return np.linspace(start, stop, param, dtype = np.float64) #Param divides the interval (start, stop) in certain number of chunks. 
