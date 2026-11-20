import numpy as np

def reshape_array(data: list, operation: str) -> np.ndarray:
    """
    Returns a float64 array with the shape selected by operation.
    """
    np_data = np.array(data, dtype=np.float64)
    if operation=='flatten':
        return np_data.flatten()
    if operation == 'transpose':
        return np_data.T #np.transpose(data)
    if operation == 'add_batch':
        return np.reshape(np_data, shape=(1,*np_data.shape))
        
