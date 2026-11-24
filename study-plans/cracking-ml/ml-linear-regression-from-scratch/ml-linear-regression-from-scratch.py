import numpy as np

def linear_regression_from_scratch(X: list, y: list, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.asarray(X, dtype=np.float64) #Shape (4,10)
    y = np.asarray(y, dtype=np.float64) #Shape (4,)
    weights = np.zeros(X.shape[1], dtype=np.float64) #Shape(10,)
    bias = 0.0

    for e in range(epochs):
        y_pred = X @ weights + bias
        error = y - y_pred #Shape(4,)

        #Grad calculation
        weights_grad = (-2.0 / len(y) ) * (X.T @ error)  #Shape(10,)
        bias_grad = (-2.0 / len(y) ) * np.sum(error) 
        
        #Weight and bias update
        weights -= lr * weights_grad
        bias -= lr * bias_grad

    return [weights, bias]
