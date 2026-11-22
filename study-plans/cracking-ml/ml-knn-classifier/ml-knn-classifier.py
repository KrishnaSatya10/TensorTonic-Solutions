import numpy as np
from collections import Counter

def knn_classifier(X_train: list, y_train: list, X_test: list, k: int) -> list:
    """
    Returns one predicted label for each test row.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    y_train = np.array(y_train)
    predictions = []
    for point in X_test:
        dist = np.sum((X_train-point) ** 2, axis=-1)
        sorted_index_k = np.argsort(dist, axis=-1, kind="stable")
        top_k_nearest_labels = y_train[sorted_index_k[:k]]
        y_preds_k=Counter(top_k_nearest_labels)
        label = sorted(y_preds_k.items(), key=lambda x: (-x[1],x[0]))[0][0]
        predictions.append(label.item())

    return predictions
        
        
        

        
