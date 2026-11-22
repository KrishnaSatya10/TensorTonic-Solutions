import numpy as np

def kmeans(X: list, k: int, max_iters: int = 100, seed: int = 42) -> tuple:
    """
    Returns cluster labels and the final centroid matrix.
    """
    X=np.asarray(X,dtype=np.float64)
    rng=np.random.RandomState(seed)
    centroids=X[rng.choice(len(X),size=k,replace=False)].copy()


    #Project the features of a point on z dimension for both X and centroids, and then take the euclidean distance between points. (n_points, n_centroids)
    for _ in range(max_iters):
        distances = np.sum((X[:,None,:]-centroids[None,:,:])**2, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = []
        for cluster in range(k):
            #Get the datapoints which fall in that cluster:
            data_points = X[np.where(labels==cluster)[0].tolist()]
            if len(data_points) != 0:
                new_centroid = np.mean(data_points,axis=0)
                new_centroids.append(new_centroid)

            else:
            # Empty cluster: keep its previous centroid
                new_centroids.append(centroids[cluster])

        centroids = np.array(new_centroids)

    #Final labels:
    distances = np.sum((X[:,None,:]-centroids[None,:,:])**2, axis=2)
    labels = np.argmin(distances, axis=1)

    return (labels,centroids)
    
