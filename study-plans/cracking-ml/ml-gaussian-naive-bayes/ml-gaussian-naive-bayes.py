import numpy as np
import math

def gaussian_naive_bayes(X_train: list, y_train: list, X_test: list) -> list:
    """
    Returns one predicted label for each test row.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=np.float64)
    classes = np.unique(y_train)
    means = []
    vars = []
    priors = []

    #Fit part.
    #mean, var, prior for every class. 
    for c in classes:
        data_points = X_train[y_train==c]
        prior = len(data_points)/ len(X_train)
        mean = np.mean(data_points,axis=0)
        var = np.var(data_points, ddof=0, axis=0) + 1e-9
        means.append(mean)
        vars.append(var)
        priors.append(prior)

    #Predict part. 
    labels = [] #Same length as X_test
    for x in X_test:
        class_wise_scores = pdf(x,means,vars,priors)
        label = classes[np.argmax(class_wise_scores)]

        labels.append(label)
        

    return labels


def pdf(x, means, vars, priors):

    class_wise_posterior = []
    for mean,var,prior in zip(means,vars,priors):
        num = (x-mean)**2/ (var)
        den = 2 * np.pi * var
        posterior = - 0.5 * np.sum(np.log(den) + num) + np.log(prior)
        class_wise_posterior.append(posterior)
        
    return class_wise_posterior
        