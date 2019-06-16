import numpy as np


def estimate_gaussian(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    return mu, sigma


def multivariate_gaussian(X, mu, sigma):
    from math import pi
    from numpy.linalg import pinv, det
    k = mu.size
    sigma = np.square(sigma)
    if sigma.ndim == 1 or sigma.shape[0] == 1 or sigma.shape[1] == 1:
        sigma = np.diag(sigma)
    X = X - mu
    p = np.power(2 * pi, -k / 2) * np.power(det(sigma), -0.5) * \
        np.exp(-0.5 * np.sum((X @ pinv(sigma)) * X, axis=1))
    return p


def select_threshold(y, p):
    epsilon = np.arange(min(p), max(p), (max(p) - min(p)) / 1000)
    predictions = np.array([np.where(p < e) for e in epsilon]).ravel()
    true_positives = [np.intersect1d(np.where(y == 1), pred).size
                      for pred in predictions]
    false_positives = [np.intersect1d(np.where(y == 0), pred).size
                       for pred in predictions]
    false_negatives = [np.setdiff1d(np.where(y == 1), pred).size
                       for pred in predictions]
    prec = np.array([tp / (tp + fp) if tp + fp > 0 else 0
                     for tp, fp in zip(true_positives, false_positives)])
    rec = np.array([tp / (tp + fn) if tp + fn > 0 else 0
                    for tp, fn in zip(true_positives, false_negatives)])
    F1 = [2 * (pr * re) / (pr + re) if pr + re > 0 else 0
          for pr, re in zip(prec, rec)]
    return epsilon[np.argmax(F1)], np.max(F1)
