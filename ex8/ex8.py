import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from anomalyDetection import *
from recommenderSystem import *

if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # ==================== Part 1.1 Dataset ====================
    data = loadmat('ex8data1.mat')
    X, X_val, y_val = (data['X'], data['Xval'], data['yval'].ravel())

    # ==================== Part 1.2 Estimating Parameters ====================
    mu, sigma = estimate_gaussian(X)
    p = multivariate_gaussian(X, mu, sigma)

    X1, X2 = np.meshgrid(np.arange(0, 35, 0.5), np.arange(0, 35, 0.5))
    Z = multivariate_gaussian(np.c_[X1.ravel('F'), X2.ravel('F')], mu, sigma)
    Z = Z.reshape(X1.shape)
    plt.plot(X[:, 0], X[:, 1], 'bx')
    if not np.any(np.isinf(Z)):
        plt.contour(X1, X2, Z, np.power(10.0, np.arange(-20, 0, 3)))
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (Mb/s)')

    # ==================== Part 1.3 Estimating Epsilon ====================
    p_val = multivariate_gaussian(X_val, mu, sigma)
    epsilon, F1 = select_threshold(y_val, p_val)
    assert np.allclose(epsilon, 8.99e-5) and np.allclose(F1, 0.875)

    outliers = np.where(p < epsilon)
    plt.plot(X[outliers, 0], X[outliers, 1], 'ro')
    plt.show()

    # ==================== Part 1.4 High-Dimensional Dataset ====================
    data = loadmat('ex8data2.mat')
    X, X_val, y_val = (data['X'], data['Xval'], data['yval'].ravel())
    mu, sigma = estimate_gaussian(X)
    p = multivariate_gaussian(X, mu, sigma)
    p_val = multivariate_gaussian(X_val, mu, sigma)
    epsilon, F1 = select_threshold(y_val, p_val)
    assert np.allclose(epsilon, 1.38e-18) and np.allclose(F1, 0.615385)
    assert np.where(p < epsilon)[0].size == 117

    # ==================== Part 2.1 Dataset ====================
    data = loadmat('ex8_movies.mat')
    Y, R = (data['Y'], data['R'])

    # ==================== Part 2.2 Collaborative Filtering ====================
    data = loadmat('ex8_movieParams.mat')
    X, Theta = (data['X'], data['Theta'])
    n_movies, n_features = X.shape
    n_users, _ = Theta.shape
