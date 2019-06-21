import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from math import isclose
from anomalyDetection import *
from recommenderSystem import CollaborativeFiltering

if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # ==================== Part 1.1 Dataset ====================
    data = loadmat('ex8data1.mat')
    X, X_val, y_val = (data['X'], data['Xval'], data['yval'].ravel())

    # ==================== Part 1.2 Estimating Parameters ====================
    mu, sigma, Sigma = estimate_gaussian(X)
    p = multivariate_gaussian(X, mu, Sigma)

    X1, X2 = np.meshgrid(np.arange(0, 35, 0.5), np.arange(0, 35, 0.5))
    Z = multivariate_gaussian(np.c_[X1.ravel('F'), X2.ravel('F')], mu, Sigma)
    Z = Z.reshape(X1.shape)
    plt.plot(X[:, 0], X[:, 1], 'bx')
    if not np.any(np.isinf(Z)):
        plt.contour(X1, X2, Z, np.power(10.0, np.arange(-20, 0, 3)))
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (Mb/s)')

    # ==================== Part 1.3 Estimating Epsilon ====================
    p_val = multivariate_gaussian(X_val, mu, Sigma)
    epsilon, F1 = select_threshold(y_val, p_val)
    assert isclose(epsilon, 8.99e-5, rel_tol=1e-2)
    assert isclose(F1, 0.875)

    outliers = np.where(p < epsilon)
    plt.plot(X[outliers, 0], X[outliers, 1], 'ro')
    plt.show()

    # ==================== Part 1.4 High-Dimensional Dataset ====================
    data = loadmat('ex8data2.mat')
    X, X_val, y_val = (data['X'], data['Xval'], data['yval'].ravel())
    mu, sigma, Sigma = estimate_gaussian(X)
    p = multivariate_gaussian(X, mu, Sigma)
    p_val = multivariate_gaussian(X_val, mu, Sigma)
    epsilon, F1 = select_threshold(y_val, p_val)
    assert isclose(epsilon, 1.38e-18, rel_tol=1)
    assert isclose(F1, 0.615385, rel_tol=1)
    assert isclose(np.where(p < epsilon)[0].size, 117, abs_tol=5)

    # ==================== Part 2.1 Dataset ====================
    data = loadmat('ex8_movies.mat')
    Y, R = (data['Y'], data['R'])
    data = loadmat('ex8_movieParams.mat')
    X, Theta = (data['X'], data['Theta'])
    cofi = CollaborativeFiltering()

    # ==================== Part 2.2 Collaborative Filtering ====================
    n_users, n_movies, n_features = (4, 5, 3)
    X_t, Theta_t = (X[:n_movies, :n_features], Theta[:n_users, :n_features])
    Y_t, R_t = (Y[:n_movies, :n_users], R[:n_movies, :n_users])
    theta = np.block([X_t.flatten('F'), Theta_t.flatten('F')])
    params = {'X.size': X_t.size, 'T.size': Theta_t.size,
              'X.shape': X_t.shape, 'T.shape': Theta_t.shape}
    J = cofi.cost(theta, Y_t, R_t, params)
    assert isclose(J, 22.22, rel_tol=1e-3)

    assert cofi.check_grad()

    J = cofi.cost(theta, Y_t, R_t, params, reg=1.5)
    assert isclose(J, 31.34, rel_tol=1e-3)

    assert cofi.check_grad(1.25)

    J, X, Theta = cofi.optimize(X, Theta, Y, R, debug=True)
