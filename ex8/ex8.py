import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def estimate_gaussian(X):
    pass
    return mu, sigma


def multivariate_gaussian(X, mu, sigma):
    pass
    return p


def select_threshold(y, p):
    pass
    return epsilon, F1


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # ==================== Part 1.1 Dataset ====================
    data = loadmat('ex8data1.mat')
    X, X_val, y_val = (data['X'], data['Xval'], data['yval'])
    mu, sigma = estimate_gaussian(X)
    p = multivariate_gaussian(X, mu, sigma)

    X1, X2 = np.meshgrid(np.arange(0, 35, 0.5))
    Z = multivariate_gaussian(np.c_[X1, X2], mu, sigma)
    Z = Z.reshape(X1.shape)
    plt.plot(X[:, 0], X[:, 1], 'bx')
    if not np.any(np.isinf(Z)):
        plt.contour(X1, X2, Z, np.power(10.0, np.arange(-20, 0, 3)))
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (Mb/s)')
    plt.show(True)

    p_val = multivariate_gaussian(X_val, mu, sigma)
    epsilon, F1 = select_threshold(y_val, p_val)
    assert np.allclose(epsilon, 8.99e-5) and np.allclose(F1, 0.875)

    outliers = np.where(p < epsilon)
    plt.plot(X[outliers, 0], X[outliers, 1], 'ro')
    plt.show()

    data = loadmat('ex8data2.mat')
    X, X_val, y_val = (data['X'], data['Xval'], data['yval'])
    mu, sigma = estimate_gaussian(X)
    p = multivariate_gaussian(X, mu, sigma)
    p_val = multivariate_gaussian(X_val, mu, sigma)
    epsilon, F1 = select_threshold(y_val, p_val)
    assert np.allclose(epsilon, 1.38e-18) and np.allclose(F1, 0.615385)
    assert sum(np.where(p < epsilon, 1, 0)) == 1
