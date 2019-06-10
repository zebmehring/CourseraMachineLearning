import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from kmeans import KMeans

if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # ==================== Part 1.1 Finding Clusters ====================
    data = loadmat('ex7data2.mat')
    kmeans = KMeans(data['X'], 3)
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
    colors = kmeans.color(initial_centroids)
    assert np.array_equal(colors[0:3], np.array([0, 2, 1]))

    # ==================== Part 1.2 Finding Centroids ====================
    centroids = kmeans.compute_centroids(colors)
    expected_centroids = np.array([[2.428301, 3.157924],
                                   [5.813503, 2.633656],
                                   [7.119387, 3.616684]])
    assert np.allclose(centroids, expected_centroids)

    # ==================== Part 1.3 K-Means ====================
    kmeans.update_parameters(X=data['X'], k=3, max_iters=10)
    kmeans.cluster()
    expected_mu = np.array([[1.9540, 5.0256],
                            [6.0337, 3.0005],
                            [3.0437, 1.0154]])
    assert np.all([np.any([np.allclose(i, j, atol=1e-4) for j in kmeans.mu])
                   for i in expected_mu])

    # ==================== Part 1.4 K-Means for Pixels ====================
    X = plt.imread('bird_small.png')
    w, h, d = X.shape
    X = X.reshape(w * h, d)
    kmeans.update_parameters(X=X, k=16, iters=1, max_iters=10)
    kmeans.cluster()
    img = np.array([kmeans.mu[kmeans.c[i]] for i in range(kmeans.m)])
    img = img.reshape(w, h, d)
    f, axarr = plt.subplots(1, 2)
    axarr[0].set_axis_off()
    axarr[1].set_axis_off()
    axarr[0].imshow(X.reshape(w, h, d))
    axarr[1].imshow(img)
    plt.show()
