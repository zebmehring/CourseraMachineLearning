import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import neuralNetwork


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    nn = neuralNetwork.NeuralNetwork()

    layer_dims = [400, 25, 10]
    data = scipy.io.loadmat('ex4data1.mat')
    X, y = (data['X'], data['y'])
    m = y.shape[0]
    y = np.where(y == 10, 0, y).reshape(m)
    data = scipy.io.loadmat('ex4weights.mat')
    Theta = [data['Theta1'], data['Theta2']]
    theta = np.block([t.reshape(t.size, order='F') for t in Theta])

    J = nn.cost(theta, X, y, layer_dims)
    print('Cost: {0:0.6f} (expected: 0.287629)'.format(J))

    J = nn.cost(theta, X, y, layer_dims, 1)
    print('Regularized cost: {0:0.6f} (expected: 0.383770)'.format(J))

    g = nn.sigmoid_gradient(np.array([-1.0, -0.5, 0, 0.5, 1.0]))
    print('Sigmoid gradient: [{0:0.6f}, {1:0.6f}, {2:0.6f}, {3:0.6f}, {4:0.6f}] (expected: [0.19661193, 0.23500371, 0.250000, 0.23500371, 0.19661193])'.format(
        g[0], g[1], g[2], g[3], g[4]))

    sample_dims = [3, 5, 3]
    sample_m = 5
    X_sample = nn.initialize_weights(
        sample_dims[0] - 1, sample_m, None, debug=True)
    y_sample = 1 + (np.arange(1, sample_m + 1) % sample_dims[-1])
    Theta = [nn.initialize_weights(sample_dims[i], sample_dims[i + 1], None, debug=True)
             for i in range(len(sample_dims) - 1)]
    theta = np.block([t.reshape(t.size, order='F') for t in Theta])
    grads = nn.cost_grad(theta, X_sample, y_sample, sample_dims)
    n_grads = nn.cost_grad_numerical(theta, X_sample, y_sample, sample_dims)
    print('Relative difference in analytic vs. numerical gradients: {0}'.format(
        np.linalg.norm(n_grads - grads) / np.linalg.norm(n_grads + grads)))

    Theta = [data['Theta1'], data['Theta2']]
    theta = np.block([t.reshape(t.size, order='F') for t in Theta])
    J = nn.cost(theta, X, y, layer_dims, 3)
    print('Regularized cost: {0:0.6f} (expected: 0.576051)'.format(J))
