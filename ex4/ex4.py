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
    y = np.where(y == 10, 0, y).reshape(y.shape[0])
    data = scipy.io.loadmat('ex4weights.mat')
    Theta_1, Theta_2 = (data['Theta1'], data['Theta2'])

    J, grads = nn.cost(X, y, [Theta_1, Theta_2], layer_dims)
    print('Cost: {0:0.6f} (expected: {1:0.6f})'.format(J, 0.287629))

    J, grads = nn.cost(X, y, [Theta_1, Theta_2], layer_dims, 1)
    print('Regularized cost: {0:0.6f} (expected: {1:0.6f})'.format(
        J, 0.383770))

    g = nn.sigmoid_gradient(np.array([-1.0, -0.5, 0, 0.5, 1.0]))
    print('Sigmoid gradient: [{0:0.6f}, {1:0.6f}, {2:0.6f}, {3:0.6f}, {4:0.6f}] (expected: [0.19661193, 0.23500371, 0.250000, 0.23500371, 0.19661193])'.format(
        g[0], g[1], g[2], g[3], g[4]))
