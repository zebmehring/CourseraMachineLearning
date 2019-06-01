import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import neuralNetwork


if __name__ == '__main__':
    layer_dims = [400, 25, 10]
    data = scipy.io.loadmat('ex4data1.mat')
    X, y = (data['X'], data['y'])
    y = np.where(y == 10, 0, y).reshape(y.size).astype(float)
    data = scipy.io.loadmat('ex4weights.mat')
    Theta = [data['Theta1'], data['Theta2']]
    theta = np.block([t.reshape(t.size, order='F') for t in Theta])
    nn = neuralNetwork.NeuralNetwork(X, y, Theta, layer_dims, 0)

    print('========== Part 1.3: Feedforward and Cost Function ==========')
    J = nn.cost(theta)
    print('Cost: {0:0.6f} (expected: 0.287629)'.format(J))

    print('========== Part 1.4: Regularized Cost Function ==========')
    nn.update_lambda(1)
    J = nn.cost(theta)
    print('Regularized cost: {0:0.6f} (expected: 0.383770)'.format(J))

    print('========== Part 2.1: Sigmoid Gradient ==========')
    g = nn.sigmoid_gradient(np.array([-1.0, -0.5, 0, 0.5, 1.0]))
    e_g = [0.19661193, 0.23500371, 0.250000, 0.23500371, 0.19661193]
    print('---------------------')
    print('   Actual |  Expected')
    print('---------------------')
    for i, j in zip(g, e_g):
        print('{0: .6f} | {1: .6f}'.format(i, j))
    diff = np.linalg.norm(e_g - g) / np.linalg.norm(e_g + g)
    print('Relative difference: {0}'.format(diff))

    print('========== Part 2.4: Gradient Checking ==========')
    sample_dims = [3, 5, 3]
    sample_m = 5
    X_s = nn.initialize_weights(sample_dims[0] - 1, sample_m, None, debug=True)
    y_s = 1 + (np.arange(1, sample_m + 1) % sample_dims[-1])
    Theta = [nn.initialize_weights(sample_dims[i], sample_dims[i + 1], None, debug=True)
             for i in range(len(sample_dims) - 1)]
    theta = np.block([t.reshape(t.size, order='F') for t in Theta])
    sample_nn = neuralNetwork.NeuralNetwork(X_s, y_s, Theta, sample_dims)

    grads = sample_nn.cost_grad(theta)
    n_grads = sample_nn.cost_grad_numerical(theta)
    print('----------------------')
    print('Analytical | Numerical')
    print('----------------------')
    for g, n_g in zip(grads, n_grads):
        print('   {0: .4f} |   {1: .4f}'.format(g, n_g))
    diff = np.linalg.norm(n_grads - grads) / np.linalg.norm(n_grads + grads)
    print('Relative difference: {0}'.format(diff))

    print('========== Part 2.4: Regularized Neural Networks ==========')
    Theta = [data['Theta1'], data['Theta2']]
    theta = np.block([t.reshape(t.size, order='F') for t in Theta])
    nn.update_lambda(3)
    J = nn.cost(theta)
    print('Regularized cost: {0:0.6f} (expected: 0.576051)'.format(J))

    print('========== Part 2.5: Learning Parameters ==========')
    Theta = [nn.initialize_weights(layer_dims[i], layer_dims[i + 1], 0.12)
             for i in range(len(layer_dims) - 1)]
    theta = np.block([t.reshape(t.size, order='F') for t in Theta])
    nn.update_lambda(1)
    J, theta = nn.optimize(theta)
    p = nn.predict(theta)
    print('Trained accuracy: {}'.format(np.mean(p == y) * 100))
