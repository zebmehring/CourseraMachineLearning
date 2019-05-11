import numpy as np
import matplotlib.pyplot as plt


def loadData(filename):
    """
    parameters:
        filename: string from which to read CSV data

    returns: (<float> X, <float> y)
        X: a numpy 2D matrix where each row is an example and each column is a feature
        y: a numpy column vector containing the labeled outcomes

    Load CSV data from a file into data structures for a regression problem.
    """
    try:
        f = open(filename, 'r')
    except FileNotFoundError:
        print('{} not found'.format(filename))
        return []
    data = None
    for line in f:
        ex = np.array([float(i) for i in line.split(',')])
        if data is not None:
            data = np.block([[data], [ex]])
        else:
            data = ex
    f.close()
    return np.block([np.ones((data.shape[0], 1)), data[:, :-1]]), np.array([data[:, -1]]).T


def computeCost(X, y, theta):
    """
    parameters:
        X: a numpy 2D matrix where each row is an example and each column is a feature
        y: a numpy column vector containing the labeled outcomes
        theta: a numpy column vector containing the hypothesis

    returns: <float> J
        J: MSE cost for the training batch X with the hypothesis theta

    Compute the mean-squared-error for the data in X with the hypothesis theta.
    """
    m = len(y)
    error = np.dot(X, theta) - y
    J = np.reciprocal(2.0 * m) * np.dot(error.T, error)
    return J.flatten()[0]


def featureNormalize(X):
    pass


def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    J = np.zeros((iterations, 1))
    for i in range(iterations):
        grad = np.dot(X.T, (np.dot(X, theta) - y))
        theta -= (alpha / m) * grad
        J[i] = computeCost(X, y, theta)
    return theta, J


def normalEquation(X, y):
    pass


def polyFit(x, n):
    X = np.ones(x.shape)
    for i in range(1, n):
        X = np.block([[X, np.power(x, i)]])
    return X


def plotData(x, y, ax):
    ax.plot(x, y, 'rx', markersize=10, label=r'Training data')
    ax.set_ylabel(r'Profit in \$10,000s')
    ax.set_xlabel(r'Population of City in 10,000s')
    return ax


if __name__ == '__main__':
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    f, ax = plt.subplots()

    # Part 2: Plotting
    filename = 'ex1data1.txt'
    X, y = loadData(filename)
    m = len(y)
    ax = plotData(np.delete(X, 0, 1), y, ax)

    # Part 3: Cost and Gradient Descent
    theta = np.zeros((X.shape[1], 1))
    optimal = np.array([[-3.6303, 1.1664]]).T
    iterations = 1500
    alpha = 0.01
    J = computeCost(X, y, theta)
    print('Initial cost for population data: {}'.format(J))
    theta, J = gradientDescent(X, y, theta, alpha, iterations)
    print('Final cost for population data: {}'.format(J[-1][0]))
    print('Hypothesis for population data: [{0:0.4f}, {1:0.4f}] (expected: [{2}, {3}])'.format(
        theta[0, 0], theta[1, 0], optimal[0, 0], optimal[1, 0]))
    ax.plot(X[:, -1], np.dot(X, theta), 'b-', label='Linear regression')
    ax.legend()
    plt.show()
    f.clf()

    X, y = loadData(filename)
    theta = np.zeros((X.shape[1], 1))
    J = computeCost(X, y, theta)
    print('Initial cost for housing data: {}'.format(J))
