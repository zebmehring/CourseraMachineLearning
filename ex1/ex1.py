import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
    X = X[:, 1:]
    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X_norm = np.block([np.ones((X.shape[0], 1)), (X - mu) / sigma])
    return X_norm, mu, sigma


def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    J = np.zeros((iterations, 1))
    for i in range(iterations):
        grad = np.dot(X.T, (np.dot(X, theta) - y))
        theta -= (alpha / m) * grad
        J[i] = computeCost(X, y, theta)
    return theta, J


def normalEquation(X, y):
    return np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)


def polyFit(x, n):
    X = np.ones(x.shape)
    for i in range(1, n + 1):
        X = np.block([[X, np.power(x, i)]])
    return X


def plotData(x, y, ax):
    ax.plot(x, y, 'rx', markersize=10, label=r'Training data')
    ax.set_ylabel(r'Profit in \$10,000s')
    ax.set_xlabel(r'Population of City in 10,000s')
    return ax


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Part 1.2: Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    filename = 'ex1data1.txt'
    X, y = loadData(filename)
    m = len(y)
    ax = plotData(np.delete(X, 0, 1), y, ax)

    # Part 1.3: Cost and Gradient Descent
    print('========== Part 1.3: Population Data (GD) ==========')
    theta = np.zeros((X.shape[1], 1))
    optimal = np.array([[-3.6303, 1.1664]]).T
    iterations = 150
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
    fig.clf()

    # Part 1.4: Visualizing J
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    theta_0 = np.linspace(-10, 10, num=100)
    theta_1 = np.linspace(-1, 4, num=100)
    J = np.array([[computeCost(X, y, np.block([[theta_0[i]], [theta_1[j]]]))
                   for j in range(len(theta_1))] for i in range(len(theta_0))]).T

    ax.plot_surface(theta_0, theta_1, J, cmap=plt.cm.jet)
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
    ax.set_zlabel(r'$J(\theta)$')
    plt.show()
    fig.clf()

    plt.contour(theta_0, theta_1, J, np.logspace(-2, 3, 20))
    plt.plot(theta[0], theta[1], 'rx', label='Regression hypothesis')
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    plt.legend()
    plt.show()

    # Part 2.1: Feature Normalization
    filename = 'ex1data2.txt'
    X, y = loadData(filename)
    m = len(y)
    X, mu, sigma = featureNormalize(X)

    # Part 2.2: Gradient Descent
    print('========== Part 2.2: Housing Data (GD) ==========')
    theta = np.zeros((X.shape[1], 1))
    iterations = 1500
    alpha = 0.01
    J = computeCost(X, y, theta)
    print('Initial cost for housing data: {}'.format(J))
    theta, J = gradientDescent(X, y, theta, alpha, iterations)
    plt.plot(np.arange(iterations), J, 'b-', label='Cost')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.show()
    print('Final cost for housing data: {}'.format(J[-1][0]))
    print('Hypothesis for housing data using gradient descent: [{0:0.4f}, {1:0.4f}, {2:0.4f}]'.format(
        theta[0, 0], theta[1, 0], theta[2, 0]))

    # Part 2.3: Normal Equation
    print('========== Part 2.3: Housing Data (NE) ==========')
    X, y = loadData(filename)
    theta = normalEquation(X, y)
    print('Final cost for housing data: {}'.format(computeCost(X, y, theta)))
    print('Hypothesis for housing data using normal equation: [{0:0.4f}, {1:0.4f}, {2:0.4f}]'.format(
        theta[0, 0], theta[1, 0], theta[2, 0]))
