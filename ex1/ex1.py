import numpy as np
import regression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def loadData(filename):
    """
    parameters:
        filename: string from which to read CSV data

    returns: (<float> X, <float> y)
        X: 2D matrix where each row is an example and each column is a feature
        y: column vector containing the labeled outcomes

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


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    linreg = regression.Regression()

    # Part 1.2: Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    filename = 'ex1data1.txt'
    X, y = loadData(filename)

    ax.plot(np.delete(X, 0, 1), y, 'rx', markersize=10, label=r'Training data')
    ax.set_ylabel(r'Profit in \$10,000s')
    ax.set_xlabel(r'Population of City in 10,000s')

    # Part 1.3: Cost and Gradient Descent
    print('========== Part 1.3: Population Data (GD) ==========')
    theta = np.zeros((X.shape[1], 1))
    optimal = np.array([[-3.6303, 1.1664]]).T
    iterations = 1500
    alpha = 0.01

    J = linreg.computeCost(X, y, theta)
    print('Initial cost for population data: {}'.format(J))
    theta, J = linreg.gradientDescent(X, y, theta, alpha, iterations)
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
    J = np.array([[linreg.computeCost(X, y, np.block([[theta_0[i]], [theta_1[j]]]))
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
    X, mu, sigma = linreg.featureNormalize(X)

    # Part 2.2: Gradient Descent
    print('========== Part 2.2: Housing Data (GD) ==========')
    theta = np.zeros((X.shape[1], 1))
    iterations = 1500
    alpha = 0.01

    J = linreg.computeCost(X, y, theta)
    print('Initial cost for housing data: {}'.format(J))
    theta, J = linreg.gradientDescent(X, y, theta, alpha, iterations)

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
    theta = linreg.normalEquation(X, y)
    print('Final cost for housing data: {}'.format(
        linreg.computeCost(X, y, theta)))
    print('Hypothesis for housing data using normal equation: [{0:0.4f}, {1:0.4f}, {2:0.4f}]'.format(
        theta[0, 0], theta[1, 0], theta[2, 0]))
