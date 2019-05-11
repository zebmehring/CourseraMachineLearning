import numpy as np
import matplotlib.pyplot as plt


def loadData(filename):
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
    return np.block([np.ones((np.shape(data)[0], 1)), data[:, :-1]]), data[:, -1][:, np.newaxis]


def computeCost(X, y, theta):
    m = len(y)
    error = np.dot(X, theta) - y
    J = np.reciprocal(2.0 * m) * np.dot(np.transpose(error), error)
    return J.flatten()[0]


def featureNormalize(X):
    pass


def gradientDescent(X, y, theta, alpha, iterations):
    pass


def normalEquation(X, y):
    pass


if __name__ == '__main__':
    iterations = 1500
    alpha = 0.01
    filename = 'ex1data2.txt'

    X, y = loadData(filename)
    theta = np.zeros((np.shape(X)[1], 1))
    J = computeCost(X, y, theta)
    print('Initial cost: {}'.format(J))
