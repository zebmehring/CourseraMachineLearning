import numpy as np
import matplotlib.pyplot as plt
import logisticRegression


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
    return np.block([np.ones((data.shape[0], 1)), data[:, :-1]]), np.array([data[:, -1]]).T


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    logreg = logisticRegression.LogisticRegression()
