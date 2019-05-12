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
    return np.block([np.ones((data.shape[0], 1)), data[:, :-1]]), data[:, -1]


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    logreg = logisticRegression.LogisticRegression()

    # Part 1.1: Plotting
    filename = 'ex2data1.txt'
    X, y = loadData(filename)

    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.plot(X[np.where(y == 1)[0], 1], X[np.where(y == 1)[0], 2],
             'k+', markersize=7, label=r'Admitted')
    plt.plot(X[np.where(y == 0)[0], 1], X[np.where(y == 0)[0], 2],
             'ko', markersize=7, label=r'Not admitted')

    # Part 1.2: Compute Cost and Gradient
    print('========== Parts 1.2-1.3: Cost, Gradient, and Optimization ==========')
    theta = np.zeros((X.shape[1],))
    J = logreg.cost(theta, X, y)
    print('Initial cost for exam data: {0:.3f} (expected 0.693)'.format(J))
    grad = logreg.cost_grad(theta, X, y)
    print('Initial gradient for exam data: [{0:.4f}, {1:0.4f}, {2:0.4f}] (expected [-0.1000, -12.0092, -11.2628])'.format(
        grad[0], grad[1], grad[2]))

    # Part 1.3: Optimization
    J, theta = logreg.optimize(theta, X, y)
    print('Final cost for exam data: {0:.3f} (expected 0.203)'.format(J))
    print('Hypothesis for exam data: [{0:.3f}, {1:0.3f}, {2:0.3f}] (expected [-25.161, 0.206, 0.201])'.format(
        theta[0], theta[1], theta[2]))

    plot_x = np.array([min(X[:, 1]) - 2, max(X[:, 1]) + 2])
    plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])
    plt.plot(plot_x, plot_y, 'k-', label='Decision boundary')
    plt.legend()
    plt.show()
    plt.clf()
