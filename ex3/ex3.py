import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import logisticRegression


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    logreg = logisticRegression.LogisticRegression()

    # Part 1.1: Dataset
    n_labels = 10
    data = scipy.io.loadmat('ex3data1.mat')
    X, y = (data['X'], data['y'])
    X = np.block([np.ones((X.shape[0], 1)), X])
    y = np.where(y == 10, 0.0, y).reshape(y.shape[0])

    # Part 1.3: Vectorized Logistic Regression
    print('========== Part 1: One vs. All Logistic Regression ==========')
    theta_t = np.array([-2, -1, 1, 2])
    X_t = np.block([np.ones((5, 1)), np.reshape(
        np.arange(1, 16), (5, 3), 'F') / 10])
    y_t = np.where(np.array([1, 0, 1, 0, 1]) >= 0.5, 1, 0)
    lambda_t = 3
    J = logreg.cost(theta_t, X_t, y_t, lambda_t)
    grad_t = logreg.cost_grad(theta_t, X_t, y_t, lambda_t)

    print('Cost: {0:0.6f} (expected: {1:0.6f})'.format(J, 2.534819))
    print('Gradient: [{0:0.6f}, {1:0.6f}, {2:0.6f}, {3:0.6f}] (expected: [0.146561, -0.548558, 0.724722, 1.398003])'.format(
        grad_t[0], grad_t[1], grad_t[2], grad_t[3]))

    # Part 1.4: One vs. All Classification
    reg = 0.1
    theta = None
    theta_0 = np.zeros(X.shape[1])

    for i in range(n_labels):
        J, theta_n = logreg.optimize(theta_0, X, np.where(y == i, 1, 0), reg)
        if theta is None:
            theta = theta_n
        else:
            theta = np.block([[theta], [theta_n]])

    p = logreg.predict(theta, X)
    print('Training set accuracy: {}'.format(np.mean(p == y) * 100))

    # Part 2.2: Feedforward Propagation and Prediction
    print('========== Part 2: Neural Network Classification ==========')
    layer_dims = [400, 25, 10]
    data = scipy.io.loadmat('ex3weights.mat')
    Theta_1, Theta_2 = (data['Theta1'], data['Theta2'])

    # Part 3.3: NN Predictions
    p = logreg.predict_nn(X, [Theta_1, Theta_2], n_labels)
    print('Training set accuracy: {}'.format(np.mean(p == y) * 100))
