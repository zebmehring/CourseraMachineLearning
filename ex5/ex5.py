import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from linearRegression import LinearRegression


def learning_curve(X, y, X_val, y_val, l=0, randomize=False, iters=1):
    linreg = LinearRegression(X, y, l)
    linreg_val = LinearRegression(X_val, y_val, 0)
    X = linreg.poly_fit(8)
    X = linreg.feature_normalize()
    X_val = linreg_val.poly_fit(8)
    X_val = linreg_val.feature_normalize()
    m_val = X_val.shape[0]

    error_train = np.zeros(linreg.m)
    error_val = np.zeros(linreg.m)

    for n in range(linreg.m):
        e_t = np.zeros(iters)
        e_v = np.zeros(iters)
        for i in range(iters):
            if randomize:
                train_i = [np.random.randint(linreg.m) for _ in range(n + 1)]
                val_i = [np.random.randint(linreg_val.m) for _ in range(n + 1)]
            else:
                train_i = np.arange(n + 1)
                val_i = np.arange(m_val)
            linreg.update_training_set(X[train_i, :], y[train_i])
            linreg_val.update_training_set(X_val[val_i, :], y_val[val_i])
            _, theta = linreg.optimize()
            e_t[i] = linreg.cost(theta, l=0)
            e_v[i] = linreg_val.cost(theta, l=0)
        error_train[n] = np.mean(e_t)
        error_val[n] = np.mean(e_v)

    return error_train, error_val


def validation_curve(X, y, X_val, y_val, lambdas):
    linreg = LinearRegression(X, y)
    linreg_val = LinearRegression(X_val, y_val)
    X = linreg.poly_fit(8)
    X = linreg.feature_normalize()
    X_val = linreg_val.poly_fit(8)
    X_val = linreg_val.feature_normalize()

    error_train = np.zeros(len(lambdas))
    error_val = np.zeros(len(lambdas))

    for i in range(len(lambdas)):
        linreg.update_lambda(lambdas[i])
        _, theta = linreg.optimize()
        error_train[i] = linreg.cost(theta, l=0)
        error_val[i] = linreg_val.cost(theta, l=0)

    return error_train, error_val


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # ==================== Part 1.1 Dataset ====================
    data = loadmat('ex5data1.mat')
    X, y = (data['X'], data['y'])
    X_val, y_val = (data['Xval'], data['yval'])
    X_test, y_test = (data['Xtest'], data['ytest'])
    linreg = LinearRegression(X, y, 0)
    linreg_val = LinearRegression(X_val, y_val, 0)
    linreg_test = LinearRegression(X_test, y_test, 0)
    regs = [linreg, linreg_val, linreg_test]

    # ========== Part 1.2 Regularized Linear Regression Cost ==========
    theta = np.array([1.0, 1.0])
    linreg.update_lambda(1)
    J = linreg.cost(theta)
    assert np.round(J, decimals=3) == 303.993

    # ========== Part 1.3 Regularized Linear Regression Gradient ==========
    grad = linreg.cost_grad(theta)
    assert np.array_equal(np.round(grad, decimals=3),
                          np.array([-15.303, 598.251]))

    # ========== Part 1.3 Training Linear Regression ==========
    linreg.update_lambda(0)
    J, theta = linreg.optimize()
    J_star, theta_star = linreg.normal_equation()
    theta = [np.round(t, decimals=3) for t in theta]
    theta_star = [np.round(t, decimals=3) for t in theta_star]
    assert np.array_equal(theta, theta_star)

    plt.plot(X, y, 'rx')
    plt.plot(X, linreg.predict(theta), 'b-', label=r'$\lambda = 0$')
    plt.xlabel('Change in water level')
    plt.ylabel('Outgoing water flow')
    plt.legend()
    plt.show()

    # ========== Part 2.1 Learning Curves (Linear Regression) ==========
    error_train, error_val = learning_curve(X, y, X_val, y_val, l=0)

    plt.plot(np.arange(error_val.size), error_train, label='Training error')
    plt.plot(np.arange(error_val.size), error_val, label='Validation error')
    plt.xlabel('Number of training examples')
    plt.ylabel(r'$J(\theta)$')
    plt.legend()
    plt.show()

    # ==================== Part 2.2 Feature Mapping ====================
    for r in regs:
        r.poly_fit(8)
        r.feature_normalize()

    # ========== Part 3.1-3.2 Learning Curves (Polynomial Regression) ==========
    lambdas = [0, 1, 100]
    for l in lambdas:
        for r in regs:
            r.update_lambda(l)
        J, theta = linreg.optimize()
        x = np.arange(min(X) - 15, max(X) + 25, 0.05)
        z = np.polyfit(X.flatten(), linreg.predict(theta), 8)
        f = np.poly1d(z)

        plt.plot(X, y, 'rx', label='Actual value')
        plt.plot(X, linreg.predict(theta), 'g+', label='Predicted value')
        plt.plot(x, f(x), 'b--',
                 label=r'Predicted trend for $p=8,\ \lambda={0}$'.format(l))
        plt.xlabel('Change in water level')
        plt.ylabel('Outgoing water flow')
        plt.legend()
        plt.show()

        error_train, error_val = learning_curve(X, y, X_val, y_val, l)
        print('================== lambda = {0} =================='.format(l))
        print('# of Examples\tTrain Error\tValidation Error')
        for i in range(linreg.m):
            print('{0}\t\t{1:.6f}\t{2:.6f}'.format(
                i + 1, error_train[i], error_val[i]))
        print('')

        plt.plot(np.arange(error_val.size),
                 error_train, label='Training error')
        plt.plot(np.arange(error_val.size),
                 error_val, label='Validation error')
        plt.xlabel('Number of training examples')
        plt.ylabel(r'$J(\theta)$')
        plt.legend()
        plt.show()

    # ==================== Part 3.3 Selecting Lambda ====================
    lambdas = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    error_train, error_val = validation_curve(X, y, X_val, y_val, lambdas)
    print('============ Selecting Lambda ============')
    print('lambda\tTrain Error\tValidation Error')
    for i in range(len(lambdas)):
        print('{0:.3f}\t{1:.6f}\t{2:.6f}'.format(
            lambdas[i], error_train[i], error_val[i]))
    print('')

    plt.plot(np.arange(error_val.size), error_train, label='Training error')
    plt.plot(np.arange(error_val.size), error_val, label='Validation error')
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$J(\theta)$')
    plt.legend()
    plt.show()

    # ==================== Part 3.4 Test Set Error ====================
    for r in regs:
        r.update_lambda(3)
    J, theta = linreg.optimize()
    J_val = linreg_val.cost(theta)
    J_test = linreg_test.cost(theta)
    print('Cost on validation set: {0}'.format(J_val))
    print('Cost on test set: {0}'.format(J_test))

    # =============== Part 3.5 Randomized Learning Curves ===============
    error_train, error_val = learning_curve(
        X, y, X_val, y_val, l=0.01, randomize=True, iters=50)

    plt.plot(np.arange(error_val.size),
             error_train, label='Training error')
    plt.plot(np.arange(error_val.size),
             error_val, label='Validation error')
    plt.xlabel('Number of training examples')
    plt.ylabel(r'$J(\theta)$')
    plt.legend()
    plt.show()
