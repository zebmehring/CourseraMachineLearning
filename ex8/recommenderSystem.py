import numpy as np


class CollaborativeFiltering:
    def extract_params(self, theta, params):
        X = np.reshape(theta[:params['X.size']], params['X.shape'], 'F')
        Theta = np.reshape(theta[params['X.size']:], params['T.shape'], 'F')
        return X, Theta

    def cost(self, theta, Y, R, params, reg=0):
        X, Theta = self.extract_params(theta, params)
        err = ((X @ Theta.T) * R) - (Y * R)
        r = reg * (theta @ theta.T)
        J = (1 / 2) * (err.flatten('F') @ err.flatten('F').T + r)
        return J

    def grad(self, theta, Y, R, params, reg=0):
        X, Theta = self.extract_params(theta, params)
        X_grad = np.zeros(X.shape)
        Theta_grad = np.zeros(Theta.shape)

        for i in range(X.shape[0]):
            Theta_i = Theta[R[i, :] == 1, :]
            Y_i = Y[i, R[i, :] == 1]
            err_i = (X[i, :] @ Theta_i.T) - Y_i
            reg_i = reg * X[i, :]
            X_grad[i, :] = err_i @ Theta_i + reg_i

        for j in range(Theta.shape[0]):
            X_j = X[R[:, j] == 1, :]
            Theta_j = Theta[j, :]
            Y_j = Y[R[:, j] == 1, j]
            err_j = ((X_j @ Theta_j.T) - Y_j).T
            reg_j = reg * Theta[j, :]
            Theta_grad[j, :] = err_j @ X_j + reg_j

        return np.block([X_grad.flatten('F'), Theta_grad.flatten('F')])

    def grad_numerical(self, theta, Y, R, params, reg=0):
        """
        parameters:
            <ax1> theta: column-wise flattened vector containing the network weights

        returns:
            <ax1> grads: column-wise flattened vector containing the network approximate-gradients

        Numerically approximate the gradient of the cost function using the weights specified by theta.
        """
        grads = np.zeros(theta.size)
        perturbation = np.zeros(theta.size)
        e = 1e-4
        for p in range(theta.size):
            perturbation[p] = e
            loss1 = self.cost(theta - perturbation, Y, R, params, reg)
            loss2 = self.cost(theta + perturbation, Y, R, params, reg)
            grads[p] = (loss2 - loss1) / (2 * e)
            perturbation[p] = 0
        return grads

    def check_grad(self, reg=0):
        from numpy.linalg import norm
        X_s = np.random.rand(4, 3)
        Theta_s = np.random.rand(5, 3)
        Y = X_s @ Theta_s.T
        Y[np.random.random(Y.shape) > 0.5] = 0
        R = np.where(Y != 0, 1, 0)
        X = np.random.randn(X_s.shape[0], X_s.shape[1])
        Theta = np.random.randn(Theta_s.shape[0], Theta_s.shape[1])
        theta = np.block([X.flatten('F'), Theta.flatten('F')])
        params = {'X.size': X.size, 'X.shape': X.shape,
                  'T.size': Theta.size, 'T.shape': Theta.shape}
        grad = self.grad(theta, Y, R, params, reg)
        num_grad = self.grad_numerical(theta, Y, R, params, reg)
        diff = norm(num_grad - grad) / norm(num_grad + grad)
        return diff < 1e-9

    def print_cost(self, theta):
        print("Evaluating gradient...")

    def optimize(self, X, Theta, Y, R, reg=0, debug=False):
        from scipy.optimize import minimize
        theta = np.block([X.flatten('F'), Theta.flatten('F')])
        params = {'X.size': X.size, 'T.size': Theta.size,
                  'X.shape': X.shape, 'T.shape': Theta.shape}
        options = {'maxiter': 50, 'disp': True}
        if debug:
            cbfn = self.print_cost
        else:
            cbfn = None
        theta = minimize(self.cost, theta, args=(Y, R, params, reg), callback=cbfn,
                         method='BFGS', jac=self.grad, options=options).x
        J = self.cost(theta, Y, R, params, reg)
        X, Theta = self.extract_params(theta, params)
        return J, X, Theta
