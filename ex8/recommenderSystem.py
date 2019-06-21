import numpy as np


class CollaborativeFiltering:
    def normalize(self, Y, R):
        """
        parameters:
            <mxu> Y: 2D matrix where Y[i, j] is the rating of movie i by user j (or 0)
            <mxu> R: boolean-valued matrix where R[i, j] = 1 iff Y[i, j] != 0

        returns:
            <mx1> Y_mean: vector containing the mean rating of each movie
            <mxu> Y_norm: 2D matrix equivalent to Y but rescaled to have 0 mean

        Rescale the ratings in Y to have zero mean.
        """
        m, n = Y.shape
        Y_mean = [np.mean(Y[i, R[i, :] == 1]) for i in range(m)]
        Y_norm = np.zeros(Y.shape)
        for i in range(m):
            Y_norm[i, R[i, :] == 1] = Y[i, R[i, :] == 1] - Y_mean[i]
        return np.array(Y_mean), np.array(Y_norm)

    def extract_params(self, theta, params):
        """
        parameters:
            <(m*n+n*u)x1> theta: column-wise flattened vector containing the hypothesis matrices (X, Theta)
            <dict> params: dictionary containing information about the shape of X and Theta

        returns:
            <mxn> X: 2D matrix where X[i, j] is the amount of feature j in movie i
            <uxn> Theta: 2D matrix where Theta[i, j] is proportional to how much user i likes feature j

        Reconstruct the parameter matrices X and Theta from theta.
        """
        X = np.reshape(theta[:params['X.size']], params['X.shape'], 'F')
        Theta = np.reshape(theta[params['X.size']:], params['T.shape'], 'F')
        return X, Theta

    def cost(self, theta, Y, R, params, reg=0):
        """
        parameters:
            <(m*n+n*u)x1> theta: column-wise flattened vector containing the hypothesis matrices (X, Theta)
            <mxu> Y: 2D matrix where Y[i, j] is the rating of movie i by user j (or 0)
            <mxu> R: boolean-valued matrix where R[i, j] = 1 iff Y[i, j] != 0
            <dict> params: dictionary containing information about the shape of X and Theta
            <float> reg: regularization parameter

        returns:
            <float> J: cost for the hypothesis theta

        Compute the cost of the collaborative filtering algorithm with flattened hypothesis theta.
        """
        X, Theta = self.extract_params(theta, params)
        err = ((X @ Theta.T) * R) - (Y * R)
        r = reg * (theta @ theta.T)
        J = (1 / 2) * (err.flatten('F') @ err.flatten('F').T + r)
        return J

    def grad(self, theta, Y, R, params, reg=0):
        """
        parameters:
            <(m*n+n*u)x1> theta: column-wise flattened vector containing the hypothesis matrices (X, Theta)
            <mxu> Y: 2D matrix where Y[i, j] is the rating of movie i by user j (or 0)
            <mxu> R: boolean-valued matrix where R[i, j] = 1 iff Y[i, j] != 0
            <dict> params: dictionary containing information about the shape of X and Theta
            <float> reg: regularization parameter

        returns:
            <(m*n+n*u)x1> grads: column-wise flattened vector containing the gradients of the hypothesis

        Analytically compute the gradient of the collaborative filtering cost function with respect to each parameter.
        """
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

    def num_grad(self, theta, Y, R, params, reg=0):
        """
        parameters:
            <(m*n+n*u)x1> theta: column-wise flattened vector containing the hypothesis matrices (X, Theta)
            <mxu> Y: 2D matrix where Y[i, j] is the rating of movie i by user j (or 0)
            <mxu> R: boolean-valued matrix where R[i, j] = 1 iff Y[i, j] != 0
            <dict> params: dictionary containing information about the shape of X and Theta
            <float> reg: regularization parameter

        returns:
            <(m*n+n*u)x1> num_grads: column-wise flattened vector containing the numerically-approximated gradients

        Numerically approximate the gradient of the cost function using the values specified by theta.
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
        """
        parameters:
            <float> reg: regularization parameter

        returns:
            <bool> succ: True if the analytical and numerical gradients agree to one part in 1 billion

        Verify the correctness of self.grad() by checking it against self.num_grad() on a small sample.
        """
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
        num_grad = self.num_grad(theta, Y, R, params, reg)
        diff = norm(num_grad - grad) / norm(num_grad + grad)
        return diff < 1e-9

    def optimize(self, X, Theta, Y, R, reg=0):
        """
        parameters:
            <mxn> X: 2D matrix where X[i, j] is the amount of feature j in movie i
            <uxn> Theta: 2D matrix where Theta[i, j] is proportional to how much user i likes feature j
            <mxu> Y: 2D matrix where Y[i, j] is the rating of movie i by user j (or 0)
            <mxu> R: boolean-valued matrix where R[i, j] = 1 iff Y[i, j] != 0
            <float> reg: regularization parameter

        returns:
            <float> J: cost for the hypothesis theta with the data from X
            <uxm> X: optimized feature matrix
            <mxn> Theta: optimized hypothesis matrix

        Optimize the collaborative filtering hypothesis using the nonlinear CG optimization algorithm.
        """
        from scipy.optimize import minimize
        theta = np.block([X.flatten('F'), Theta.flatten('F')])
        params = {'X.size': X.size, 'T.size': Theta.size,
                  'X.shape': X.shape, 'T.shape': Theta.shape}
        options = {'maxiter': 100, 'disp': True}
        theta = minimize(self.cost, theta, args=(Y, R, params, reg),
                         method='CG', jac=self.grad, options=options).x
        J = self.cost(theta, Y, R, params, reg)
        X, Theta = self.extract_params(theta, params)
        return J, X, Theta
