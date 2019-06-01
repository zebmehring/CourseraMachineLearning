import numpy as np
from scipy.optimize import minimize


class NeuralNetwork:
    def cost(self, theta, X, y, layer_dims, l=0, reorder=True):
        """
        parameters:
            <mxn> X: 2D matrix where each row is an example and each column is a feature
            <mx1> y: 1D vector containing the labeled outcomes
            [<axb>] Theta: list of 2D matricies containing weights for the neural network layers
            [<int>] layer_dims: list representing the size of each layer in the network
            <float> reg: regularization parameter (default 0)

        returns:
            <float> J: cost for the neural network with parameters Theta using the data from X

        Compute the regularized cost for the data in X with the nerual network specified by Theta.
        """
        np.seterr(divide='ignore', invalid='ignore')  # ignore log(0) errors
        m = len(y)
        Theta = self.reconstruct_Theta(theta, layer_dims)

        # feedforward and compute the output
        a = X
        for t in Theta:
            a = self.sigmoid(np.block([np.ones((a.shape[0], 1)), a]) @ t.T)
        # reorder columns for 0-based indexing
        if reorder:
            a = np.block([a[:, -1].reshape((a.shape[0], 1)), a[:, :-1]])

        # compute the summed cost for each label
        cost = 0
        for k in range(layer_dims[-1]):
            y_k = np.where(y == k, 1, 0)
            h_k = a[:, k]
            cost += (y_k.T @ np.log(h_k)) + ((1 - y_k).T @ np.log(1 - h_k))

        # compute the regularization term
        reg = sum([sum(sum(np.square(t[:, 1:]))) for t in Theta])

        # compute the total cost
        return -(1 / m) * cost + (l / (2 * m)) * reg

    def cost_grad(self, theta, X, y, layer_dims, l=0):
        """
        parameters:


        returns:



        """
        m = len(y)
        Theta = self.reconstruct_Theta(theta, layer_dims)

        grads = [np.zeros(Theta[i].shape) for i in range(len(Theta))]
        for t in range(m):
            a = [X[t, :]]
            z = [None]
            for i in range(len(Theta)):
                z.append(Theta[i] @ np.block([1, a[i]]))
                a.append(self.sigmoid(z[-1]))
            y_t = np.where(np.arange(layer_dims[-1]) == y[t], 1, 0)

            delta = [a[-1] - y_t]
            for i in range(len(a) - 2, 0, -1):
                delta.insert(0, (Theta[i].T @ delta[0])
                             * np.block([1, self.sigmoid_gradient(z[i])]))
                delta[0] = delta[0][1:]

            for i in range(len(Theta)):
                grads[i] = grads[i] + np.outer(delta[i], np.block([1, a[i]]))

        for i in range(len(Theta)):
            grads[i] = (1 / m) * grads[i] + (l / m) * \
                np.block([np.zeros((Theta[i].shape[0], 1)), Theta[i][:, 1:]])

        return np.block([g.reshape(g.size, order='F') for g in grads])

    def cost_grad_numerical(self, theta, X, y, layer_dims, l=0):
        """
        """
        grads = np.zeros(theta.size)
        perturbation = np.zeros(theta.size)
        e = 1e-4
        for p in range(theta.size):
            perturbation[p] = e
            loss1 = self.cost(theta - perturbation, X, y,
                              layer_dims, l, reorder=False)
            loss2 = self.cost(theta + perturbation, X, y,
                              layer_dims, l, reorder=False)
            grads[p] = (loss2 - loss1) / (2 * e)
            perturbation[p] = 0
        return grads

    def predict(self, Theta, X):
        """
        parameters:
            [<axb>] Theta: list of 2D matricies containing weights for the neural network layers
            <mxn> X: 2D matrix where each row is an example and each column is a feature

        returns:
            <mx1> y: 1D vector containing the prediction for each example in X

        Compute the output of a neural network using the weights provided. The architecture is specified by the dimensions of these weights.
        """
        for theta in Theta:
            a = self.sigmoid(np.block([np.ones((a.shape[0], 1)), a]) @ theta.T)
        return (np.argmax(a, 1) + 1) % Theta[-1].shape[1]

    def sigmoid(self, z):
        """
        parameters:
            <qxp> z: argument of the sigmoid function

        returns:
            <qxp> g: sigmoid(z)

        Compute the sigmoid function.
        """
        return np.reciprocal(1 + np.exp(-z))

    def sigmoid_gradient(self, z):
        """
        parameters:
            <qxp> z: argument of the sigmoid function

        returns:
            <qxp> g: d/dz sigmoid(z)

        Compute the gradient of the sigmoid function.
        """
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def initialize_weights(self, n_in, n_out, epsilon_init, debug=False):
        """
        """
        if debug:
            W = np.zeros((n_out, n_in + 1))
            return np.reshape(np.sin(np.arange(1, W.size + 1)), W.shape, order='F') / 10
        else:
            return 2 * np.random.rand(n_out, 1 + n_in) * epsilon_init - epsilon_init

    def reconstruct_Theta(self, theta, layer_dims):
        """
        """
        Theta = []
        t_i = 0
        for i in range(len(layer_dims) - 1):
            Theta.append(np.reshape(
                theta[t_i:t_i+((layer_dims[i] + 1) * layer_dims[i+1])], (layer_dims[i+1], layer_dims[i]+1), order='F'))
            t_i = (layer_dims[i] + 1) * layer_dims[i+1]
        return Theta

    def optimize(self, theta, X, y, layer_dims, l=0):
        """
        parameters:


        returns:



        """
        res = minimize(self.cost, theta, (X, y, layer_dims, l), method='CG',
                       jac=self.cost_grad, options={'maxiter': 50})
        theta = res.x
        J = self.cost(theta, X, y, layer_dims)
        return J, theta
