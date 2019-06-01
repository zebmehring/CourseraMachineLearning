import numpy as np
from scipy.optimize import minimize


class NeuralNetwork:
    def __init__(self, X, y, Theta, layer_dims, l=0):
        """
        parameters:
            <mxn> X: 2D matrix where each row is an example and each column is a feature
            <mx1> y: 1D vector containing the labeled outcomes
            [<axb>] Theta: list of 2D matricies containing weights for the neural network layers
            [<int>] layer_dims: list representing the size of each layer in the network
            <float> l: regularization parameter (default 0)
        """
        self.X = X
        self.y = y
        self.m = len(y)
        self.Theta = Theta
        self.layer_dims = layer_dims
        self.l = l
        self.iterations = 1

    def update_weights(self, Theta):
        self.Theta = Theta

    def update_architecture(self, layer_dims):
        self.layer_dims = layer_dims

    def update_training_set(self, X, y):
        self.X = X
        self.y = y
        self.m = len(y)

    def update_lambda(self, l):
        self.l = l

    def cost(self, theta, reorder=True):
        """
        parameters:

        returns:
            <float> J: cost for the neural network with parameters Theta using the data from X

        Compute the regularized cost for the data in X with the nerual network specified by Theta.
        """
        # np.seterr(divide='ignore', invalid='ignore')  # ignore log(0) errors
        Theta = self.reconstruct_Theta(theta)

        # feedforward and compute the output
        a = self.X
        for t in Theta:
            a = self.sigmoid(np.block([np.ones((a.shape[0], 1)), a]) @ t.T)
        # reorder columns for 0-based indexing
        if reorder:
            a = np.block([a[:, -1].reshape((a.shape[0], 1)), a[:, :-1]])

        # compute the summed cost for each label
        cost = 0
        for k in range(self.layer_dims[-1]):
            y_k = np.where(self.y == k, 1, 0)
            h_k = a[:, k]
            cost += (y_k.T @ np.log(h_k)) + ((1 - y_k).T @ np.log(1 - h_k))

        # compute the regularization term
        reg = sum([sum(sum(np.square(t[:, 1:]))) for t in Theta])

        # compute the total cost
        return -(1 / self.m) * cost + (self.l / (2 * self.m)) * reg

    def cost_grad(self, theta):
        """
        parameters:


        returns:



        """
        Theta = self.reconstruct_Theta(theta)

        grads = [np.zeros(Theta[i].shape) for i in range(len(Theta))]
        for t in range(self.m):
            a = [self.X[t, :]]
            z = [None]
            for i in range(len(Theta)):
                z.append(Theta[i] @ np.block([1, a[i]]))
                a.append(self.sigmoid(z[-1]))
            y_t = np.where(np.arange(self.layer_dims[-1]) == self.y[t], 1, 0)

            delta = [a[-1] - y_t]
            for i in range(len(a) - 2, 0, -1):
                delta.insert(0, (Theta[i].T @ delta[0])
                             * np.block([1, self.sigmoid_gradient(z[i])]))
                delta[0] = delta[0][1:]

            for i in range(len(Theta)):
                grads[i] = grads[i] + np.outer(delta[i], np.block([1, a[i]]))

        for i in range(len(Theta)):
            grads[i] = (1 / self.m) * grads[i] + (self.l / self.m) * \
                np.block([np.zeros((Theta[i].shape[0], 1)), Theta[i][:, 1:]])

        return np.block([g.reshape(g.size, order='F') for g in grads])

    def cost_grad_numerical(self, theta):
        """
        """
        grads = np.zeros(theta.size)
        perturbation = np.zeros(theta.size)
        e = 1e-4
        for p in range(theta.size):
            perturbation[p] = e
            loss1 = self.cost(theta - perturbation, reorder=False)
            loss2 = self.cost(theta + perturbation, reorder=False)
            grads[p] = (loss2 - loss1) / (2 * e)
            perturbation[p] = 0
        return grads

    def predict(self, theta):
        """
        parameters:
            [<axb>] Theta: list of 2D matricies containing weights for the neural network layers
            <mxn> X: 2D matrix where each row is an example and each column is a feature

        returns:
            <mx1> y: 1D vector containing the prediction for each example in X

        Compute the output of a neural network using the weights provided. The architecture is specified by the dimensions of these weights.
        """
        Theta = self.reconstruct_Theta(theta)
        a = self.X
        for t in Theta:
            a = self.sigmoid(np.block([np.ones((a.shape[0], 1)), a]) @ t.T)
        return np.argmax(a, 1)

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

    def reconstruct_Theta(self, theta):
        """
        """
        Theta = []
        t_i = 0
        for i in range(len(self.layer_dims) - 1):
            Theta.append(np.reshape(theta[t_i:t_i+((self.layer_dims[i] + 1) * self.layer_dims[i + 1])],
                                    (self.layer_dims[i + 1], self.layer_dims[i] + 1), order='F'))
            t_i = (self.layer_dims[i] + 1) * self.layer_dims[i + 1]
        return Theta

    def print_cost(self, theta):
        """
        """
        cost = self.cost(theta)
        print("Iteration {0}: {1}".format(self.iterations, cost))
        self.iterations += 1

    def optimize(self, theta):
        """
        parameters:


        returns:



        """
        self.iterations = 1
        res = minimize(self.cost, theta, method='CG', callback=self.print_cost,
                       jac=self.cost_grad, options={'maxiter': 50, 'disp': True})
        theta = res.x
        J = self.cost(theta)
        return J, theta
