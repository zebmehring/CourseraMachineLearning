import numpy as np
from scipy.optimize import minimize


class NeuralNetwork:
    def cost(self, Theta, X, y, reg=0):
        """
        parameters:
            [<axb>] Theta: list of 2D matricies containing weights for the neural network layers
            <mxn> X: 2D matrix where each row is an example and each column is a feature
            <mx1> y: 1D vector containing the labeled outcomes
            <float> reg: regularization parameter (default 0)

        returns:
            <float> J: cost for the neural network with parameters Theta using the data from X

        Compute the regularized cost for the data in X with the nerual network specified by Theta.
        """
        np.seterr(divide='ignore', invalid='ignore')  # ignore log(0) errors
        m = len(y)
        pass

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

    def cost_grad(self):
        """
        parameters:


        returns:



        """
        pass

    def optimize(self):
        """
        parameters:


        returns:



        """
        pass
