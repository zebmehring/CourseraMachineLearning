import numpy as np
from scipy.optimize import minimize


class LogisticRegression:
    def cost(self, theta, X, y, reg=0):
        np.seterr(divide='ignore', invalid='ignore')
        m = len(y)
        h = self.sigmoid(X @ theta)
        return (- (1 / m) * ((y.T @ np.log(h)) + ((1 - y).T @ np.log(1 - h)))) + ((reg / (2 * m)) * (theta[1:].T @ theta[1:]))

    def predict(self, theta, X):
        return np.where(X @ theta < 0, 0, 1)

    def sigmoid(self, z):
        return np.reciprocal(1 + np.exp(-z))

    def cost_grad(self, theta, X, y, reg=0):
        m = len(y)
        h = self.sigmoid(X @ theta)
        return (1 / m) * ((h - y) @ X) + (reg / m) * np.block([0, theta[1:]])

    def optimize(self, theta, X, y, reg=0):
        res = minimize(self.cost, theta, (X, y, reg), method='BFGS',
                       jac=self.cost_grad, options={'maxiter': 400})
        theta = res.x
        J = self.cost(theta, X, y)
        return J, theta
