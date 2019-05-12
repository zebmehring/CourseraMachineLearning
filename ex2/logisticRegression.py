import numpy as np
from scipy.optimize import minimize


class LogisticRegression:
    def cost(self, theta, X, y):
        np.seterr(divide='ignore', invalid='ignore')
        g = self.sigmoid(X @ theta)
        return - (1 / len(y)) * sum((np.log(g).T * y) +
                                    (np.log(1 - g).T * (1 - y)))

    def predict(self, theta, X):
        return np.where(X @ theta < 0, 0, 1)

    def sigmoid(self, z):
        return np.reciprocal(1 + np.exp(-z))

    def cost_grad(self, theta, X, y):
        return (1 / len(y)) * X.T @ (self.sigmoid(X @ theta) - y)

    def optimize(self, theta, X, y):
        res = minimize(self.cost, theta, (X, y), method='BFGS',
                       jac=self.cost_grad, options={'maxiter': 400})
        theta = res.x
        J = self.cost(theta, X, y)
        return J, theta
