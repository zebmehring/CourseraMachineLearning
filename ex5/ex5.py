import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from linearRegression import LinearRegression


if __name__ == '__main__':
    data = scipy.io.loadmat('ex5data1.mat')
    X, y = (data['X'], data['y'])
    X_val, y_val = (data['Xval'], data['yval'])
    X_test, y_test = (data['Xtest'], data['ytest'])
    linreg = LinearRegression(X, y, 0)

    print('========== Part 1 ==========')
