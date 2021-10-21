import numpy as np


def sigmoid(z):
    g = np.zeros(np.size(z))
    g = 1 / (1 + np.exp(-z))
    return g


def cost_function(theta, X, y):
    m = len(y)  # number of training examples
    J = 0
    grad = np.zeros(np.size(theta))

    # calculate h_theta(X), gradient and the cost J
    sig_func = sigmoid(np.dot(X, theta))
    grad = 1 / m * np.dot(X.T, (sig_func - y))  # gradient
    J = -1 / m * (np.sum(y * np.log(sig_func) + (1 - y)
                         * np.log(1 - sig_func)))

    return J, grad
