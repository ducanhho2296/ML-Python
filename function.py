import matplotlib.pyplot as plt
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


def predict(theta, X):
    """Predict whether the label
    is 0 or 1 using learned logistic
    regression parameters """
    m, n = X.shape
    p = np.zeros(shape=(m, 1))

    h = sigmoid(np.dot(X, theta))

    for i in range(0, h.shape[0]):
        if h[i] > 0.5:
            p[i, 0] = 1
        else:
            p[i, 0] = 0

    return p

def plot_decision_boundary(theta, X, y):
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i, j] = 0
