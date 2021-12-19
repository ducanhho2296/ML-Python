import numpy as np

def sigmoid(z):
    return  1/ (1 + np.exp(-z))

def lrCostFuntion(theta, X, y, Lambda):
    m = len(y)
    predictions = sigmoid(X * theta)
    error = (-y * np.log(predictions)) - (1 - y) * np.log(1 - predictions)
    cost = (1/m) * sum(error)
    regCost = cost + Lambda/(2 * m) * sum(theta[1:] ** 2)