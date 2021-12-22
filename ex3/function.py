import numpy as np

def sigmoid(z):
    return  1/ (1 + np.exp(-z))

def lrCostFuntion(theta, X, y, Lambda):
    m = len(y)
    predictions = sigmoid(X * theta)
    error = (-y * np.log(predictions)) - (1 - y) * np.log(1 - predictions)
    cost = (1/m) * sum(error)
    regCost = cost + Lambda/(2 * m) * sum(theta[1:] ** 2)

    #gradient
    j_0 = (1 / m) * (X.transpose() @ (predictions - y))[0]    # @: matrices multiplication
    j_1 = (1 / m) * (X.transpose() @ (predictions - y))[1:] + (Lambda/m) * theta[1:]
    grad = np.vstack((j_0[:,np.newaxis],j_1))
    return regCost[0], grad

def gradientDecent(X,y theta, alpha,num_iters, Lambda):
    m = len(y)
    J_history = []

