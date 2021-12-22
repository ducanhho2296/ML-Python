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

def gradientDecent(X,y, theta, alpha, num_iters, Lambda):
    m = len(y)
    J_history = []

    for i in range(num_iters):
        cost, grad = lrCostFuntion(theta, X, y, Lambda)
        theta = theta - (alpha * grad)
        J_history.append(cost)
    return theta, J_history

def onevsAll(X, y, num_labels, Lambda):
    m, n = X.shape[0], X.shape[1]
    init_theta = np.zeros((n+1,1))
    all_theta = []
    all_J = []

    X = np.hstack((np.ones((m,1)),X))

    for i in range(1, num_labels + 1):
        theta, J_history = gradientDecent(X, np.where(y==i, 1, 0), init_theta,1,300, Lambda)
        all_theta.append(theta)
        all_J.append(J_history)
    return np.array(all_theta).reshape(num_labels, n+1), all_J

def predictOnevsAll(all_theta, X):
    m = X.shape[0]
    X = np.hstack((np.ones((m,1)),X))

    predictions = np.dot(X, all_theta.T)
    return np.argmax(predictions, axis=1) + 1


