import numpy as np
from matplotlib.pylab import scatter, show, legend, xlabel, ylabel
import function

data = np.loadtxt('ex2data1.txt', delimiter=',')

X = data[:, 0:2]  # take column 0 and 1 of database
y1 = data[:, 2]
y = y1.reshape(np.size(y1), 1)

# set positive and negative values
pos = np.where(y == 1)  # np.where() equivalent to find() from matlab
neg = np.where(y == 0)

scatter(X[pos, 0], X[pos, 1], marker='x', c='b')
scatter(X[neg, 0], X[neg, 1], marker='o', c='r')
xlabel('exam 1 score')
ylabel('exam 2 score')
legend(['Admitted', 'Not Admitted'])
show()

# setup datamatrix, add a column of 1 to X
[m, n] = np.shape(X)
X1 = np.ones((m, 1))
X = np.hstack((X1, X))

# initialize fitting parameters
initial_theta = np.zeros((n + 1, 1))
grad = []
# compute cost and gradient with zero_theta
cost, grad = function.cost_function(initial_theta, X, y)

print(f"cost at initial theta = 0:\n", cost)
print("Expected cost (approx): 0.693\n")
print(f'gradient at initial theta = 0: \n', grad)
print("Expected gradients (approx):\n -0.1000\n -12.0092\n"
      "-11.2628\n")

# cost and gradient with non_theta
test_theta = np.array([[-24], [0.2], [0.2]])
cost, grad = function.cost_function(test_theta, X, y)

print(f"cost at test theta: \n", cost)
print("expected cost (approx): 0.218\n")
print(f"gradient at test theta:\n", grad)
print("Expected gradients (approx):\n0.043\n 2.566\n 2.647\n")

input("Press Enter to continue...")
# predict and compute accuracy of training set

p = function.predict(test_theta, X)
print('Train Accuracy: %f'
      % ((y[np.where(p == y)].size / float(y.size)) * 100.0))
