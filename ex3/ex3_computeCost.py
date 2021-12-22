import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.io import loadmat
from function import *
# Use loadmat to load matlab files
mat = loadmat("ex3data1.mat")
X = mat["X"]
y = mat["y"]

fig, axis = plt.subplots(10, 10, figsize=(8, 8))
for i in range(10):
    for j in range(10):
        axis[i, j].imshow(X[np.random.randint(0, 5001), :].reshape(20, 20, order="F"), cmap="hot")
        axis[i, j].axis("off")

theta_t = np.array([-2, -1, 1, 2]).reshape(4, 1)
X_t = np.array([np.linspace(0.1, 1.5, 15)]).reshape(3, 5).T
# adding column 1 to X_t
X_t = np.hstack(np.ones((5,1)), X_t)
y_t = np.array([1,0,1,0,1]).reshape(5,1)
J, grad = lrCostFuntion(theta_t, X_t, y_t, 3)

print("cost:", J, "Expected cost: 2.534819")
print("Gradients:\n",grad,"\nExpected gradients:\n 0.146561\n -0.548558\n 0.724722\n 1.398003")