import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.io import loadmat
# Use loadmat to load matlab files
mat=loadmat("ex3data1.mat")
X=mat["X"]
y=mat["y"]

fig, axis = plt.subplots(10,10,figsize=(8,8))
for i in range(10):
    for j in range(10):
