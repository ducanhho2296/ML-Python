import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
# Use loadmat to load matlab files
mat=loadmat("ex3data1.mat")
X=mat["X"]
y=mat["y"]