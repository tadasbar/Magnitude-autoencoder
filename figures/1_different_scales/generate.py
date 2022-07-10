import numpy as np
import matplotlib.pyplot as plt
from synthetic_data_datamodule import Circle
from utils import *

data = Circle(1000)
data.setup()

data_0 = data.train_set.clone()
data_1 = data.train_set.clone() * 1e1
data_m5 = data.train_set.clone() * 1e-5

plt.figure(figsize=[15, 5])
plt.subplot(1, 3, 1)
plt.scatter(*data_0.T, c=calculate_weighvec(data_0), cmap="binary")
plt.subplot(1, 3, 2)
plt.scatter(*data_1.T, c=calculate_weighvec(data_1), cmap="binary")
plt.subplot(1, 3, 3)
plt.scatter(*data_m5.T, c=calculate_weighvec(data_m5), cmap="binary")
plt.savefig("plot.png", dpi=200)
