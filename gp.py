import pandas as pd
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, ConstantKernel, WhiteKernel
)

# Read files
df = pd.read_csv("map.csv")
#df2 = pd.read_csv("map1.csv")

# Combine
#df = pd.concat([df1, df2], ignore_index=True)

# Training data
X = df[['x', 'y']].values
y = -df['sonar_depth'].values

# GP Kernel
kernel = (
    ConstantKernel(1.0, (1e-3, 1e3))
    * RBF(length_scale=10.0, length_scale_bounds=(1e-2, 1e3))
    + WhiteKernel(noise_level=0.1)
)

gp = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    n_restarts_optimizer=5
)

gp.fit(X, y)

print("Learned kernel:")
print(gp.kernel_)