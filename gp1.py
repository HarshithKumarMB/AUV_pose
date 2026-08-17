import pandas as pd
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# =====================================================
# READ DATA
# =====================================================

df1 = pd.read_csv("map.csv")
df2 = pd.read_csv("map1.csv")

df = pd.concat([df1, df2], ignore_index=True)

X = df[['x', 'y']].values.astype(np.float32)
y = (-df['sonar_depth'].values).astype(np.float32)

print("Total points:", len(X))

# =====================================================
# SCALE INPUTS
# =====================================================

x_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X)

y_mean = y.mean()
y_std = y.std()

y_scaled = (y - y_mean) / y_std

# =====================================================
# TENSORS
# =====================================================

train_x = torch.tensor(X_scaled, dtype=torch.float32)
train_y = torch.tensor(y_scaled, dtype=torch.float32)

# =====================================================
# DATALOADER
# =====================================================

batch_size = 5000

dataset = TensorDataset(train_x, train_y)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)

# =====================================================
# INDUCING POINTS
# =====================================================

n_inducing = 500

idx = np.random.choice(
    len(train_x),
    n_inducing,
    replace=False
)

inducing_points = train_x[idx]

# =====================================================
# SVGP MODEL
# =====================================================

class SVGPModel(gpytorch.models.ApproximateGP):

    def __init__(self, inducing_points):

        variational_distribution = (
            gpytorch.variational.CholeskyVariationalDistribution(
                inducing_points.size(0)
            )
        )

        variational_strategy = (
            gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            )
        )

        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):

        mean_x = self.mean_module(x)

        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(
            mean_x,
            covar_x
        )

# =====================================================
# CREATE MODEL
# =====================================================

model = SVGPModel(inducing_points)

likelihood = gpytorch.likelihoods.GaussianLikelihood()

# =====================================================
# TRAIN
# =====================================================

model.train()
likelihood.train()

optimizer = torch.optim.Adam(
    [
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ],
    lr=0.01,
)

mll = gpytorch.mlls.VariationalELBO(
    likelihood,
    model,
    num_data=len(train_y)
)

epochs = 20

for epoch in range(epochs):

    epoch_loss = 0

    for batch_idx, (x_batch, y_batch) in enumerate(loader, start=1):

        optimizer.zero_grad()

        output = model(x_batch)

        loss = -mll(output, y_batch)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Batch {batch_idx}/{len(loader)} | "
            f"Loss={loss.item():.4f}"
        )

    print(
        f"Epoch {epoch+1} complete | "
        f"Average Loss={epoch_loss/len(loader):.4f}"
    )

print("Training complete.")

# =====================================================
# SAVE TRAINED MODEL
# =====================================================

import pickle

model_package = {
    "model_state_dict": model.state_dict(),
    "likelihood_state_dict": likelihood.state_dict(),
    "inducing_points": inducing_points.cpu(),
    "x_scaler": x_scaler,
    "y_mean": y_mean,
    "y_std": y_std,
    "n_inducing": n_inducing
}

with open("svgp_bathymetry.pkl", "wb") as f:
    pickle.dump(model_package, f)

print("Model saved to svgp_bathymetry.pkl")

# =====================================================
# CREATE PREDICTION GRID
# =====================================================

xmin, xmax = df['x'].min(), df['x'].max()
ymin, ymax = df['y'].min(), df['y'].max()

nx = 200
ny = 200

xgrid = np.linspace(xmin, xmax, nx)
ygrid = np.linspace(ymin, ymax, ny)

xx, yy = np.meshgrid(xgrid, ygrid)

grid_points = np.column_stack([
    xx.ravel(),
    yy.ravel()
])

# =====================================================
# SCALE GRID
# =====================================================

grid_scaled = x_scaler.transform(grid_points)

grid_tensor = torch.tensor(
    grid_scaled,
    dtype=torch.float32
)

# =====================================================
# PREDICTION
# =====================================================

model.eval()
likelihood.eval()

means = []

chunk_size = 5000

with torch.no_grad(), gpytorch.settings.fast_pred_var():

    for i in range(0, len(grid_tensor), chunk_size):

        chunk = grid_tensor[i:i + chunk_size]

        pred = likelihood(model(chunk))

        means.append(
            pred.mean.cpu().numpy()
        )

        print(
            f"Predicted "
            f"{min(i + chunk_size, len(grid_tensor))}"
            f"/{len(grid_tensor)}"
        )

# =====================================================
# UNDO NORMALIZATION
# =====================================================

mean_depth = np.concatenate(means)

mean_depth = mean_depth * y_std + y_mean

print("Prediction shape:", mean_depth.shape)

# =====================================================
# RESHAPE FOR SURFACE
# =====================================================

Z = mean_depth.reshape(ny, nx)

print("Surface shape:", Z.shape)

# =====================================================
# 3D PLOT
# =====================================================

fig = plt.figure(figsize=(14, 10))

ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    xx,
    yy,
    Z,
    cmap='viridis',
    linewidth=0,
    antialiased=True
)

fig.colorbar(
    surf,
    shrink=0.6,
    aspect=15,
    label='Depth'
)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Depth')

ax.set_title('Gaussian Process Bathymetry Surface')

ax.invert_zaxis()

plt.tight_layout()
# Save figure
plt.savefig("gp_bathymetry_surface.png", dpi=300, bbox_inches="tight")
plt.show()


