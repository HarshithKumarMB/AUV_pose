import pandas as pd
import numpy as np

import torch
import gpytorch

from torch.utils.data import TensorDataset, DataLoader

# ======================
# Read data
# ======================

df1 = pd.read_csv("map.csv")
df2 = pd.read_csv("map1.csv")

df = pd.concat([df1, df2], ignore_index=True)

X = df[['x', 'y']].values.astype(np.float32)
y = (-df['sonar_depth'].values).astype(np.float32)

# ======================
# Convert to tensors
# ======================

train_x = torch.tensor(X)
train_y = torch.tensor(y)

# ======================
# DataLoader
# ======================

batch_size = 10000

dataset = TensorDataset(train_x, train_y)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)

# ======================
# Select inducing points
# ======================

n_inducing = 500

idx = np.random.choice(
    len(train_x),
    n_inducing,
    replace=False
)

inducing_points = train_x[idx]


# ======================
# SVGP model
# ======================

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


model = SVGPModel(inducing_points)

likelihood = gpytorch.likelihoods.GaussianLikelihood()

# ======================
# Training
# ======================

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
            f"Epoch {epoch+1}/{epochs} "
            f"Batch {batch_idx}/{len(loader)} "
            f"Loss={loss.item():.4f}"
        )

    print(
        f"Epoch {epoch+1} complete. "
        f"Average Loss={epoch_loss/len(loader):.4f}"
    )

print("Training complete.")