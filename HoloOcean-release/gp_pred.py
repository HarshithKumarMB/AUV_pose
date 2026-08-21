import pickle
import torch
import gpytorch
import numpy as np

# Recreate the model class
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
                learn_inducing_locations=True
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

# Load checkpoint
with open("svgp_bathymetry.pkl", "rb") as f:
    checkpoint = pickle.load(f)

model = SVGPModel(checkpoint["inducing_points"])
likelihood = gpytorch.likelihoods.GaussianLikelihood()

model.load_state_dict(checkpoint["model_state_dict"])
likelihood.load_state_dict(checkpoint["likelihood_state_dict"])

x_scaler = checkpoint["x_scaler"]
y_mean = checkpoint["y_mean"]
y_std = checkpoint["y_std"]

model.eval()
likelihood.eval()

points = np.array([
    [-19.412034324804928,-9.419920050422297]
], dtype=np.float32)

points_scaled = x_scaler.transform(points)

x = torch.tensor(points_scaled, dtype=torch.float32)

with torch.no_grad():
    pred = likelihood(model(x))

depth = pred.mean.numpy()
depth = depth * y_std + y_mean

print(depth)