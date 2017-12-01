import torch
import torch.nn as nn
import torch.nn.functional as F


class PlanarNormalizingFlow(nn.Module):
    """
    Based on Normalizing Flow implementation from Parmesan
    https://github.com/casperkaae/parmesan
    """
    def __init__(self, features):
        super(PlanarNormalizingFlow, self).__init__()
        self.u = nn.Parameter(torch.rand(features))
        self.y = nn.Parameter(torch.rand(features))
        self.b = nn.Parameter(torch.rand(1))

    def forward(self, z):
        # Create uhat such that it is parallel to w
        uw = torch.dot(self.u, self.y)
        uhat = self.u + ((-1 + F.softplus(uw)) - uw) * torch.transpose(self.y, 0, -1) / torch.norm(self.y, 2) ** 2

        # Equation 21 - Transform z
        linear = torch.dot(z, self.y) + self.b
        f_z = z + uhat * F.tanh(linear)

        # Compute the Jacobian using the fact that
        # tanh(x) dx = 1 - tanh(x)**2
        psi = (1 - F.tanh(linear)**2) * self.y
        psi_u = torch.dot(psi, uhat)

        # Return the transformed output along
        # with log determninant of J
        logdet_jacobian = torch.log(torch.abs(1 + psi_u) + 1e-8)
        return f_z, logdet_jacobian