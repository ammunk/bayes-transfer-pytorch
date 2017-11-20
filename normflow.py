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
        self.u = nn.Parameter(torch.Tensor(features))
        self.w = nn.Parameter(torch.Tensor(features))
        self.b = nn.Parameter(torch.Tensor(1))

    def forward(self, z):
        # Create uhat such that it is parallel to w
        uw = torch.dot(self.u, self.w)
        uhat = self.u + ((-1 + F.softplus(uw)) - uw) * torch.transpose(self.w, 0, -1) / torch.norm(self.w, 2)**2

        # Equation 21 - Transform z
        linear = torch.dot(z, self.w) + self.b
        f_z = z + uhat * F.tanh(linear)

        # Compute the Jacobian using the fact that
        # tanh(x) dx = 1 - tanh(x)**2
        psi = (1 - F.tanh(linear)**2) * self.w
        psi_u = torch.dot(psi, uhat)

        # Return the transformed output along
        # with log determninant of J
        logdet_jacobian = torch.log(torch.abs(1 + psi_u))
        return f_z, logdet_jacobian