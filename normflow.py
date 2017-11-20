import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalizingFlow(nn.Module):

    def __init__(self, out_features, in_features):
        super(NormalizingFlow, self).__init__(self)
        self.u = nn.Parameter(torch.Tensor(in_features, out_features))
        self.w = nn.Parameter(torch.Tensor(in_features, out_features))
        self.b = nn.Parameter(torch.Tensor(out_features))

    def forward(self, z):
        uw = torch.dot(self.u, self.w)
        uhat = self.u + ((-1 + F.softplus(uw)) - uw) * self.w.T / torch.norm(self.w, 2)**2

        # Equation 21
        linear = torch.dot(z, self.w) + self.b
        f_z = z + uhat * F.tanh(linear)

        # tanh(x)dx = 1 - tanh(x)**2
        psi = torch.dot((1 - F.tanh(linear) ** 2), self.w)
        psi_u = torch.dot(psi, uhat)

        logdet_jacobian = torch.log(torch.abs(1 + psi_u))

        return f_z, logdet_jacobian