import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalizingFlowLayer(nn.Module):
    def __init__(self, n, features):
        super(NormalizingFlowLayer, self).__init__()
        flows = [PlanarNormalizingFlow(features) for _ in range(n)]
        self.flows = nn.ModuleList(flows)

    def forward(self, z):
        log_dets = []

        for flow in self.flows:
            z, log_det_jacobian = flow(z)
            log_dets.append(log_det_jacobian)

        return z, log_dets


class PlanarNormalizingFlow(nn.Module):
    """
    Based on Normalizing Flow implementation from Parmesan
    https://github.com/casperkaae/parmesan
    """
    def __init__(self, features):
        super(PlanarNormalizingFlow, self).__init__()
        self.u = nn.Parameter(torch.randn(features)) # (features,)
        self.w = nn.Parameter(torch.randn(features)) # (features,)
        self.b = nn.Parameter(torch.ones(1))        # (1,)

    def forward(self, z):
        # Create uhat such that it is parallel to w
        uw = torch.dot(self.u, self.w) # (1,)
        muw = -1 + F.softplus(uw)  # (1,)
        uhat = self.u + (muw - uw) * torch.transpose(self.w, 0, -1) / torch.sum(self.w ** 2) # (features,)

        # Equation 21 - Transform z
        zwb = torch.mv(z, self.w) + self.b # (batch_size,)

        f_z = z + (uhat.view(1, -1) * F.tanh(zwb).view(-1, 1)) # (batch_size, features)

        # Compute the Jacobian using the fact that
        # tanh(x) dx = 1 - tanh(x)**2
        psi = (1 - F.tanh(zwb)**2).view(-1, 1) * self.w.view(1, -1)  # (batch_size, features)
        psi_u = torch.mv(psi, uhat) # (batch_size,)

        # Return the transformed output along
        # with log determninant of J
        logdet_jacobian = torch.log(torch.abs(1 + psi_u) + 1e-8) #(batch_size,)

        return f_z, logdet_jacobian