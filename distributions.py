import torch
import math
import numpy as np
from torch.autograd import Variable
from torch.nn import Parameter


class Distribution():
    """
    Base class for torch-based probability distributions.
    """
    def pdf(self, x):
        raise NotImplementedError

    def logpdf(self, x):
        raise NotImplementedError

    def cdf(self, x):
        raise NotImplementedError

    def logcdf(self, x):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError


class Normal(Distribution):
    # scalar version
    def __init__(self, loc, logvar):
        self.loc = loc
        self.logvar = logvar
        self.shp = loc.size()

        super(Normal, self).__init__()

    def logpdf(self, x, eps=0.0):
        c = - float(0.5 * math.log(2 * math.pi))
        return c - 0.5 * self.logvar - (x - self.loc).pow(2) / (2 * torch.exp((self.logvar)) + eps)

    def pdf(self, x):
        return torch.exp(self.logpdf(x))

    def sample(self):
        if self.loc.is_cuda:
            eps = torch.randn(self.shp).cuda()
        else:
            eps = torch.randn(self.shp)
        return self.loc + torch.exp(0.5 * self.logvar) * Variable(eps)

    def entropy(self):
        return 0.5 * math.log(2. * math.pi * math.e) + 0.5 * self.logvar


class FixedMixtureNormal(torch.nn.Module):
    # needs to be a nn.Module to register the parameters correcly
    # takes loc, logvar and pi as list of float values and assumes they are shared across all dimenisions
    def __init__(self, loc, logvar, pi):
        super(FixedMixtureNormal, self).__init__()
        assert sum(pi) - 1 < 0.0001
        self.loc    = Parameter(torch.from_numpy(np.array(loc)).float(), requires_grad=False)
        self.logvar = Parameter(torch.from_numpy(np.array(logvar)).float(), requires_grad=False)
        self.pi     = Parameter(torch.from_numpy(np.array(pi)).float(), requires_grad=False)

    def _component_logpdf(self, x, eps=0.0):
        ndim = len(x.size())
        shpexpand = ndim * (None,)
        x = x.unsqueeze(-1)

        c = - float(0.5 * math.log(2 * math.pi))
        loc = self.loc[shpexpand]
        logvar = self.logvar[shpexpand]
        pi = self.pi[shpexpand]

        return c - 0.5 * logvar - (x - loc).pow(2) / (2 * torch.exp(logvar) + eps)

    def logpdf(self, x):
        ndim = len(x.size())
        shpexpand = ndim * (None,)
        pi = self.pi[shpexpand]
        px = torch.exp(self._component_logpdf(x))  # ... x num_components
        return torch.log(torch.sum(pi * px, -1))


class FixedNormal(Distribution):
    # takes loc and logvar as float values and assumes they are shared across all dimenisions
    def __init__(self, loc, logvar):
        self.loc = loc
        self.logvar = logvar
        super(FixedNormal, self).__init__()

    def logpdf(self, x, eps=0.0):
        c = - float(0.5 * math.log(2 * math.pi))
        return c - 0.5 * self.logvar - (x - self.loc).pow(2) / (2 * math.exp((self.logvar)) + eps)


def distribution_selector(loc, logvar, pi):
    if isinstance(logvar,(list,tuple)) and isinstance(pi, (list,tuple)):
        assert len(logvar) == len(pi)
        num_components = len(logvar)
        if not isinstance(loc, (list,tuple)):
            loc = (loc,) * num_components
        return FixedMixtureNormal(loc, logvar, pi)
    else:
        return FixedNormal(loc, logvar)