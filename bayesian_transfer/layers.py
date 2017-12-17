import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from .distributions import Normal, distribution_selector


class BBBLinearFactorial(nn.Module):
    """
    Describes a Linear fully connected Bayesian layer with
    a distribution over each of the weights and biases
    in the layer.
    """
    def __init__(self, in_features, out_features, p_logvar_init=-3, p_pi=1.0, q_logvar_init=-5, nflows=0):
        # p_logvar_init, p_pi can be either
        #    (list/tuples): prior model is a mixture of gaussians components=len(p_pi)=len(p_logvar_init)
        #    float: Gussian distribution
        # q_logvar_init: float, the approximate posterior is currently always a factorized gaussian
        super(BBBLinearFactorial, self).__init__()

        self.in_features   = in_features
        self.out_features  = out_features
        self.p_logvar_init = p_logvar_init
        self.q_logvar_init = q_logvar_init

        # Normalizing flows
        self.nflows = nflows
        self.normalizing_flow_w = NormalizingFlows(n=self.nflows, features=in_features*out_features)
        self.normalizing_flow_b = NormalizingFlows(n=self.nflows, features=out_features)

        # Approximate Posterior model
        self.qw_mean   = Parameter(torch.Tensor(out_features, in_features))
        self.qw_logvar = Parameter(torch.Tensor(out_features, in_features))
        self.qb_mean   = Parameter(torch.Tensor(out_features))
        self.qb_logvar = Parameter(torch.Tensor(out_features))

        self.qw = Normal(mu=self.qw_mean, logvar=self.qw_logvar)
        self.qb = Normal(mu=self.qb_mean, logvar=self.qb_logvar)

        # Prior Model (the prior model does not have any trainable parameters so we use special versions of the normal distributions)
        self.pw = distribution_selector(mu=0.0, logvar=p_logvar_init, pi=p_pi)
        self.pb = distribution_selector(mu=0.0, logvar=p_logvar_init, pi=p_pi)

        # initialize all paramaters
        self.reset_parameters()

    def reset_parameters(self):
        # initialize (learnable) approximate posterior parameters
        stdv = 10. / math.sqrt(self.in_features)
        self.qw_mean.data.uniform_(-stdv, stdv)
        self.qw_logvar.data.uniform_(-stdv, stdv).add_(self.q_logvar_init)
        self.qb_mean.data.uniform_(-stdv, stdv)
        self.qb_logvar.data.uniform_(-stdv, stdv).add_(self.q_logvar_init)

    def forward(self, input):
        raise NotImplementedError()

    def probforward(self, input, MAP=False):
        """
        Probabilistic forwarding method.
        :param input: data tensor
        :param MAP: boolean whether to take the MAP estimate.
        :return: output, kl-divergence
        """
        if MAP:
            w_sample = self.qw.mu
            b_sample = self.qb.mu
        else:
            w_sample = self.qw.sample()
            b_sample = self.qb.sample()

        if self.nflows > 0:
            f_w_sample, log_det_w = self.normalizing_flow_w(w_sample.view(1, -1))
            f_b_sample, log_det_b = self.normalizing_flow_b(b_sample.view(1, -1))

            f_w_sample = f_w_sample.view(w_sample.size())
            f_b_sample = f_b_sample.view(b_sample.size())

            # Subtracting log det J is the same as multiplying by 1/(det J)
            qw_logpdf = self.qw.logpdf(w_sample) - sum(log_det_w)
            qb_logpdf = self.qb.logpdf(b_sample) - sum(log_det_b)
        else:
            qw_logpdf = self.qw.logpdf(w_sample)
            qb_logpdf = self.qb.logpdf(b_sample)

            f_w_sample = w_sample
            f_b_sample = b_sample

        kl_w = torch.sum(qw_logpdf - self.pw.logpdf(f_w_sample))
        kl_b = torch.sum(qb_logpdf - self.pb.logpdf(f_b_sample))
        kl = kl_w + kl_b

        output = F.linear(input, f_w_sample, f_b_sample)

        return output, kl

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GaussianVariationalInference(nn.Module):
    def __init__(self, loss=nn.CrossEntropyLoss()):
        super(GaussianVariationalInference, self).__init__()
        self.loss = loss

    def forward(self, logits, y, kl, beta):
        logpy = -self.loss(logits, y)  # sample average

        ll = logpy - beta * kl  # ELBO
        loss = -ll

        return loss


class NormalizingFlows(nn.Module):
    def __init__(self, n, features):
        super(NormalizingFlows, self).__init__()
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
        self.u = Parameter(torch.randn(features))
        self.w = Parameter(torch.randn(features))
        self.b = Parameter(torch.ones(1))

    def forward(self, z):
        # Create uhat such that it is parallel to w
        uw = torch.dot(self.u, self.w)
        muw = -1 + F.softplus(uw)
        uhat = self.u + (muw - uw) * torch.transpose(self.w, 0, -1) / torch.sum(self.w ** 2)

        # Equation 21 - Transform z
        zwb = torch.mv(z, self.w) + self.b

        f_z = z + (uhat.view(1, -1) * F.tanh(zwb).view(-1, 1))

        # Compute the Jacobian using the fact that
        # tanh(x) dx = 1 - tanh(x)**2
        psi = (1 - F.tanh(zwb)**2).view(-1, 1) * self.w.view(1, -1)
        psi_u = torch.mv(psi, uhat)

        # Return the transformed output along
        # with log determninant of J
        logdet_jacobian = torch.log(torch.abs(1 + psi_u) + 1e-8)

        return f_z, logdet_jacobian