import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from collections import defaultdict

from distributions import Normal, distribution_selector


class BBBLinearFactorial(nn.Module):
    def __init__(self, in_features, out_features, p_logvar_init=-3, p_pi=1.0, q_logvar_init=-5):
        # p_logvar_init, p_pi can be either
        #    (list/tuples): prior model is a mixture of gaussians components=len(p_pi)=len(p_logvar_init)
        #    float: Gussian distribution
        # q_logvar_init: float, the approximate posterior is currently always a factorized gaussian
        super(BBBLinearFactorial, self).__init__()

        self.in_features   = in_features
        self.out_features  = out_features
        self.p_logvar_init = p_logvar_init
        self.q_logvar_init = q_logvar_init

        # Approximate Posterior model
        self.qw_mean   = Parameter(torch.Tensor(out_features, in_features))
        self.qw_logvar = Parameter(torch.Tensor(out_features, in_features))
        self.qb_mean   = Parameter(torch.Tensor(out_features))
        self.qb_logvar = Parameter(torch.Tensor(out_features))

        self.qw = Normal(loc=self.qw_mean, logvar=self.qw_logvar)
        self.qb = Normal(loc=self.qb_mean, logvar=self.qb_logvar)

        # Prior Model (the prior model does not have any trainable parameters so we use special versions of the normal distributions)
        self.pw = distribution_selector(loc=0.0, logvar=p_logvar_init, pi=p_pi)
        self.pb = distribution_selector(loc=0.0, logvar=p_logvar_init, pi=p_pi)

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
        # input: BS, in_features
        # W: BS, in_features
        # MAP: maximum a posterior (return the mode instead of sampling from the distributions)
        if MAP:
            w_sample = self.qw.loc
            b_sample = self.qb.loc
        else:
            w_sample = self.qw.sample()
            b_sample = self.qb.sample()

        kl_w = torch.sum(self.qw.logpdf(w_sample) - self.pw.logpdf(w_sample))
        kl_b = torch.sum(self.qb.logpdf(b_sample) - self.pb.logpdf(b_sample))
        kl = kl_w + kl_b

        diagnostics = {'kl_w': kl_w.data.mean(), 'kl_b': kl_b.data.mean(),
                       'Hq_w': self.qw.entropy().data.mean(),
                       'Hq_b': self.qb.entropy().data.mean()}  # Hq_w and Hq_b are the differential entropy
        output = F.linear(input, w_sample, b_sample)

        return output, kl, diagnostics

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class BBBMLP(nn.Module):
    def __init__(self, in_features, num_class, num_hidden, num_layers, p_logvar_init=-3, p_pi=1.0, q_logvar_init=-5):
        # create a simple MLP model with probabilistic weights
        super(BBBMLP, self).__init__()
        layers = [
            BBBLinearFactorial(in_features, num_hidden, p_logvar_init=p_logvar_init, p_pi=p_pi,
                               q_logvar_init=q_logvar_init), nn.ELU()]
        for i in range(num_layers - 1):
            layers += [BBBLinearFactorial(num_hidden, num_hidden, p_logvar_init=p_logvar_init,
                                          p_pi=p_pi, q_logvar_init=q_logvar_init), nn.ELU()]
        layers += [
            BBBLinearFactorial(num_hidden, num_class, p_logvar_init=p_logvar_init, p_pi=p_pi,
                               q_logvar_init=q_logvar_init)]

        self.layers = nn.ModuleList(layers)
        self.loss = nn.CrossEntropyLoss()

    def probforward(self, x, MAP=False):
        diagnostics = defaultdict(list)
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'probforward') and callable(layer.probforward):
                # Get intermediate diagnostics
                x, _kl, _diagnostics = layer.probforward(x, MAP=MAP)
                kl += _kl
                for k, v in _diagnostics.items():
                    diagnostics[k].append(v)
            else:
                x = layer(x)
        logits = x
        return logits, kl, diagnostics

    def getloss(self, x, y, dataset_size, MAP=False):
        logits, kl, _diagnostics = self.probforward(x, MAP=MAP)
        # _diagnostics is here a dictinary of list of floats
        # We need the dataset_size in order to 'spread' the KL divergence across all samples
        # this is dscribed in EQ (8) in Blundel et. al. 2015

        logpy = -self.loss(logits, y)  # sample average
        kl /= dataset_size  # see EQ (8) in Blundell et al 2015

        ll = logpy - kl  # ELBO
        loss = -ll

        _, predicted = logits.max(1)
        acc = (predicted.data == y.data).float().mean()  # accuracy

        # the xxx.data.mean() is just an easy way to transfer to cpu and convert from torch to normal floats
        diagnostics = {'loss': [loss.data.mean()],
                       'll': [ll.data.mean()],
                       'kl': [kl.data.mean()],
                       'logpy': [logpy.data.mean()],
                       'acc': [acc],
                       'kl_w': _diagnostics['kl_w'],
                       'kl_b': _diagnostics['kl_b'],
                       'Hq_w': _diagnostics['Hq_w'],
                       'Hq_b': _diagnostics['Hq_b'], }
        return logits, loss, diagnostics