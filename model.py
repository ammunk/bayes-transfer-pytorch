import torch.nn as nn
from torch.autograd import Variable

from distributions import Normal
from layers import BBBLinearFactorial


class BBBMLP(nn.Module):
    def __init__(self, in_features, num_class, num_hidden, num_layers, p_logvar_init=0, p_pi=1.0, q_logvar_init=-5,
                 nflows=0):
        # create a simple MLP model with probabilistic weights
        super(BBBMLP, self).__init__()
        layers = [
            BBBLinearFactorial(in_features, num_hidden, p_logvar_init=p_logvar_init, p_pi=p_pi,
                               q_logvar_init=q_logvar_init, nflows=nflows), nn.ELU()]
        for i in range(num_layers - 1):
            layers += [BBBLinearFactorial(num_hidden, num_hidden, p_logvar_init=p_logvar_init,
                                          p_pi=p_pi, q_logvar_init=q_logvar_init, nflows=nflows), nn.ELU()]

        layers += [BBBLinearFactorial(num_hidden, num_class, p_logvar_init=p_logvar_init,
                                      p_pi=p_pi, q_logvar_init=q_logvar_init, nflows=nflows)]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x, MAP=False):
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'probforward') and callable(layer.probforward):
                # Get intermediate diagnostics
                x, _kl, = layer.probforward(x, MAP=MAP)
                kl += _kl
            else:
                x = layer(x)
        logits = x
        return logits, kl

    def load_prior(self, state_dict):
        d_q = {k: v for k, v in state_dict.items() if "q" in k}
        for i, layer in enumerate(self.layers):
            if type(layer) is BBBLinearFactorial:
                layer.pw = Normal(mu=Variable(d_q["layers.{}.qw_mean".format(i)]),
                                  logvar=Variable(d_q["layers.{}.qw_logvar".format(i)]))

                layer.pb = Normal(mu=Variable(d_q["layers.{}.qb_mean".format(i)]),
                                  logvar=Variable(d_q["layers.{}.qb_logvar".format(i)]))
