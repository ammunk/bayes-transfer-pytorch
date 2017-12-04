import os
import pickle

import torch
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def listdict2dictlist(LD):
    DL = dict(zip(sorted(LD[0].keys()),zip(*[[v for k,v in sorted(d.items())]for d in LD])))
    return DL

class Logger(object):
    def __init__(self):
        pass

    def initialise(self):
        pass

    def dump(self):
        pass


class WeightLogger(Logger):
    def __init__(self):
        super(WeightLogger, self).__init__()

    def initialise(self, expname):
        weights_dir = os.path.join(expname, 'weights')
        figure_dir = os.path.join(expname, 'figures')

        self.diagnosticsfile = os.path.join(expname, 'diagnostics.pkl')
        self.weightsfile = os.path.join(weights_dir, 'model_epoch%i.pkl')
        self.histfigurefile = os.path.join(figure_dir, 'weighthistogram_epoch%i.png')

        if not os.path.exists(expname): os.makedirs(expname)
        if not os.path.exists(weights_dir): os.makedirs(weights_dir)
        if not os.path.exists(figure_dir): os.makedirs(figure_dir)

    def dump(self, model, epoch, diagnostics, *args):
        weights = model.state_dict()
        for k in weights:
            weights[k] = weights[k].cpu()
        self._plothist(model, self.histfigurefile % (epoch), args[0])

        if diagnostics is not None:
            diagnostics_batch_train, diagnostics_batch_valid, diagnostics_batch_valid_MAP = diagnostics

            with open(self.diagnosticsfile, 'wb') as fh:
                pickle.dump({'train': listdict2dictlist(diagnostics_batch_train),
                             'valid': listdict2dictlist(diagnostics_batch_valid),
                             'validMAP': listdict2dictlist(diagnostics_batch_valid_MAP)}, fh)

        with open(self.weightsfile % (epoch), 'wb') as fh:
            pickle.dump(weights, fh)

    def _plothist(self, model, filename, p_logvar_init):
        N = norm(loc=0, scale=np.exp(p_logvar_init))
        x = np.linspace(-0.5, 0.5, 100)
        W = torch.cat([model.layers[0].qw_mean.view(-1), model.layers[2].qw_mean.view(-1),
                       ]).data.cpu().numpy()
        b = torch.cat([model.layers[0].qb_mean.view(-1), model.layers[2].qb_mean.view(-1),
                       ]).data.cpu().numpy()

        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        _ = plt.hist(W, np.linspace(-0.5, 0.5, 100), normed=True, label='q samples')
        plt.plot(x, N.pdf(x), label='prior pdf')
        plt.xlim([-0.5, 0.5])
        plt.ylim([0, 10])
        plt.legend()
        plt.title('Weights')

        plt.subplot(122)
        _ = plt.hist(b, np.linspace(-0.5, 0.5, 100), normed=True, label='q samples')
        plt.plot(x, N.pdf(x), label='prior pdf')
        plt.xlim([-0.5, 0.5])
        plt.ylim([0, 10])
        plt.legend()
        plt.title('bias')

        plt.savefig(filename)
        plt.close('all')

class PrintLogger(Logger):
    def __init__(self):
        super(PrintLogger, self).__init__()

    def initialise(self, expname):
        self.logfile = os.path.join(expname, 'logfile.txt')
        with open(self.logfile, 'w') as fh: fh.write('')

    def dump(self, epoch, diagnostics):
        diagnostics_batch_train, diagnostics_batch_valid, diagnostics_batch_valid_MAP = diagnostics

        ltr = "Train %i | " % (epoch) + ", ".join(["%s: %s" % (k, "|".join(["%0.3f" % (_v) for _v in v])) for k, v in
                                               sorted(diagnostics_batch_train[-1].items())])
        lte = "Test %i | " % (epoch) + ", ".join(["%s: %s" % (k, "|".join(["%0.3f" % (_v) for _v in v])) for k, v in
                                              sorted(diagnostics_batch_valid[-1].items())])
        ltemap = "MAP-Test %i | " % (epoch) + ", ".join(["%s: %s" % (k, "|".join(["%0.3f" % (_v) for _v in v])) for k, v in
                                                     sorted(diagnostics_batch_valid_MAP[-1].items())])

        print('\n'.join([ltr, lte, ltemap]))
        with open(self.logfile, 'a') as fh: fh.write('\n'.join([ltr, lte, ltemap]) + '\n')