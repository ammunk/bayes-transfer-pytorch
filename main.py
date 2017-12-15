import math
import os
import gc
import pickle

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from sacred import Experiment

from layers import GaussianVariationalInference
from model import BBBMLP
from datasets import LimitedMNIST

cuda = torch.cuda.is_available()

def get_model(num_output, num_flows=0, pretrained=None):
    model = BBBMLP(in_features=784, num_class=num_output, num_hidden=100, num_layers=2, nflows=num_flows)

    if pretrained:
        d = pickle.load(open(pretrained + "/weights.pkl", "rb"))
        model.load_prior(d)

    if cuda: model.cuda()

    return model


def get_data(digits=[0, 1], fraction=1.0):
    target_transform = lambda x: {str(digit): k for digit, k in zip(digits, range(len(digits)))}[str(int(x))]

    mnist_train = LimitedMNIST(root="./", set_type="train", target_transform=target_transform,
                               digits=digits, fraction=fraction)

    mnist_val = LimitedMNIST(root="./", set_type="validation", target_transform=target_transform,
                             digits=digits, fraction=fraction)

    loader_train = DataLoader(mnist_train, batch_size=128, num_workers=2, pin_memory=cuda)
    loader_val = DataLoader(mnist_val, batch_size=128, num_workers=2, pin_memory=cuda)

    return loader_train, loader_val


ex = Experiment("Bayesian Deep Transfer Learning")


@ex.config
def cfg():
    pretrained = None


@ex.automain
def main(experiment_name, digits, fraction, pretrained=None, n_samples=16, num_flows=0, beta_type="Blundell"):
    if not os.path.exists(experiment_name): os.makedirs(experiment_name)
    logfile = os.path.join(experiment_name, 'diagnostics.txt')

    loader_train, loader_val = get_data(digits, fraction)

    model = get_model(len(digits), num_flows=num_flows, pretrained=pretrained)

    vi = GaussianVariationalInference(torch.nn.CrossEntropyLoss())
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    def run_epoch(loader, epoch, is_training=False):
        m = math.ceil(len(loader.dataset) / loader.batch_size)

        accuracies = []
        losses = []

        for i, (data, labels) in tqdm(enumerate(loader)):
            # Repeat samples
            x = data.view(-1, 784).repeat(n_samples, 1)
            y = labels.repeat(n_samples, 0)

            y = Variable(y)
            x = Variable(x)

            if cuda:
                x = x.cuda()
                y = y.cuda()

            if beta_type is "Blundell":
                beta = 2 ** (m - (i + 1)) / (2 ** m - 1)
            elif beta_type is "Sønderbye":
                beta = min(epoch / 100, 1)
            elif beta_type is "Standard":
                beta = 1 / m
            else:
                beta = 0

            logits, kl = model.probforward(x)
            loss = vi(logits, y, kl, beta)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            _, predicted = logits.max(1)
            accuracy = (predicted.data == y.data).float().mean()

            accuracies.append(accuracy)
            losses.append(loss.data.mean())

        diagnostics = {'loss': sum(losses)/len(losses),
                       'acc': sum(accuracies)/len(accuracies)}

        return diagnostics

    n_epochs = 51
    for epoch in range(n_epochs):
        diagnostics_train = run_epoch(loader_train, epoch, is_training=True)
        diagnostics_val = run_epoch(loader_val, epoch)

        diagnostics_train = dict({"type": "train"}, **diagnostics_train)
        diagnostics_val = dict({"type": "validation"}, **diagnostics_val)

        weightsfile = os.path.join(experiment_name, 'weights.pkl')

        with open(logfile, 'a') as fh:
            fh.write(str(diagnostics_train))
            fh.write(str(diagnostics_val))

        if epoch == n_epochs-1:
            with open(weightsfile, 'wb') as fh:
                pickle.dump(model.state_dict(), fh)

        # Save model and diagnostics
        print(diagnostics_train)
        print(diagnostics_val)

        gc.collect()
