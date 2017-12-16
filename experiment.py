import math
import os
import gc
import pickle

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from sacred import Experiment

from bayesian_transfer.layers import GaussianVariationalInference
from bayesian_transfer.model import BBBMLP
from bayesian_transfer.datasets import LimitedMNIST

cuda = torch.cuda.is_available()


def get_model(num_output, num_hidden=100, num_layers=2, num_flows=0, pretrained=None):
    model = BBBMLP(in_features=784, num_class=num_output, num_hidden=num_hidden, num_layers=num_layers, nflows=num_flows)

    if pretrained:
        d = pickle.load(open(pretrained + "/weights.pkl", "rb"))
        model.load_prior(d)

    return model


def get_data(digits, fraction):
    target_transform = lambda x: {str(digit): k for digit, k in zip(digits, range(len(digits)))}[str(int(x))]

    mnist_train = LimitedMNIST(root="./", set_type="train", target_transform=target_transform,
                               digits=digits, fraction=fraction)

    mnist_val = LimitedMNIST(root="./", set_type="validation", target_transform=target_transform,
                             digits=digits, fraction=fraction)

    loader_train = DataLoader(mnist_train, batch_size=128, num_workers=2, pin_memory=cuda)
    loader_val = DataLoader(mnist_val, batch_size=128, num_workers=2, pin_memory=cuda)

    return loader_train, loader_val


ex = Experiment("Bayesian Deep Transfer Learning")


@ex.named_config
def blundell():
    experiment_name="results/blundell"
    num_layers = 2
    num_hidden = 400
    num_samples = 10
    num_epochs = 51
    beta_type = "Blundell"


@ex.named_config
def normflow():
    experiment_name = "results/normflow"
    num_layers = 2
    num_hidden = 400
    num_samples = 10
    num_epochs = 51
    beta_type = "Blundell"


@ex.named_config
def domain_a():
    digits = [0, 1, 2, 3, 4]
    experiment_name = "results/domain_a"
    num_layers = 2
    num_hidden = 400
    num_samples = 10
    num_epochs = 51
    beta_type = "Blundell"


@ex.named_config
def domain_b(fraction):
    experiment_name = "results/domain_b_{}".format(fraction)
    num_layers = 2
    num_hidden = 400
    num_samples = 10
    num_epochs = 51
    beta_type = "Blundell"


@ex.named_config
def transfer(fraction):
    experiment_name = "results/domain_b_{}".format(fraction)
    pretrained = "results/domain_a"
    num_layers = 2
    num_hidden = 400
    num_samples = 10
    num_epochs = 51
    beta_type = "Blundell"


@ex.named_config
def beta(beta_type):
    experiment_name = "results/beta_{}".format(beta_type)
    num_layers = 2
    num_hidden = 400
    num_samples = 10
    num_epochs = 51


@ex.automain
def main(experiment_name, digits=list(range(10)), fraction=1.0, pretrained=None, num_samples=16, num_flows=0, beta_type="Blundell",
         num_layers=2, num_hidden=100, num_epochs=51):
    if not os.path.exists(experiment_name): os.makedirs(experiment_name)
    logfile = os.path.join(experiment_name, 'diagnostics.txt')
    with open(logfile, "w") as fh:
        fh.write("")

    loader_train, loader_val = get_data(digits, fraction)

    model = get_model(len(digits), num_hidden=num_hidden, num_layers=num_layers, num_flows=num_flows, pretrained=pretrained)

    vi = GaussianVariationalInference(torch.nn.CrossEntropyLoss())
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    def run_epoch(loader, epoch, is_training=False):
        m = math.ceil(len(loader.dataset) / loader.batch_size)

        accuracies = []
        losses = []

        for i, (data, labels) in enumerate(tqdm(loader)):
            # Repeat samples
            x = data.view(-1, 784).repeat(num_samples, 1)
            y = labels.repeat(num_samples, 0)

            if beta_type == "Blundell":
                beta = 2 ** (m - (i + 1)) / (2 ** m - 1)
            elif beta_type == "Soenderby":
                # Deterministic warmup
                beta = min(epoch / (num_epochs//2), 1)
            elif beta_type == "Standard":
                beta = 1 / m
            else:
                beta = 0

            logits, kl = model.probforward(Variable(x))
            loss = vi(logits, Variable(y), kl, beta)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            _, predicted = logits.max(1)
            accuracy = (predicted.data == y).float().mean()

            accuracies.append(accuracy)
            losses.append(loss.data.mean())

        diagnostics = {'loss': sum(losses)/len(losses),
                       'acc': sum(accuracies)/len(accuracies)}

        return diagnostics

    for epoch in range(num_epochs):
        diagnostics_train = run_epoch(loader_train, epoch, is_training=True)
        diagnostics_val = run_epoch(loader_val, epoch)

        diagnostics_train = dict({"type": "train"}, **diagnostics_train)
        diagnostics_val = dict({"type": "validation"}, **diagnostics_val)

        weightsfile = os.path.join(experiment_name, 'weights.pkl')

        with open(logfile, 'a') as fh:
            fh.write(str(diagnostics_train))
            fh.write(str(diagnostics_val))

        if epoch == num_epochs-1:
            with open(weightsfile, 'wb') as fh:
                pickle.dump(model.state_dict(), fh)

        # Save model and diagnostics
        print(diagnostics_train)
        print(diagnostics_val)

        gc.collect()