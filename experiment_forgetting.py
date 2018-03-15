import math
import os
import gc
import pickle
from functools import reduce
from operator import __or__

import numpy
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
from sacred import Experiment

from bayesian_transfer.layers import GaussianVariationalInference
from bayesian_transfer.models import BBBMLP

cuda = torch.cuda.is_available()

def get_model(num_output, num_hidden=100, num_layers=2, num_flows=0, pretrained=None, p_logvar_init = 0, q_logvar_init=-8):
    model = BBBMLP(in_features=784, num_class=num_output, num_hidden=num_hidden, num_layers=num_layers, nflows=num_flows, p_logvar_init = p_logvar_init, q_logvar_init=q_logvar_init)

    if pretrained:
        d = pickle.load(open(pretrained + "/weights.pkl", "rb"))
        model.load_prior(d)

    return model


def get_data(digits, fraction):
    target_transform = lambda x: {str(digit): k for digit, k in zip(digits, range(len(digits)))}[str(int(x))]
    
    convert = transforms.Compose([
            # convert PIL image to numpy
            transforms.Lambda(lambda x: numpy.array(x)),
            # divide values by 126 like Blundell => make data sets i.i.d.
            transforms.Lambda(lambda x: x.astype(numpy.float32)/ 126),
            # return tensor
            torch.from_numpy
            ])
    """
    mnist_train = MNIST(root="./", download=True, transform=transforms.Compose([transforms.Normalize(0.485, 0.229), 
                                                                                torch.from_numpy]),                                                              
                        target_transform=target_transform)
                        
    mnist_valid = MNIST(root="./", train=False, download=True, transform=transforms.Compose([transforms.Normalize(0.485, 0.229), 
                                                                                torch.from_numpy]), 
                        target_transform=target_transform)
    
    """
    mnist_train = MNIST(root="./", download=True, transform=convert,
                        target_transform=target_transform)
    mnist_valid = MNIST(root="./", train=False, download=True, transform=convert,
                        target_transform=target_transform) 
    
    def get_sampler(labels):
        (indices,) = numpy.where(reduce(__or__, [labels == i for i in digits]))
        indices = torch.from_numpy(numpy.random.permutation(indices))
        sampler = SubsetRandomSampler(indices[:int(len(indices)*fraction)])
        return sampler


    loader_train = DataLoader(mnist_train, batch_size=128, num_workers=2, pin_memory=cuda,
                              sampler=get_sampler(mnist_train.train_labels.numpy()))

    loader_valid = DataLoader(mnist_valid, batch_size=128, num_workers=2, pin_memory=cuda,
                              sampler=get_sampler(mnist_valid.test_labels.numpy()))

    return loader_train, loader_valid


ex = Experiment("Bayesian Deep Transfer Learning")


@ex.named_config
def blundell():
    experiment_name = "results/blundell"
    beta_type = "Blundell"
    num_epochs = 600


@ex.named_config
def normflow():
    experiment_name = "results/normflow"
    num_flows = 16
    num_epochs = 600
    q_logvar_init = -8
    lr=1e-5
    beta_type = "Blundell"


@ex.named_config
def domain_a():
    digits = [0,1,2,3,4]
    experiment_name = "results/domain_a"
    beta_type = "Blundell"
    num_epochs = 300


@ex.named_config
def domain_b():
    digits = [5,6,7,8,9]
    experiment_name = "results/domain_b"
    beta_type = "Blundell"
    num_epochs = 600


@ex.named_config
def transfer():
    digits = [5,6,7,8,9]
    p_logvar_init = 0
    q_logvar_init=-8
    lr=1e-5
    pretrained = "results/domain_a"
    beta_type = "Blundell"
    num_epochs = 300

@ex.named_config
def forgetting():
    digits = [0,1,2,3,4]
    p_logvar_init = 0
    q_logvar_init=-8
    lr=1e-5
    pretrained = "results/domain_b"
    beta_type = "Blundell"
    num_epochs = 300


@ex.named_config
def beta_blundell():
    experiment_name="results/beta_blundell"
    beta_type = "Blundell"
    num_epochs = 600


@ex.named_config
def beta_standard():
    experiment_name = "results/beta_standard"
    beta_type = "Standard"
    num_epochs = 600


@ex.named_config
def beta_soenderby():
    experiment_name = "results/beta_soenderby"
    beta_type = "Soenderby"
    num_epochs = 600


@ex.named_config
def beta_none():
    experiment_name = "results/beta_none"
    beta_type = "None"
    num_epochs = 600

@ex.automain
def main(experiment_name, digits=list(range(10)), fraction=1.0, pretrained=None, num_samples=10, num_flows=0, beta_type="Blundell",
         num_layers=2, num_hidden=400, num_epochs=101, p_logvar_init = 0, q_logvar_init=-8, lr=1e-5):
    if not os.path.exists(experiment_name): os.makedirs(experiment_name)
    logfile = os.path.join(experiment_name, 'diagnostics.txt')
    with open(logfile, "w") as fh:
        fh.write("")

    loader_train, loader_val = get_data(digits, fraction)

    model = get_model(len(digits), num_hidden=num_hidden, num_layers=num_layers, num_flows=num_flows, pretrained=pretrained, p_logvar_init=p_logvar_init, q_logvar_init=q_logvar_init)

    vi = GaussianVariationalInference(torch.nn.CrossEntropyLoss())
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    if cuda:
        model.cuda()

    def run_epoch(loader, epoch, is_training=False):
        m = math.ceil(len(loader.dataset) / loader.batch_size)

        accuracies = []
        likelihoods = []
        kls = []
        losses = []

        for i, (data, labels) in enumerate(tqdm(loader)):
            # Repeat samples
            x = data.view(-1, 784).repeat(num_samples, 1)
            y = labels.repeat(num_samples)

            if cuda:
                x = x.cuda()
                y = y.cuda()

            if beta_type == "Blundell":
                beta = 2 ** (m - (i + 1)) / (2 ** m - 1)
            elif beta_type == "Soenderby":
                beta = min(epoch / (num_epochs//4), 1)
            elif beta_type == "Standard":
                beta = 1 / m
            else:
                beta = 0

            logits, kl = model.probforward(Variable(x))
            loss = vi(logits, Variable(y), kl, beta)
            ll = -loss.data.mean() + beta*kl.data.mean()

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            _, predicted = logits.max(1)
            accuracy = (predicted.data == y).float().mean()

            accuracies.append(accuracy)
            losses.append(loss.data.mean())
            kls.append(beta*kl.data.mean())
            likelihoods.append(ll)

        diagnostics = {'loss': sum(losses)/len(losses),
                       'acc': sum(accuracies)/len(accuracies),
                       'kl': sum(kls)/len(kls),
                       'likelihood': sum(likelihoods)/len(likelihoods)}

        return diagnostics

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        diagnostics_train = run_epoch(loader_train, epoch, is_training=True)
        diagnostics_val = run_epoch(loader_val, epoch)

        diagnostics_train = dict({"type": "train", "epoch": epoch}, **diagnostics_train)
        diagnostics_val = dict({"type": "validation", "epoch": epoch}, **diagnostics_val)

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
