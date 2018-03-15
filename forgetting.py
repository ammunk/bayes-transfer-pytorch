# Imports and declarations
import torch
import gc
import sys
import numpy as np
sys.path.append("bayesian_transfer")
from bayesian_transfer.models import BBBMLP
from experiment_forgetting import get_data
from tqdm import tqdm
import math
from bayesian_transfer.layers import GaussianVariationalInference
from torch.autograd import Variable
import os

cuda = torch.cuda.is_available()
num_epochs = 250
num_samples = 16


def get_model(num_output, num_hidden=100, num_layers=2, num_flows=0, pretrained=None):
    model = BBBMLP(in_features=784, num_class=num_output, num_hidden=num_hidden, num_layers=num_layers, nflows=num_flows, p_logvar_init = -2)

    if pretrained:
        d = pretrained.state_dict()
        model.load_prior(d)

    return model


# Training Loop. Here we are forward declaring the loss function
def run_epoch(model, loader, epoch, is_training=False):
    # Number of mini.batches
    m = math.ceil(len(loader.dataset) / loader.batch_size)

    diagnostics = {"accuracy": [], "likelihood": [],
                   "KL": [], "loss": []}

    for i, (data, labels) in enumerate(tqdm(loader)):
        # Repeat samples
        x = data.view(-1, 784).repeat(num_samples, 1)
        y = labels.repeat(num_samples)

        if cuda:
            x = x.cuda()
            y = y.cuda()

        # Blundell Beta-scheme
        beta = 2 ** (m - (i + 1)) / (2 ** m - 1)

        # Calculate loss
        logits, kl = model.probforward(Variable(x))
        loss = loss_fn(logits, Variable(y), kl, beta)
        ll = -loss.data.mean() + beta * kl.data.mean()

        # Update gradients
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute accuracy
        _, predicted = logits.max(1)
        accuracy = (predicted.data == y).float().mean()

        diagnostics["accuracy"].append(accuracy / m)
        diagnostics["loss"].append(loss.data.mean() / m)
        diagnostics["KL"].append(beta * kl.data.mean() / m)
        diagnostics["likelihood"].append(ll / m)

    return diagnostics


logfile = os.path.join('diagnostics.txt')
with open(logfile, "w") as fh:
    fh.write("")

"""
train on domain A
"""

digits = [0, 1, 2, 3, 4]
loader_train, loader_val = get_data(digits, fraction=1.0)

model_a = get_model(5, num_hidden=400, num_layers=2, num_flows=0)

# Define the objective, in this case we want to minimize the negative free free energy.
loss_fn = GaussianVariationalInference(torch.nn.CrossEntropyLoss())
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_a.parameters()), lr=1e-3)

if cuda: model_a.cuda()

for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch, num_epochs))
    diagnostics_train = run_epoch(model_a, loader_train, epoch, is_training=True)
    diagnostics_val = run_epoch(model_a, loader_val, epoch)

    diagnostics_train = dict({"type": "train", "epoch": epoch}, **diagnostics_train)
    diagnostics_val = dict({"type": "validation", "epoch": epoch}, **diagnostics_val)
    diagnostics = diagnostics_train, diagnostics_val

    # Save model and diagnostics
    print(diagnostics)

    with open(logfile, 'a') as fh:
        fh.write(str(diagnostics))


torch.save(model_a.state_dict(), "weights_A.pkl")

"""
transfer weights from domain A and train on domain B


digits = [5, 6, 7, 8, 9]
loader_train, loader_val = get_data(digits, fraction=1.0)

model_b = get_model(5, num_hidden=400, num_layers=2, num_flows=0, pretrained=model_a)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_b.parameters()), lr=1e-3)

if cuda: model_b.cuda()

for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch, num_epochs))
    diagnostics_train = run_epoch(model_b, loader_train, epoch, is_training=True)
    diagnostics_val = run_epoch(model_b, loader_val, epoch)

    diagnostics_train = dict({"type": "train", "epoch": epoch}, **diagnostics_train)
    diagnostics_val = dict({"type": "validation", "epoch": epoch}, **diagnostics_val)
    diagnostics = diagnostics_train, diagnostics_val

    # Save model and diagnostics
    print(diagnostics)

    with open(logfile, 'a') as fh:
        fh.write(str(diagnostics))


torch.save(model_b.state_dict(), "weights_B.pkl")

"""

"""
validate on domain A after domain both domains A and B were trained

digits = [0, 1, 2, 3, 4]
loader_train, loader_val = get_data(digits, fraction=1.0)

model_av = get_model(5, num_hidden=400, num_layers=2, num_flows=0, pretrained=model_b)

if cuda: model_av.cuda()
    
for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch, num_epochs))
    diagnostics_val = run_epoch(model_av, loader_val, epoch)

    diagnostics_val = dict({"type": "validation", "epoch": epoch}, **diagnostics_val)

    # Save model and diagnostics
    print(diagnostics_val)
    
    with open(logfile, 'a') as fh:
        fh.write(str(diagnostics))


torch.save(model_av.state_dict(), "weights_B.pkl")
"""