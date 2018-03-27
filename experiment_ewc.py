import os
import pickle
import torch
import torch.nn as nn
from sacred import Experiment
from sacred.observers import FileStorageObserver

from ingredients import data_ingredient, load_mnist
from ewc import ElasticWeightConsolidation

cuda = torch.cuda.is_available()

ex = Experiment("Elastic Weight Consolidation", ingredients=[data_ingredient])
observer = FileStorageObserver.create("ewc-experiments")
ex.observers.append(observer)


@ex.named_config
def domain_A():
    digits = [0, 1, 2, 3, 4]


@ex.named_config
def domain_B():
    digits = [5, 6, 7, 8, 9]


@ex.named_config
def transfer_A_B():
    pretrained = "ewc-experiments/3"
    digits = [5, 6, 7, 8, 9]
    lambd = 5.


@ex.capture
def model_definition(digits=(1,), pretrained=None):
    # The model need to be overparametrized
    # in order to find a sufficient minimum for
    # multitask learning.
    num_digits = len(digits)

    model = nn.Sequential(nn.Linear(784, 512),
                          nn.ReLU(inplace=True),
                          nn.Linear(512, 256),
                          nn.ReLU(inplace=True),
                          nn.Linear(256, 128),
                          nn.ReLU(inplace=True),
                          nn.Linear(128, num_digits))

    if pretrained:
        d = pickle.load(open(os.path.join(pretrained, "weights.pkl"), "rb"))
        model.load_state_dict(d)

    return model


@ex.automain
def main(digits=(1,), lambd=0.0, pretrained=None, num_epochs=101, lr=3e-4):
    if pretrained is not None:
        import json
        model = model_definition(digits, pretrained)

        # Load the configuration for the digits
        # of the previous exoeriment.
        config = json.load(open(os.path.join(pretrained, "config.json")))
        loader, _ = load_mnist(config["digits"])

        if cuda: model.cuda()
        ewc = ElasticWeightConsolidation(model).calculate_fisher(loader)
    else:
        model = model_definition(digits)
        if cuda: model.cuda()

    loader_train, loader_val = load_mnist(digits)

    objective = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    def run_epoch(loader, is_training=False):
        if is_training:
            model.train()
        else:
            model.eval()

        total_loss, accuracy = 0, 0

        for batch_idx, (x, y) in enumerate(loader):
            if cuda:
                x = x.cuda()
                y = y.cuda()

            x = torch.autograd.Variable(x)
            y = torch.autograd.Variable(y)

            optimizer.zero_grad()

            logits = model(x)

            loss = objective(logits, y)

            # Add the EWC penalty to the loss
            if pretrained is not None:
                loss += lambd * ewc.penalty()

            if is_training:
                loss.backward()
                optimizer.step()

            _, predicted = logits.max(1)
            accuracy += (predicted.data == y.data).float().mean()
            total_loss += loss.data.mean()

        return {'loss': total_loss/(batch_idx+1), 'acc': accuracy/(batch_idx+1)}

    for epoch in range(num_epochs):
        diagnostics_train = run_epoch(loader_train, is_training=True)
        diagnostics_val = run_epoch(loader_val)

        diagnostics_train = dict({"type": "train", "epoch": epoch}, **diagnostics_train)
        diagnostics_val = dict({"type": "validation", "epoch": epoch}, **diagnostics_val)

        print(diagnostics_train)
        print(diagnostics_val)

        if epoch == num_epochs-1:
            weightsfile = os.path.join(observer.dir, 'weights.pkl')
            with open(weightsfile, 'wb') as fh:
                pickle.dump(model.state_dict(), fh)

            ex.add_artifact(weightsfile)
