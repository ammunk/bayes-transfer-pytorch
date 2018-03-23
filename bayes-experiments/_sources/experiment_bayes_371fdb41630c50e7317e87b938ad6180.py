import math
import gc
import pickle
import os
import torch
from torch.autograd import Variable
from sacred import Experiment
from sacred.observers import FileStorageObserver

from ingredients import data_ingredient, load_mnist
from bayesian_transfer.layers import GaussianVariationalInference
from bayesian_transfer.models import BBBMLP

cuda = torch.cuda.is_available()

ex = Experiment("Bayesian Deep Transfer Learning", ingredients=[data_ingredient])
observer = FileStorageObserver.create("bayes-experiments")
ex.observers.append(observer)


@ex.named_config
def normflow():
    num_flows = 16
    beta_type = "Blundell"


@ex.named_config
def domain_a():
    digits = [0,1,2,3,4,5,6,7,8,9]  # not rotated
    beta_type = "Blundell"
    rotation = 0


@ex.named_config
# do this first and see if val acc is high
def transfer_b():
    digits = [0,1,2,3,4,5,6,7,8,9]  # up tp 15° randomly rotated
    beta_type = "Blundell"
    pretrained = "bayes-experiments/1"
    rotation = 15


@ex.named_config
# do this second and see if network can remember domain a
def forgetting_b():
    digits = [0,1,2,3,4,5,6,7,8,9]  # not rotated
    beta_type = "Blundell"
    pretrained = "bayes-experiments/1"
    rotation = 0
    # add argument for not training again, only validate
    is_training = False


@ex.named_config
# do this first and see if val acc is high
def transfer_c():
    digits = [0,1,2,3,4,5,6,7,8,9]  # up to 30° randomly rotated
    beta_type = "Blundell"
    pretrained = "results/transfer_b"
    rotation = 30


@ex.named_config
# do this second and see if network can remember domain b
def forgetting_c():
    digits = [0,1,2,3,4,5,6,7,8,9]  # not rotated and randomly up to 15°
    beta_type = "Blundell"
    pretrained = "results/transfer_b"
    rotation = 15   # includes not rotated
    # add argument for not training again, only validate
    is_training = False


@ex.named_config
# do this first and see if val acc is high
def transfer_d():
    digits = [0,1,2,3,4,5,6,7,8,9]  # up to 45° randomly rotated
    beta_type = "Blundell"
    pretrained = "results/transfer_c"
    rotation = 45


@ex.named_config
# do this second and see if network can remember domain c
def forgetting_d():
    digits = [0,1,2,3,4,5,6,7,8,9]  # not rotated and randomly up to 30°
    beta_type = "Blundell"
    pretrained = "results/transfer_c"
    rotation = 30   # includes not rotated and up to 15°
    # add argument for not training again, only validate
    is_training = False


@ex.named_config
def beta_blundell():
    beta_type = "Blundell"


@ex.named_config
def beta_standard():
    beta_type = "Standard"


@ex.named_config
def beta_soenderby():
    beta_type = "Soenderby"


@ex.named_config
def beta_none():
    beta_type = "None"


def model_definition(num_output, num_hidden=100, num_layers=2, num_flows=0, pretrained=None, p_logvar_init = 0, q_logvar_init=-8):
    model = BBBMLP(in_features=784, num_class=num_output, num_hidden=num_hidden, num_layers=num_layers, nflows=num_flows, p_logvar_init = p_logvar_init, q_logvar_init=q_logvar_init)

    if pretrained:
        d = pickle.load(open(os.path.join(pretrained, "weights.pkl"), "rb"))
        model.load_prior(d)

    return model


@ex.automain
def main(digits=list(range(10)), fraction=1.0, rotation=0, pretrained=None, num_samples=10, num_flows=0, beta_type="Blundell",
         num_layers=2, num_hidden=400, num_epochs=200, p_logvar_init = 0, q_logvar_init=-8, lr=1e-5):

    loader_train, loader_val = load_mnist(digits, fraction, rotation)

    model = model_definition(len(digits), num_hidden=num_hidden, num_layers=num_layers, num_flows=num_flows, pretrained=pretrained, p_logvar_init=p_logvar_init, q_logvar_init=q_logvar_init)

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

        for i, (data, labels) in enumerate(loader):
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
        diagnostics_train = run_epoch(loader_train, epoch, is_training=True)
        diagnostics_val = run_epoch(loader_val, epoch)

        diagnostics_train = dict({"type": "train", "epoch": epoch}, **diagnostics_train)
        diagnostics_val = dict({"type": "validation", "epoch": epoch}, **diagnostics_val)

        print(diagnostics_train)
        print(diagnostics_val)

        if epoch == num_epochs-1:
            weightsfile = os.path.join(observer.dir, 'weights.pkl')
            with open(weightsfile, 'wb') as fh:
                pickle.dump(model.state_dict(), fh)

            ex.add_artifact(weightsfile)

        gc.collect()
