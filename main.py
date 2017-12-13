import gc
import pickle

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from auxiliary import merge_add, merge_average
from bbbmlp import BBBMLP, BBBLinearFactorial
from datasets import LimitedMNIST
from distributions import Normal
from loggers import PrintLogger, WeightLogger

###### Hyperparameters ###### (you _might_ want to change this)
cuda        = torch.cuda.is_available()
NUM_EPOCHS  = 50
SAVE_EVERY  = 9
N_SAMPLES   = 1
LR          = 1e-3
MNIST       = "./"

p_logvar_init = 0.
q_logvar_init = -5.

file_logger = WeightLogger()
print_logger = PrintLogger()

number_of_flows = 0

# Define network
def train_model(filename, digits=[0], fraction=1.0, pretrained=False):
    mnist_train = LimitedMNIST(root=MNIST, set_type="train", target_transform=lambda x: x - min(digits),
                               digits=digits, fraction=fraction)

    mnist_val = LimitedMNIST(root=MNIST, set_type="validation", target_transform=lambda x: x - min(digits),
                             digits=digits, fraction=fraction)

    batch_size = 100
    loader_train = DataLoader(mnist_train, batch_size=int(batch_size*fraction), num_workers=2, pin_memory=cuda)
    loader_val = DataLoader(mnist_val, batch_size=int(batch_size*fraction), num_workers=2, pin_memory=cuda)

    model = BBBMLP(in_features=784, num_class=len(digits), num_hidden=100, num_layers=2,
                   p_logvar_init=p_logvar_init, p_pi=1.0, q_logvar_init=q_logvar_init, nflows=number_of_flows)

    if pretrained:
        path = "original/weights/model_epoch49.pkl"
        d = pickle.load(open(path, "rb"))
        d_q = {k: v for k, v in d.items() if "q" in k}
        for i, layer in enumerate(model.layers):
            if type(layer) is BBBLinearFactorial:
                layer.pw = Normal(mu=Variable(d_q["layers.{}.qw_mean".format(i)]),
                                  logvar=Variable(d_q["layers.{}.qw_logvar".format(i)]))

                layer.pb = Normal(mu=Variable(d_q["layers.{}.qb_mean".format(i)]),
                                  logvar=Variable(d_q["layers.{}.qb_logvar".format(i)]))

    print(model)

    file_logger.initialise(filename)
    print_logger.initialise(filename)

    # Create optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    if cuda:
        model.cuda()

    def run_epoch(loader, MAP=False, is_training=False):
        diagnostics = {}
        nbatch_per_epoch = len(loader.dataset) / loader.batch_size

        for i, (data, labels) in tqdm(enumerate(loader)):
            # Repeat samples
            x = data.repeat(N_SAMPLES, 1, 1, 1)
            y = labels.repeat(N_SAMPLES, 0)
            x = x.view(-1, 784)

            if cuda:
                x = x.cuda()
                y = y.cuda()

            logits, loss, _diagnostics = model.getloss(Variable(x), Variable(y),
                                                       dataset_size=nbatch_per_epoch, MAP=MAP)

            diagnostics = merge_add(diagnostics, _diagnostics)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        diagnostics = merge_average(diagnostics, i+1)
        return diagnostics


    diagnostics_batch_train, diagnostics_batch_valid, diagnostics_batch_valid_MAP = [], [], []

    file_logger.dump(model, -1, None, p_logvar_init)

    for epoch in range(NUM_EPOCHS):
        diagnostics_batch_train += [run_epoch(loader_train, is_training=True)]
        diagnostics_batch_valid += [run_epoch(loader_val)]
        diagnostics_batch_valid_MAP += [run_epoch(loader_val, MAP=True)]

        batch_diagnostics = [diagnostics_batch_train, diagnostics_batch_valid, diagnostics_batch_valid_MAP]

        if epoch % SAVE_EVERY == 0:
            file_logger.dump(model, epoch, batch_diagnostics, p_logvar_init)

        print_logger.dump(epoch, batch_diagnostics)

        gc.collect()

    file_logger.dump(model, epoch, batch_diagnostics, p_logvar_init)

###### Parameters for experiment ######

digits = [1, 2]
transfer = [7, 8]

# Train the model on the whole data of digits
#train_model("original", digits, fraction=1.0)
#
# train_model("domain0.05", transfer, fraction=0.05, pretrained=False)
# train_model("domain0.1", transfer, fraction=0.1, pretrained=False)
# train_model("domain0.2", transfer, fraction=0.2, pretrained=False)
# train_model("domain0.3", transfer, fraction=0.3, pretrained=False)
# train_model("domain0.5", transfer, fraction=0.5, pretrained=False)
# train_model("domain1", transfer, fraction=1.0, pretrained=False)

# Transfer to the second domain with the trained model
train_model("transfer_domain0.05", transfer, fraction=0.05, pretrained=True)
train_model("transfer_domain0.1", transfer, fraction=0.1, pretrained=True)
train_model("transfer_domain0.2", transfer, fraction=0.2, pretrained=True)
train_model("transfer_domain0.3", transfer, fraction=0.3, pretrained=True)
train_model("transfer_domain0.5", transfer, fraction=0.5, pretrained=True)
train_model("transfer_domain1", transfer, fraction=1.0, pretrained=True)
