import gc
import torch as th
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from auxiliary import merge_add, merge_average
from bbbmlp import BBBMLP
from datasets import LimitedMNIST
from loggers import PrintLogger, WeightLogger

# Constants
CUDA        = False
NUM_EPOCHS  = 100
SAVE_EVERY  = 1
N_SAMPLES   = 16
LR          = 1e-3
MNIST       = "./"

p_logvar_init = 0.
q_logvar_init = -5.

# Define datasets and loaders
digits = [0, 1, 2, 3, 4]
mnist_train = LimitedMNIST(root=MNIST, set_type="train", transform=lambda x: x.reshape(-1), digits=digits)
mnist_val = LimitedMNIST(root=MNIST, set_type="validation", transform=lambda x: x.reshape(-1), digits=digits)
mnist_test = LimitedMNIST(root=MNIST, set_type="test", transform=lambda x: x.reshape(-1), digits=digits)

batch_size = 64
loader_train = DataLoader(mnist_train, batch_size=batch_size)
loader_val   = DataLoader(mnist_val, batch_size=batch_size)
loader_test  = DataLoader(mnist_test, batch_size=batch_size)

file_logger = WeightLogger()
print_logger = PrintLogger()

# Define network
def train_model(digits=[0], fraction=1.0, pretrained=False):
    mnist_train = LimitedMNIST(root=MNIST, set_type="train",
                               transform=lambda x: x.reshape(-1), digits=digits, fraction=fraction)

    mnist_val = LimitedMNIST(root=MNIST, set_type="validation",
                             transform=lambda x: x.reshape(-1), digits=digits, fraction=fraction)

    batch_size = 64
    loader_train = DataLoader(mnist_train, batch_size=batch_size)
    loader_val = DataLoader(mnist_val, batch_size=batch_size)


    model = BBBMLP(in_features=784, num_class=len(digits), num_hidden=512, num_layers=2,
                   p_logvar_init=p_logvar_init, p_pi=1.0, q_logvar_init=q_logvar_init)

    filename = 'normal_prior_qlogvar%0.1f_plogvar%0.1f_lr%0.5f_bs%i_ns%i' %\
              (q_logvar_init, p_logvar_init, LR, loader_train.batch_size, N_SAMPLES)

    if type(pretrained) is str:
        model.load_state_dict(torch.load(pretrained))

    file_logger.initialise(filename)
    print_logger.initialise(filename)

    # Create optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    if CUDA:
        model.cuda()

    def run_epoch(loader, MAP=False, is_training=False):
        diagnostics = {}
        nbatch_per_epoch = len(loader.dataset) // loader.batch_size

        for i, (data, labels) in tqdm(enumerate(loader)):
            # Repeat samples
            x = data.repeat(N_SAMPLES, 1)
            y = labels.repeat(N_SAMPLES, 0)

            if CUDA:
                x = x.cuda()
                y = y.cuda()

            logits, loss, _diagnostics = model.getloss(Variable(x), Variable(y),
                                                       dataset_size=len(loader.dataset), MAP=MAP)
            diagnostics = merge_add(diagnostics, _diagnostics)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        diagnostics = merge_average(diagnostics, nbatch_per_epoch)
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

    torch.save("pretrained.pth")


digits = [0, 1, 2, 3, 4]
transfer = [5, 6, 7, 8, 9]

# Train the model on the whole data of digits
train_model(digits, fraction=1.0)

# Transfer to the second domain with the trained model
train_model(transfer, fraction=0.1, pretrained="pretrained.pth")
train_model(transfer, fraction=0.2, pretrained="pretrained.pth")
train_model(transfer, fraction=0.25, pretrained="pretrained.pth")
train_model(transfer, fraction=0.50, pretrained="pretrained.pth")
train_model(transfer, fraction=1.0, pretrained="pretrained.pth")
