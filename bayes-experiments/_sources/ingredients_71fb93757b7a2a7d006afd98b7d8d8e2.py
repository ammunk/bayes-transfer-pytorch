from functools import reduce
from operator import __or__
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import MNIST


cuda = torch.cuda.is_available()

from sacred import Ingredient

data_ingredient = Ingredient("dataset")

@data_ingredient.config
def cfg():
    digits = list(range(10))
    fraction = 1.0
    rotation = 0

@data_ingredient.capture
def load_mnist(digits, fraction=1.0, rotation=0):
    target_transform = lambda x: {str(digit): k for digit, k in zip(digits, range(len(digits)))}[str(int(x))]
    
    if rotation is 0:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)),
                                        transforms.Lambda(lambda x: x.view(28**2))
                                        ])

    if rotation is 15:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)),
                                        transforms.Lambda(lambda x: x.view(28 ** 2)),
                                        transforms.RandomRotation(15)
                                        ])

    if rotation is 30:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)),
                                        transforms.Lambda(lambda x: x.view(28 ** 2)),
                                        transforms.RandomRotation(30)
                                        ])

    if rotation is 45:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)),
                                        transforms.Lambda(lambda x: x.view(28 ** 2)),
                                        transforms.RandomRotation(45)
                                        ])

    mnist_train = MNIST(root="./", download=True, transform=transform,
                        target_transform=target_transform)
    mnist_valid = MNIST(root="./", train=False, download=True, transform=transform,
                        target_transform=target_transform)    
      
    def get_sampler(labels, n=None):
        # Only choose digits in n_labels
        (indices,) = np.where(reduce(__or__, [labels == i for i in digits]))

        # Ensure uniform distribution of labels
        np.random.shuffle(indices)
        indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n] for i in digits])

        indices = torch.from_numpy(indices)
        sampler = SubsetRandomSampler(indices)
        return sampler

    loader_train = DataLoader(mnist_train, batch_size=128, num_workers=2, pin_memory=cuda,
                              sampler=get_sampler(mnist_train.train_labels.numpy()))

    loader_valid = DataLoader(mnist_valid, batch_size=128, num_workers=2, pin_memory=cuda,
                              sampler=get_sampler(mnist_valid.test_labels.numpy()))

    return loader_train, loader_valid
