from functools import reduce
from operator import __or__
import os
import errno

import torch
import torch.utils.data as data
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np


class LimitedMNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset with digit limitation.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        digits (list, optional): Digits to include in dataset.
    """

    def __init__(self, root, set_type="train", transform=None, target_transform=None, digits=None, fraction=1.0):
        self.transform = transform
        self.target_transform = target_transform

        self.mnist = fetch_mldata("MNIST original", data_home=root)

        np.random.seed(seed=1337)
        indices = np.arange(len(self.mnist.target))
        np.random.shuffle(indices)

        self.mnist.data = self.mnist.data[indices]
        self.mnist.target = self.mnist.target[indices]

        if set_type is "train":
            self.mnist.data = self.mnist.data[:40000]
            self.mnist.target = self.mnist.target[:40000]

            if fraction < 1.0:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=fraction)
                indices = list(sss.split(self.mnist.data, self.mnist.target))[0][1]

                self.mnist.data = self.mnist.data[indices]
                self.mnist.target = self.mnist.target[indices]
        elif set_type is "validation":
            self.mnist.data = self.mnist.data[40000:]
            self.mnist.target = self.mnist.target[40000:]

        # Filter digits
        if digits is not None:
            indices = reduce(__or__, [self.mnist.target == i for i in digits])
            self.mnist.data = self.mnist.data[indices]
            self.mnist.target = self.mnist.target[indices]


    def __getitem__(self, index):
        img, target = self.mnist.data[index], self.mnist.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        img = torch.FloatTensor((img/255).astype(np.float32))
        target = torch.LongTensor([int(target)])

        return img, target

    def __len__(self):
        return len(self.mnist.target)


class OMNIGLOT(data.Dataset):
    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    '''
    The items are (filename,category). The index of all the categories can be found in self.idx_classes

    Args:

    - root: the directory where the dataset will be stored
    - transform: how to transform the input
    - target_transform: how to transform the target
    - download: need to download the dataset

    Usage:

    class FilenameToTensor(object):
        """
        Load a PIL RGB Image from a filename.
        """
        def __call__(self, filename):
            img = Image.open(filename).convert('L')
            img = PIL.ImageOps.invert(img)
            img.thumbnail((28, 28), Image.ANTIALIAS)
            img = np.array(img).reshape(-1).astype(np.float32)
            return torch.FloatTensor(img)

    omniglot = OMNIGLOT("./", download=True, transform=FilenameToTensor())
    '''
    def __init__(self, root, transform=None, target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.'
                               + ' You can use download=True to download it')

        self.all_items=find_classes(os.path.join(self.root, self.processed_folder))
        self.idx_classes=index_classes(self.all_items)

    def __getitem__(self, index):
        filename=self.all_items[index][0]
        img=str.join('/',[self.all_items[index][2],filename])

        target=self.idx_classes[self.all_items[index][1]]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return  img,target

    def __len__(self):
        return len(self.all_items)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, "images_evaluation")) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, "images_background"))

    def download(self):
        from six.moves import urllib
        import zipfile

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            file_processed = os.path.join(self.root, self.processed_folder)
            print("== Unzip from "+file_path+" to "+file_processed)
            zip_ref = zipfile.ZipFile(file_path, 'r')
            zip_ref.extractall(file_processed)
            zip_ref.close()
        print("Download finished.")

def find_classes(root_dir):
    retour=[]
    for (root,dirs,files) in os.walk(root_dir):
        for f in files:
            if (f.endswith("png")):
                r=root.split('/')
                lr=len(r)
                retour.append((f,r[lr-2]+"/"+r[lr-1],root))
    print("== Found %d items "%len(retour))
    return retour

def index_classes(items):
    idx={}
    for i in items:
        if (not i[1] in idx):
            idx[i[1]]=len(idx)
    print("== Found %d classes"% len(idx))
    return idx
