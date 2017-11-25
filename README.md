# Bayesian deep transfer learning

This repository implements Bayes by Backprop [(Blundell 2015)](https://arxiv.org/abs/1505.05424)
along with some neat methods to increase performance. The goal of the
project is to understand how learning a posterior distribution
over some task can help in transfering to a new domain by using
the learnt posterior as a prior.

## Implemented methods

* Bayes by Backprop MLP [(Blundell 2015)](https://arxiv.org/abs/1505.05424)
* Normalizing flows [(Rezende 2015)](https://arxiv.org/abs/1505.05770)
* Bayes by Backprop Convolutional network

## How to run on Google Compute Engine

After creating a project go to **VM instances** and create a new instance.
For machine type choose atleast 4 CPU cores and 16 GB of memory and choose
the number of GPUs under advanced settings - 1 K80 will usually do.

For boot disk choose Ubuntu 14.04 and increase the disk size to atleast 40 GB.
If you do not have any project wide SSH-keys, you will need to add them inside the
management tab by copy-pasting your public key.

When the machine is up, it will give you an external IP-address from which you
can SSH into, for example:

```
ssh -i ~/.ssh/id_rsa 'your-username-for-the-key'@'external-ip'
```

When inside the machine, after cloning the repository, and being in the folder bayes_02456, you will need to run
the install script to get CUDA 8 and PyTorch.

```
chmod +x gcloud.sh
sudo sh ./gcloud.sh
```

This should install everything and you can run the `main.py` file.
