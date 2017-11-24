# This script is designed to work with ubuntu 16.04 LTS
# with the latest Pytorch with CUDA 8 support
#
# Based on https://gist.github.com/motiur/2e0cd3d35bc1c42b6e5d6046b65be6f4
##########################################################################
#This is used to install CUDA 8 driver for Tesla K80
##########################################################################

#!/bin/bash
echo "Checking for CUDA and installing."
# Check for CUDA and try to install.
if ! dpkg-query -W cuda-8-0; then
  # The 16.04 installer works with 16.10.
  curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  apt-get update
  apt-get install cuda-8-0 -y
fi

#########################################################################

#############################################################################
#Updating the system
#############################################################################

sudo apt-get update
sudo apt-get --assume-yes upgrade
sudo apt-get --assume-yes install tmux build-essential gcc g++ make binutils
sudo apt-get --assume-yes install software-properties-common

#########################################################################################################################
#Installing anaconda with the required packages
#########################################################################################################################

wget "https://repo.continuum.io/archive/Anaconda3-4.3.0-Linux-x86_64.sh" -O "Anaconda3-4.3.0-Linux-x86_64.sh"
bash Anaconda3-4.3.0-Linux-x86_64.sh -b
echo "export PATH=\"$HOME/anaconda3/bin:\$PATH\"" >> ~/.bashrc
export PATH="$HOME/anaconda3/bin:$PATH"
conda install -y bcolz
conda upgrade -y --all

#########################################################################################################################
#Installing Jupyter notebook
#########################################################################################################################

# configure jupyter and prompt for password
jupyter notebook --generate-config
jupass=`python -c "from notebook.auth import passwd; print(passwd())"`
echo "c.NotebookApp.password = u'"$jupass"'" >> $HOME/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False" >> $HOME/.jupyter/jupyter_notebook_config.py
echo "\"jupyter notebook\" will start Jupyter on port 8888"
echo "If you get an error instead, try restarting your session so your $PATH is updated"

#########################################################################################################################
#Installing google compute engine package, unzip package, and gensim package
#########################################################################################################################

sudo apt-get install unzip
pip install --upgrade gensim
pip install google-compute-engine

#########################################################################################################################
#Installing PyTorch from source
#########################################################################################################################

export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" # [anaconda root directory]

# Install basic dependencies
conda install -y numpy pyyaml mkl setuptools cmake cffi

# Add LAPACK support for the GPU
conda install -y -c soumith magma-cuda80 # or magma-cuda75 if CUDA 7.5
conda install -y tqdm

# Download master branch of PyTorch
git clone --recursive https://github.com/pytorch/pytorch

# Install
cd pytorch
python setup.py install

cd ..

source ~.profile

conda create -n torch --clone="/home/$USER/anaconda3"
source activate torch

conda install -y tqdm
pip install torchvision
