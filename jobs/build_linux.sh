#!/bin/bash
wget https://repo.continuum.io/miniconda/Miniconda3-3.7.0-Linux-x86_64.sh -O ~/miniconda.sh  
bash ~/miniconda.sh -b -p $HOME/miniconda

echo 'debug.1'
source $HOME/miniconda/bin/activate
echo 'debug.2'

export PATH="$HOME/miniconda/bin:$PATH"

echo $PATH
cd examples/notebooks/xDeepFM

echo 'debug.3'
conda env create -f environment.yml
source activate xDeepFM-criteo

bash ./run_notebooks.sh


