#!/bin/bash
wget https://repo.continuum.io/miniconda/Miniconda3-3.7.0-Linux-x86_64.sh -O ~/miniconda.sh  
bash ~/miniconda.sh -b -p $HOME/miniconda

export PATH="$HOME/miniconda/bin:$PATH"
source $HOME/miniconda/bin/activate

echo $PATH
cd examples/notebooks/xDeepFM

conda env create -f environment.yml
source activate xDeepFM-criteo

bash ./run_notebooks.sh


