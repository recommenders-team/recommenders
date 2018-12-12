#!/bin/bash
# wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh  
# bash ~/miniconda.sh -b -p $HOME/miniconda

# export PATH="$HOME/miniconda/bin:$PATH"
# source $HOME/miniconda/bin/activate

cd examples/notebooks/xDeepFM 

conda env create -f environment.yml

source activate xDeepFM-criteo

bash ./run_notebooks.sh
