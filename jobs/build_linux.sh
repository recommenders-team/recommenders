#!/bin/bash
wget -q https://repo.continuum.io/miniconda/Miniconda3-3.7.0-Linux-x86_64.sh -O ~/miniconda.sh  
bash ~/miniconda.sh -b -p $HOME/miniconda

export PATH="$HOME/miniconda/bin:$PATH"
conda create -n xDeepFM-criteo python=3.6
 
# source $HOME/miniconda/bin/activate
export PATH="$HOME/miniconda/envs/xDeepFM-criteo/bin:$PATH"
echo $PATH

# source $HOME/miniconda/bin/activate
# echo $PATH
cd examples/notebooks/xDeepFM

# echo 'debug.3'
# conda env update -f environment.yml
# source activate xDeepFM-criteo

# bash ./run_notebooks.sh


