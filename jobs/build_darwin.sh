#!/bin/bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh  
bash ~/miniconda.sh -b -p $HOME/miniconda

export PATH="$HOME/miniconda/bin:$PATH"
source $HOME/miniconda/bin/activate

echo $PATH
cd examples/notebooks/xDeepFM 

pwd

echo 'creating environment.1'
conda env create -f environment.yml
echo 'creating environment.2'

conda info --envs

echo 'trying to activate'

conda activate xDeepFM-criteo

bash ./run_notebooks.sh
