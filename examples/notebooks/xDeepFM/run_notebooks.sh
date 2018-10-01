#!/bin/bash

# restore environment
conda env create -f environment.yml
conda activate xDeepFM-criteo

pip install papermill

# run notebook
papermill criteo.ipynb criteo-subsampled.ipynb -p max_rows 10000 -p data_url 'https://marcozocriteodata.blob.core.windows.net/criteo/dac-subsampled.tar.gz'

