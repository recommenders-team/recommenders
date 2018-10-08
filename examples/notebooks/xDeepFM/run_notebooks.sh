#!/bin/bash

# run notebook
papermill criteo.ipynb criteo-subsampled.ipynb -p max_rows 10000 -p data_url 'https://marcozocriteodata.blob.core.windows.net/criteo/dac-subsampled.tar.gz'

