# Documentation

To setup the documentation, first you need to install the dependencies of the full environment. For it please follow the [SETUP.md](../SETUP.md). Then type:

    conda create -n reco_full python=3.6 cudatoolkit=10.0 cudnn>=7.6
    conda activate reco_full
    pip install -e .[gpu,spark,examples]
    pip install sphinx_rtd_theme


To build the documentation as HTML:

    cd docs
    make html

To contribute to this repository, please follow these [guidelines](https://github.com/Microsoft/Recommenders/wiki/Coding-Guidelines).