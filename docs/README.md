# Documentation

To setup the documentation, first you need to install the dependencies of the full environment. For it please follow the [SETUP.md](../SETUP.md). Then type:

    conda create -n reco_full -c conda-forge python=3.7 cudatoolkit=11.2 cudnn=8.1
    conda activate reco_full

    pip install numpy cython
    pip install --no-binary scikit-surprise "scikit-surprise@https://github.com/NicolasHug/Surprise/archive/refs/tags/v1.1.1.tar.gz"
    pip install "pymanopt@https://github.com/pymanopt/pymanopt/archive/fb36a272cdeecb21992cfd9271eb82baafeb316d.zip"
    pip install sphinx_rtd_theme


To build the documentation as HTML:

    cd docs
    make html

To contribute to this repository, please follow our [coding guidelines](https://github.com/Microsoft/Recommenders/wiki/Coding-Guidelines). See also the [reStructuredText documentation](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html) for the syntax of docstrings.
