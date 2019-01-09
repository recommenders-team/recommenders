# Recommenders 

The Microsoft Recommenders repository on github provides examples and best practices for building
recommendation systems. This package provides a demonstration of using
the smart adaptive recommender (SAR) algorithm for building a
recommendation engine.

The hard work is done using the utilities provided in
[reco_utils](reco_utils) to support common tasks such as loading
datasets in the format expected by different algorithms, evaluating
model outputs, and splitting train/test data. Implementations of
several state-of-the-art algorithms are provided for self-study and
customization in your own applications.

# Usage

MLHub is a command line utility to quickly demonstrate the
capabilities of different machine learning models and data science
processes. Visit [MLHub.ai](https://mlhub.ai) for details.

To install and run the pre-built scripts:

    $ pip3 install mlhub
    $ ml install   gjwgit/sar
    $ ml configure sar
    $ ml demo      sar

