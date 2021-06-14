.. _dataset:

Dataset module
##############

Recommendation datasets and related utilities

Recommendation datasets 
***********************

Amazon Reviews
==============

.. automodule:: reco_utils.dataset.amazon_reviews
    :members:

Azure COVID-19
==============

.. automodule:: reco_utils.dataset.covid_utils
    :members:

Criteo
======

.. automodule:: reco_utils.dataset.criteo
    :members:

MIND
====

.. automodule:: reco_utils.dataset.mind
    :members:  

MovieLens
=========

The MovieLens datasets, first released in 1998, describe people’s expressed preferences
for movies. These preferences take the form of `<user, item, rating, timestamp>` tuples, 
each the result of a person expressing a preference (a 0-5 star rating) for a movie
at a particular time.

It comes with several sizes:

* MovieLens 100k: 100,000 ratings from 1000 users on 1700 movies.
* MovieLens 1M: 1 million ratings from 6000 users on 4000 movies.
* MovieLens 10M: 10 million ratings from 72000 users on 10000 movies.
* MovieLens 20M: 20 million ratings from 138000 users on 27000 movies

Citation::

    F. M. Harper and J. A. Konstan. "The MovieLens Datasets: History and Context". 
    ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19, 
    DOI=http://dx.doi.org/10.1145/2827872, 2015.

.. automodule:: reco_utils.dataset.movielens
    :members:

Download utilities 
******************

.. automodule:: reco_utils.dataset.download_utils
    :members:


Cosmos CLI utilities
*********************

.. automodule:: reco_utils.dataset.cosmos_cli
    :members:


Pandas dataframe utilities
***************************

.. automodule:: reco_utils.dataset.pandas_df_utils
    :members:


Splitter utilities
******************

.. automodule:: reco_utils.dataset.python_splitters
    :members:

.. automodule:: reco_utils.dataset.spark_splitters
    :members:

.. automodule:: reco_utils.dataset.split_utils
    :members:


Sparse utilities
****************

.. automodule:: reco_utils.dataset.sparse
    :members:
  

Knowledge graph utilities
*************************

.. automodule:: reco_utils.dataset.wikidata
    :members:
