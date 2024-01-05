.. _recommender:

Recommender algorithms module
##############################

Recommender system algorithms and utilities.

Cornac utilities
******************************
.. automodule:: recommenders.models.cornac.cornac_utils
    :members:


DeepRec utilities
******************************
Base model
==============================
.. automodule:: recommenders.models.deeprec.models.base_model
    :members:
    :special-members: __init__

Sequential base model
==============================
.. automodule:: recommenders.models.deeprec.models.sequential.sequential_base_model
    :members:
    :special-members: __init__

Iterators
==============================
.. automodule:: recommenders.models.deeprec.io.iterator
    :members:
    :special-members: __init__
.. automodule:: recommenders.models.deeprec.io.dkn_iterator
    :members:
    :special-members: __init__
.. automodule:: recommenders.models.deeprec.io.dkn_item2item_iterator
    :members:
    :special-members: __init__
.. automodule:: recommenders.models.deeprec.io.nextitnet_iterator
    :members:
    :special-members: __init__
.. automodule:: recommenders.models.deeprec.io.sequential_iterator
    :members:
    :special-members: __init__

Data processing utilities
==============================
.. automodule:: recommenders.models.deeprec.DataModel.ImplicitCF
    :members:
    :special-members: __init__

Utilities
==============================
.. automodule:: recommenders.models.deeprec.deeprec_utils
    :members:
    :special-members: __init__, __repr__


DKN
******************************
.. automodule:: recommenders.models.deeprec.models.dkn
    :members:
    :special-members: __init__


DKN item-to-item
******************************
.. automodule:: recommenders.models.deeprec.models.dkn_item2item
    :members:
    :special-members: __init__


xDeepFM
******************************
.. automodule:: recommenders.models.deeprec.models.xDeepFM
    :members:
    :special-members: __init__


LightGCN
******************************
.. automodule:: recommenders.models.deeprec.models.graphrec.lightgcn
    :members:
    :special-members: __init__


A2SVD
******************************
.. automodule:: recommenders.models.deeprec.models.sequential.asvd
    :members:
    :special-members: __init__


Caser
******************************
.. automodule:: recommenders.models.deeprec.models.sequential.caser
    :members:
    :special-members: __init__


GRU
******************************
.. automodule:: recommenders.models.deeprec.models.sequential.gru
    :members:
    :special-members: __init__


NextItNet
******************************
.. automodule:: recommenders.models.deeprec.models.sequential.nextitnet
    :members:
    :special-members: __init__


RNN Cells
******************************
.. automodule:: recommenders.models.deeprec.models.sequential.rnn_cell_implement
    :members:
    :special-members: __init__


SUM
******************************
.. automodule:: recommenders.models.deeprec.models.sequential.sum
    :members:
    :special-members: __init__
.. automodule:: recommenders.models.deeprec.models.sequential.sum_cells
    :members:
    :special-members: __init__


SLIRec
******************************
.. automodule:: recommenders.models.deeprec.models.sequential.sli_rec
    :members:
    :special-members: __init__


FastAI utilities
******************************
.. automodule:: recommenders.models.fastai.fastai_utils
    :members:


LightFM utilities
******************************
.. automodule:: recommenders.models.lightfm.lightfm_utils
    :members:


LightGBM utilities
******************************
.. automodule:: recommenders.models.lightgbm.lightgbm_utils
    :members:


NCF
******************************
.. automodule:: recommenders.models.ncf.dataset
    :members:
    :special-members: __init__
.. automodule:: recommenders.models.ncf.ncf_singlenode
    :members:
    :special-members: __init__


NewsRec utilities
******************************
Base model
==============================
.. automodule:: recommenders.models.newsrec.models.base_model
    :members:
    :special-members: __init__

Iterators
==============================
.. automodule:: recommenders.models.newsrec.io.mind_iterator
    :members:
    :special-members: __init__
.. automodule:: recommenders.models.newsrec.io.mind_all_iterator
    :members:
    :special-members: __init__


Utilities
==============================
.. automodule:: recommenders.models.newsrec.models.layers
    :members:
    :special-members: __init__
.. automodule:: recommenders.models.newsrec.newsrec_utils
    :members:
    :special-members: __init__


LSTUR
******************************
.. automodule:: recommenders.models.newsrec.models.lstur
    :members:
    :special-members: __init__


NAML
******************************
.. automodule:: recommenders.models.newsrec.models.naml
    :members:
    :special-members: __init__


NPA
******************************
.. automodule:: recommenders.models.newsrec.models.npa
    :members:
    :special-members: __init__


NRMS
******************************
.. automodule:: recommenders.models.newsrec.models.nrms
    :members:
    :special-members: __init__


RBM
******************************
.. automodule:: recommenders.models.rbm.rbm
    :members:
    :special-members: __init__

.. FIXME: Fix Pymanopt dependency. Issue #2038
.. GeoIMC
.. ******************************
.. .. automodule:: recommenders.models.geoimc.geoimc_algorithm
..     :members:
..     :special-members: __init__
.. .. automodule:: recommenders.models.geoimc.geoimc_data
..     :members:
..     :special-members: __init__
.. .. automodule:: recommenders.models.geoimc.geoimc_predict
..     :members:
.. .. automodule:: recommenders.models.geoimc.geoimc_utils
..     :members:


.. FIXME: Fix Pymanopt dependency. Issue #2038
.. RLRMC
.. ******************************
.. .. automodule:: recommenders.models.rlrmc.RLRMCdataset
..     :members:
..     :special-members: __init__
.. .. automodule:: recommenders.models.rlrmc.RLRMCalgorithm
..     :members:
..     :special-members: __init__
.. .. automodule:: recommenders.models.rlrmc.conjugate_gradient_ms
..     :members:
..     :special-members: __init__



SAR
******************************
.. automodule:: recommenders.models.sar.sar_singlenode
    :members:
    :special-members: __init__


SASRec 
******************************
.. automodule:: recommenders.models.sasrec.model
    :members:
    :special-members: __init__
.. automodule:: recommenders.models.sasrec.sampler
    :members:
    :special-members: __init__
.. automodule:: recommenders.models.sasrec.util
    :members:


SSE-PT 
******************************
.. automodule:: recommenders.models.sasrec.ssept
    :members:
    :special-members: __init__


Surprise utilities
******************************
.. automodule:: recommenders.models.surprise.surprise_utils
    :members:


TF-IDF utilities
******************************
.. automodule:: recommenders.models.tfidf.tfidf_utils
    :members:


Standard VAE
******************************
.. automodule:: recommenders.models.vae.standard_vae
    :members:
    :special-members: __init__


Multinomial VAE
******************************
.. automodule:: recommenders.models.vae.multinomial_vae
    :members:
    :special-members: __init__


Vowpal Wabbit utilities
******************************
.. automodule:: recommenders.models.vowpal_wabbit.vw
    :members:


Wide & Deep
******************************
.. automodule:: recommenders.models.wide_deep.wide_deep_utils
    :members:
    :special-members: __init__