#!/usr/bin/env python

from distutils.core import setup

setup(name='RecoUtils',
      version=VERSION,
      description='Reco Utils',
      author='Microsoft',
      packages=['reco_utils','reco_utils.common','reco_utils.dataset','reco_utils.evaluation','reco_utils.recommender'],
     )
