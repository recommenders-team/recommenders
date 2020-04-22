# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
from reco_utils.recommender.tfidf.tfidf_utils import TfidfRecommender

import pandas as pd
import scipy

CLEAN_COL = 'cleaned_text'
K = 2

@pytest.fixture(scope='module')
def df():
    mock_text = {
        'cord_uid': ['ej795nks','9mzs5dl4','u7lz3spe'],
        'doi': ['10.1289/ehp.7117','10.1289/ehp.7491','10.1371/journal.pmed.0030149'],
        'title': ['Understanding the Spatial Clustering of','The Application of the Haddon Matrix to','Cynomolgus Macaque as an Animal Model for'],
        'authors': ['Lai, P.C.; Wong, C.M.; Hedley, A.J.; Lo,','Barnett, Daniel J.; Balicer, Ran D.;','Lawler, James V; Endy, Timothy P; Hensley,'],
        'journal': ['Environ Health Perspect','Environ Health Perspect','PLoS Med'],
        'abstract': ['We applied cartographic and geostatistical met','State and local health departments continue to','BACKGROUND: The emergence of severe acute resp.'],
        'publish_time': ['2004-07-27','2005-02-02','2006-04-18'],
        'url': ['https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11','https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12','https://www.ncbi.nlm.nih.gov/pmc/articles/PMC13'],
        'full_text': ['Since the emergence and rapid spread of the e...','sudden fever and dry cough, along with chills','The emergence of severe acute respiratory syndrome (SARS)']
    }
    return pd.DataFrame(mock_text)

@pytest.fixture(scope='module')
def model():
    return TfidfRecommender(id_col='cord_uid',tokenization_method='scibert')

def test_init(model):
    assert model.id_col == 'cord_uid'
    assert model.tokenization_method == 'scibert'

def test_clean_dataframe(model, df):
    df_clean = model.clean_dataframe(df, ['abstract','full_text'], new_col_name=CLEAN_COL)

    isalphanumeric = list()
    for idx, _ in df_clean.iterrows():
        s1 = str(df_clean[CLEAN_COL][idx])
        isalphanumeric.append(s1.replace(' ','').isalnum())
    
    assert False not in isalphanumeric

@pytest.fixture(scope='module')
def df_clean(model, df):
    return model.clean_dataframe(df, ['abstract','full_text'], new_col_name=CLEAN_COL)

def test_tokenize_text(model, df_clean):
    _, vectors_tokenized = model.tokenize_text(df_clean)
    assert True not in list(df_clean[CLEAN_COL] == vectors_tokenized)

def test_fit(model, df_clean):
    tf, vectors_tokenized = model.tokenize_text(df_clean)
    model.fit(tf, vectors_tokenized)
    assert type(model.tfidf_matrix)==scipy.sparse.csr.csr_matrix

@pytest.fixture(scope='module')
def model_fit(model, df_clean):
    model_fit = TfidfRecommender(id_col='cord_uid',tokenization_method='scibert')
    tf, vectors_tokenized = model_fit.tokenize_text(df_clean)
    model_fit.fit(tf, vectors_tokenized)

    return model_fit

def test_get_tokens(model_fit):
    tokens = model_fit.get_tokens()
    assert type(tokens) == dict
    assert type(list(tokens.keys())[0]) == str

def test_get_stop_words(model_fit):
    stop_words = model_fit.get_stop_words()
    assert type(list(stop_words)[0]) == str

def test_recommend_top_k_items(model_fit, df_clean):
    top_k_recommendations = model_fit.recommend_top_k_items(df_clean, k=K)
    assert len(top_k_recommendations) > len(df_clean)

def test_get_top_k_recommendations(model_fit, df_clean):
    query_id = 'ej795nks'
    displayed_top_k = model_fit.get_top_k_recommendations(df_clean, query_id=query_id)
    assert len(displayed_top_k.data) == K