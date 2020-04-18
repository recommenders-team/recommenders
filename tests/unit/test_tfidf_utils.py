# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
from reco_utils.recommender.tfidf.tfidf_utils import (
    clean_text,
    clean_dataframe_for_rec,
    tokenize_with_BERT,
    recommend_with_tfidf,
    organize_results_as_tabular,
    get_single_item_info,
    make_clickable,
    display_top_recommendations
)

import pandas as pd

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

def test_clean_text(df):
    # Ensure cleaned text is alphanumeric
    example_text = clean_text(df['full_text'][0])
    assert example_text.replace(' ','').isalnum() == True

def test_clean_dataframe_for_rec(df):
    # Ensure cleaned text in each row is alphanumeric
    df_clean = clean_dataframe_for_rec(df, ['abstract','full_text'])

    isalphanumeric = list()
    for idx, _ in df_clean.iterrows():
        s1 = str(df_clean['cleaned_text'][idx])
        isalphanumeric.append(s1.replace(' ','').isalnum())
    
    assert False not in isalphanumeric

@pytest.fixture(scope='module')
def df_clean(df):
    return clean_dataframe_for_rec(df, ['abstract','full_text'], for_BERT=True)

def test_tokenize_with_BERT(df_clean):
    tokens = tokenize_with_BERT(df_clean['cleaned_text'])
    assert True not in list(df_clean['cleaned_text'] == tokens)

def test_recommend_with_tfidf(df_clean):
    results = recommend_with_tfidf(df_clean)
    first_rec_id = results[list(results.keys())[0]][0][1]
    assert type(results) == dict
    assert list(results.keys()) == list(df_clean['cord_uid'])
    assert first_rec_id in list(df_clean['cord_uid'])

@pytest.fixture(scope='module')
def results(df_clean):
    return recommend_with_tfidf(df_clean)

def test_organize_results_as_tabular(df_clean, results):
    rec_table = organize_results_as_tabular(df_clean, results, k=2)
    assert type(rec_table) == pd.core.frame.DataFrame
    assert len(rec_table) > len(results)
    assert 'rec_score' in rec_table.columns
    assert 'rec_title' in rec_table.columns

def test_get_single_item_info(df):
    rec_info = get_single_item_info(df, rec_id='ej795nks')
    assert rec_info['doi'] == '10.1289/ehp.7117'

def test_make_clickable(df):
    # TODO
    pass

def test_display_top_recommendations(df, df_clean, results):
    rec_table = organize_results_as_tabular(df_clean, results, k=2)
    top_recs = display_top_recommendations(rec_table, df, query_id='ej795nks', verbose=False)
    assert len(top_recs.data) == len(df)-1