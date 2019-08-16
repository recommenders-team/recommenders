import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import pandas as pd
from algorithm import Algorithm
from error import *
from mock import patch
import json

# Initial trivial test to check compilations
def test_trivial():
	assert 1 == 1

class AlgTestDB:
	def db_lookup(selection, condition, query):
		pass

alg = Algorithm(AlgTestDB)

def test_deserialize_empty_genre():
	obj = {'Genre': {'0': ''}}
	assert alg.deserialize_genres(obj) == {'Genre': {'0': ['']}}

def test_deserialize_wrong_genre():
	obj = []
	with pytest.raises(TypeError) as excinfo:
		assert alg.deserialize_genres(obj)

def test_deserialize_no_genre():
	obj = {}
	with pytest.raises(KeyError) as excinfo:
		assert alg.deserialize_genres(obj)

def test_deerialize_correct():
	obj = {'Genre': {'0': 'Adventure|Fantasy', '1': "Action|Children's|Fantasy"}}
	assert alg.deserialize_genres(obj) == {'Genre': {'0': ['Adventure', 'Fantasy'], '1': ['Action', "Children's", 'Fantasy']}}

def test_empty_ids():
	id_list = []
	with pytest.raises(RecommendationError) as excinfo:
		assert alg.get_sar_recommendations(id_list)
	with pytest.raises(RecommendationError) as excinfo:
		assert alg.get_lgbm_recommendations(id_list)

def test_wrong_ids():
	# Only testing ints now since data is transformed from the frontend to ints
	# Error handling is in Database class so we must manually raise the exception
	id_list = ['100000']
	with pytest.raises(LookupError) as excinfo:
		assert alg.get_sar_recommendations(id_list)

	def bad_lookup(selection, condition, query):
		# Database error from trying execute a select that doesn't exist
		raise LookupError('Movie lookup failed')

	with patch.object(AlgTestDB, 'db_lookup', new=bad_lookup):
		alg_lgbm = Algorithm(AlgTestDB)
		with pytest.raises(LookupError) as excinfo:
			assert alg_lgbm.get_lgbm_recommendations(id_list)

# Test for when tmdb doesn't have our movie
# Essentially trivial since this raises an error in the db_lookup
def test_rec_without_tmdbinfo():
	# Arbitrary inputs irrelevant for the specific error we want to see
	id_list = ['1']
	class badTMDB_DB:
		# Result of get_tmdb_info from database class when invalid movie id is inputted
		# due to error from the db_lookup part of its original implementation
		def get_tmdb_info(df):
			raise LookupError('Movie lookup failed')

		def db_lookup(selection, condition, query):
			pass

	alg_sar_non_tmdb = Algorithm(badTMDB_DB)
	with pytest.raises(LookupError) as excinfo:
			assert alg_sar_non_tmdb.get_sar_recommendations(id_list)

	def dummy_lookup(selection, condition, query):
		# Some dummy genres that will be accepted
		return "Action|Fantasy"

	with patch.object(badTMDB_DB, 'db_lookup', new=dummy_lookup):
		alg_lgbm_non_tmdb = Algorithm(badTMDB_DB)
		with pytest.raises(LookupError) as excinfo:
			assert alg_lgbm_non_tmdb.get_lgbm_recommendations(id_list)

def test_healthy_sar():
	id_list = [2]

	# healthy mock DB for recommendations using movie 2 as input
	class healthyDB_Mock:
		def get_tmdb_info(df):
			pass

	with open('backend_tests/sar_test_data.json') as f:
		data = json.load(f)

	res = data['Healthy_SAR']
	fixed_titles = data['Sar_Titles']

	# correct database response for movie 2
	def mocked_get_tmdb_info_sar(df):         
		return pd.DataFrame(res)

	with patch.object(healthyDB_Mock, 'get_tmdb_info', new=mocked_get_tmdb_info_sar):
		alg_healthy_sar = Algorithm(healthyDB_Mock)
		for i in range(10):
			assert alg_healthy_sar.get_sar_recommendations(id_list)['json'][i] == {
				"Genre": res['Genre'][i],
				"Imageurl": res['Imageurl'][i],
				"ItemID": res['ItemID'][i],
				"Overview": res['Overview'][i],
				"Title": fixed_titles[i],
				"UserID": res['UserID'][i],
				"Year": res['Year'][i],
				"prediction": res['prediction'][i]
			}

def test_healthy_lgbm():
	id_list = [2]

	# healthy mock DB for recommendations using movie num. 2 as input
	class healthyDB_Mock:
		def db_lookup(selection, condition, query):
			return "Adventure|Children's|Fantasy"

		def get_tmdb_info(df):
			pass

	with open('backend_tests/lgbm_test_data.json') as f:
		data = json.load(f)

	res = data['Healthy_LGBM']
	fixed_titles = data['LGBM_Titles']

	# correct database response for movie 2
	def mocked_get_tmdb_info_lgbm(df):        
		return pd.DataFrame(res)

	with patch.object(healthyDB_Mock, 'get_tmdb_info', new=mocked_get_tmdb_info_lgbm):
		alg_healthy_lgbm = Algorithm(healthyDB_Mock)
		for i in range(10):
			assert alg_healthy_lgbm.get_lgbm_recommendations(id_list)['json'][i] == {
				"Genre": res['Genre'][i],
				"Imageurl": res['Imageurl'][i],
				"ItemID": res['ItemID'][i],
				"Overview": res['Overview'][i],
				"Title": fixed_titles[i],
				"UserID": res['UserID'][i],
				"Year": res['Year'][i],
				"prediction": res['prediction'][i]
			}
