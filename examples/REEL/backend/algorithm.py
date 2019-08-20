import config
import json
import pandas as pd
import requests
import re
from flask import request
from error  import *

class Algorithm:
	def __init__(self, database):
		self.sar_headers = {
			'Content-Type': "application/json",
			'Authorization': config.sar_token,
			}

		self.lgbm_headers = {
			'Content-Type': "application/json",
			'Authorization': config.lgbm_token,
			}
		self.db = database

	# Converts pipe-separated genre string to list
	# Ex: "Action|Comedy" -> ['Action', 'Comedy']
	def deserialize_genres(self, json_object):
		genre_dict = json_object['Genre']
		for key in genre_dict.keys():
			genre_dict[key] = genre_dict[key].split('|')
		return json_object

	def receive_request(self, url, payload, headers):
		return requests.request("POST", url, json=payload, headers=headers)

	def get_sar_recommendations(self, id_list):
		if len(id_list) < 1:
			raise RecommendationError('ID required')

		payload = {"ItemID": id_list}
		response = self.receive_request(config.sar_url, payload, self.sar_headers)

		# We use this negative column error since this is what the ML endpoint returns
		if response.json() == 'negative column index found':
			raise LookupError('Invalid Movie ID')

		response = self.deserialize_genres(response.json())

		df = pd.DataFrame.from_dict(response, orient='columns')

		df = self.db.get_tmdb_info(df)
		
		# Remove year from title
		df['Title'] = df['Title'].apply(lambda x: re.sub(r'\s[(]\d\d\d\d[)]$', '',x))

		df['json'] = df.apply(lambda x: json.loads(x.to_json()), axis=1)
		
		return df

	def get_lgbm_recommendations(self, id_list):
		if len(id_list) < 1:
			raise RecommendationError('ID required')

		payload = {"UserID": 0, "key": 0}
		with open('data/genre_list.txt') as file:
			for line in file:
				payload[line.rstrip()] = 0

		movie_genres = [self.db.db_lookup('movieGenres', 'movieID', x).split("|") for x in id_list]

		for movie_list in movie_genres:
			for genre in movie_list:
				payload["genre_" + genre] += 1

		factor = 1 / sum(payload.values())
		for key in payload:
			payload[key] *= factor 

		response = self.receive_request(config.lgbm_url, payload, self.lgbm_headers)

		response = self.deserialize_genres(response.json())

		df = pd.DataFrame.from_dict(response, orient='columns')

		df = self.db.get_tmdb_info(df)

		# Remove year from title
		df['Title'] = df['Title'].apply(lambda x: re.sub(r'\s[(]\d\d\d\d[)]$', '',x))

		df['json'] = df.apply(lambda x: json.loads(x.to_json()), axis=1)

		return df
