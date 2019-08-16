from flask import abort
import config
import pyodbc
import requests
import json
import urllib.parse
from werkzeug.security import check_password_hash
from error import *

class Database:
    def __init__(self):
        self.cursor = pyodbc.connect('DRIVER=' + config.driver + ';SERVER=' + config.server + ';PORT=1433;DATABASE=' + config.database + ';UID=' + config.username + ';PWD=' + config.password).cursor()
    
    def db_lookup(self, selection, condition, query):
        baseQuery = "SELECT " + selection + " FROM MOVIES WHERE " + condition + " = ?"
        self.cursor.execute(baseQuery, query)
        try:
            result = self.cursor.fetchall()
            return result[0][0]
        except:
            raise LookupError('Movie lookup failed')

    def get_tmdb_info(self, df):
        # Contains and adds poster_path, description
        tmbdb_id = [self.db_lookup('tmdbID', 'movieID', x) for x in df['ItemID']]
        mov_info = [self.tmdb_search(x) for x in tmbdb_id]
        df['Imageurl'] = [entry[0] for entry in mov_info]
        df['Overview'] = [entry[1] for entry in mov_info]
        return df

    """
    Sends GET request to TMDB API for the first movie result from the query and returns 
    the movie poster and description.
    """
    def tmdb_search(self, query):
        payload = {
            'api_key': config.tmdb_key,
            }
        r = requests.get('https://api.themoviedb.org/3/movie/{}'.format(query), params=payload)
        try:
            movie = json.loads(r.text)
            poster_path = None
            if movie['poster_path']:
                poster_path = "https://image.tmdb.org/t/p/w500" + movie['poster_path']
            description = movie['overview']
            return [poster_path, description]
        except:
            return [None, None]

    """
    Sends GET request to Azure Search Service and returns movies that have a movie title similar to the user input
    """
    def movie_search(self, user_input, page=1):
        try: 
            user_input = urllib.parse.quote(user_input, safe='')
            url = config.search_url + user_input + "*&$count=true&$top=20&$skip=" + str((page - 1) * 2)
            json_movies = requests.get(url)
            movie_list = json.loads(json_movies.text)
        except:
            LookupError('Azure Search failed')
        movie_dict  = {"Movies" : []}
 
        for movie in movie_list['value']:
            # Gets the movie picture and description
            [poster, description] = self.tmdb_search(movie['tmdbID'])
            
            # Creates new json object while adding poster and description with the same format as frontend
            movie_object = {"Genre": movie['movieGenres'].split('|'), "Imageurl": poster, "ItemID": int(movie['movieID']), "Overview": description, 
                            "Title": movie['movieTitle'], "Year": movie['movieYear']}
            movie_dict['Movies'].append(movie_object)

        # Convert movie dictionary into JSON object and return it
        movie_json = json.dumps(movie_dict)
        return movie_json
