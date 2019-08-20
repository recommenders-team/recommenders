from flask import Flask, request, jsonify, redirect, abort, session
import flask_injector
import injector
import providers
import requests
import json
import pandas as pd
import config
import pyodbc
from algorithm import Algorithm
from database import Database
from random import randint
from error import *

'''
Using injections to lessen the burden of dependencies:
    7/29: Database and Algorithm classes

Chose to use injections rather than taking class constructors as
arguments because of requirements of the flask app that I couldn't figure
out how to get around.

The same issue with the Algorithm class was simpler, and a class constructor
used as an argument is implemented there.
'''
INJECTOR_DEFAULT_MODULES = dict(
    db = providers.DatabaseModule(),
    algorithm = providers.AlgorithmModule()
)

def _configure_dependency_injection(
    flask_app, injector_modules, custom_injector
) -> None:
    modules = dict(INJECTOR_DEFAULT_MODULES)

    if injector_modules:
        modules.update(injector_modules)

    flask_injector.FlaskInjector(
        app = flask_app,
        injector = custom_injector,
        modules = modules.values())

def create_app(
    *,
    custom_injector: injector.Injector = None,
    injector_modules = None
):
    app = Flask(__name__)

    """
    Handles invalid JSON requests
    """
    @app.errorhandler(LookupError)
    def handle_database_error(error):
        return error.message, error.status_code

    @app.errorhandler(SelectionError)
    def handle_login_error(error):
        return error.message, error.status_code


    """
    A GET method for returning search results for a given query
    """
    @injector.inject
    @app.route('/search')
    def search(db: Database):
        query = request.args.get("q")
        page = int(request.args.get("page"))
        return db.movie_search(query, page)

    """
    A GET method that sends POST request to the ML modl endpoint (SAR notebook) and returns recommendations
    as a JSON object
    """
    @injector.inject
    @app.route('/recommendations')
    def recommend(db: Database, algorithm: Algorithm):

        movie_params = request.args.get("movies")
        rec_params = request.args.get("alg")

        try:
            id_list = [int(x) for x in movie_params.split('|')]
        except:
            raise LookupError('Incorrect Movie Ids')

        df = None

        if rec_params == "sar":
            df = algorithm(db).get_sar_recommendations(id_list)
        elif rec_params == "lgbm":
            df = algorithm(db).get_lgbm_recommendations(id_list)
        else:
            raise SelectionError('Algorithm does not exist')
        return jsonify({"Movies": list(df['json'])})


    """
    Gets lists of 'popular' movies sorted by genre
    """
    @app.route('/popular-movies')
    def popular():
        with open('data/starter_movies.json') as f:
            data = json.load(f)
        params = request.args.get("genres")

        response = dict.fromkeys(params.split('|'))

        for key in response:
            res = [movie for movie in data['Movies'] if key in movie['Genre']]
            if res:
                response[key] = res
            else:
                raise LookupError('Genre not found in popular movies')
        return json.dumps(response)

    """
    Gets lists of predetermined personas and their liked movies
    """
    @app.route('/personas')
    def persona():
        data = None
        with open('data/personas.json') as f:
            data = json.load(f)

        return json.dumps(data)

    """
    Returns list of onboarding movies
    """
    @app.route('/onboarding')
    def onboarding():
        genres = request.args.get("genres").split('|')
        num_per_category = 10 // len(genres)
        data = None 
        with open('data/movies_by_genre.json') as f:
            data = json.load(f)

        json_object = {"Movies": []}

        try:
            for genre in genres:
                for i in range(num_per_category):
                    json_object["Movies"].append(data[genre][i])
            return json.dumps(json_object)
        except:
            raise LookupError('Invalid Genre(s)')

    """
    Home page for health check
    """
    @app.route('/')
    def check():
        return "Backend is up and running!"

    _configure_dependency_injection(
        app, injector_modules, custom_injector)

    return app

if __name__ == '__main__':
    application = create_app()
    application.run(host='0.0.0.0')