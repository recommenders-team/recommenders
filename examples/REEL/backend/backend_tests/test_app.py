import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import pytest
from mock import patch
from error import *
import injector
import providers
from app import create_app

class TestDB:
	pass

class TestAlg:
	pass


@pytest.fixture
def default_dependencies():
	return {
		'db': providers.TestDB(),
		'algorithm': providers.TestAlg(),
	}

@pytest.fixture
def test_injector():
	return injector.Injector()

@pytest.fixture
def app(default_dependencies, test_injector):
	app = create_app(
		custom_injector = test_injector,
		injector_modules = default_dependencies)
	app.testing = True

	with app.app_context():
		yield app

@pytest.fixture
def flask_test_client(app):
	with app.test_client() as test_client:
		yield test_client

# Helper function that checks genre membership of movie
def check_genre(movie, genre):
	return genre in movie['Genre']

# Initial trivial test to check compilations
def test_trivial(flask_test_client):
	assert 1 == 1

# Test landing health
def test_health(flask_test_client):
	subject = flask_test_client.get('/')
	assert subject.status_code == 200

def test_onboarding_healthy(flask_test_client):
	subject = flask_test_client.get('/onboarding?genres=Comedy|Action')
	res = json.loads(subject.data.decode("utf-8"))
	for movie in res['Movies']:
		assert check_genre(movie, "Comedy") or check_genre(movie, "Action")

def test_onboarding_bad_genre(flask_test_client):
	subject = flask_test_client.get('/onboarding?genres=Bad|Genre')
	assert subject.status_code == 400
	assert subject.data.decode("utf-8") == 'Invalid Genre(s)'
	#assert subject.headers.items() == 'Genre not found in popular movies'

def test_popular_healthy(flask_test_client):
	subject = flask_test_client.get('/popular-movies?genres=Thriller|Romance')
	res = json.loads(subject.data.decode("utf-8"))
	for movie in res['Thriller']:
		assert check_genre(movie, "Thriller")
	for movie in res['Romance']:
		assert check_genre(movie, "Romance")

def test_popular_bad_genre(flask_test_client):
	subject = flask_test_client.get('/popular-movies?genres=Bad|Genre')
	assert subject.status_code == 400
	assert subject.data.decode("utf-8") == 'Genre not found in popular movies'

'''
Can add tests to /recommendations route if backend API layer changes to perform
non-trivial transformation logic.
'''
