import flask
import flask_injector
import injector
import database
import algorithm

class DatabaseModule(injector.Module):
	def configure(self, binder):
		binder.bind(database.Database,
					to=self.create,
					scope=flask_injector.request)

	@injector.inject
	def create(self) -> database.Database:
		return database.Database()

class AlgorithmModule(injector.Module):
	def configure(self, binder):
		binder.bind(algorithm.Algorithm,
					to=self.create,
					scope=flask_injector.request)

	@injector.inject
	def create(self) -> algorithm.Algorithm:
		return algorithm.Algorithm

# All classes below are for injections during testing

class AppTestDB:
	pass

class AppTestAlg:
	def __init__(self, database):
		self.db = database

class TestDB(injector.Module):
	def configure(self, binder):
		binder.bind(AppTestDB,
					to=self.create,
					scope=flask_injector.request)

	@injector.inject
	def create(self) -> AppTestDB:
		return AppTestDB()

class TestAlg(injector.Module):
	def configure(self, binder):
		binder.bind(AppTestAlg,
					to=self.create,
					scope=flask_injector.request)

	@injector.inject
	def create(self) -> AppTestAlg:
		return AppTestAlg
