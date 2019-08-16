
"""
Base class for exceptions
"""
class Error(Exception):
	pass


"""
Error handler for database lookup failures
"""
class LookupError(Error):
	def __init__(self, message):
		Exception.__init__(self)
		self.message = message
		self.status_code = 400

"""
Error handler for invalid credentials
"""
class SelectionError(Error):
	def __init__(self, message):
		Exception.__init__(self)
		self.message = message
		self.status_code = 400

"""
Error handler for invalid jsons
"""
class JsonError(Error):
	def __init__(self, message):
		Exception.__init__(self)
		self.message = message
		self.status_code = 400

"""
Error handler for invalid movie recommendations
"""
class RecommendationError(Error):
	def __init__(self, message):
		Exception.__init__(self)
		self.message = message
		# Can change from 300, but currently represents additional information needed
		# from the client.
		self.status_code = 300
