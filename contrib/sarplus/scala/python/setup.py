from distutils.core import setup
import os

version = os.getenv("VERSION")
if version is None:
    version = "0.5.0"

setup(
    name="pysarplus_dummy",
    version=os.environ["VERSION"],
    description="pysarplus dummy package to trigger spark packaging",
    author="Markus Cozowicz",
    author_email="marcozo@microsoft.com",
    url="https://github.com/Microsoft/Recommenders/contrib/sarplus",
    packages=["pysarplus_dummy"],
)
