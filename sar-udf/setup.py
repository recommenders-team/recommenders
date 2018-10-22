import sysconfig

from setuptools import setup
from setuptools.extension import Extension


class get_pybind_include(object):
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11

        return pybind11.get_include(self.user)


def readme():
    with open("README.rst") as f:
        return f.read()


setup(
    name="pysar",
    version="0.1",
    description="SAR prediction for use with PySpark",
    long_description=readme(),
    url="https://github.com/Microsoft/Recommenders",
    author="Markus Cozowicz",
    author_email="marcozo@microsoft.com",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    setup_requires=["pytest-runner"],
    install_requires=["pybind11>=2.2"],
    tests_require=["pytest"],
    packages=["pysar"],
    ext_modules=[
        Extension(
            "pysar_cpp",
            ["src/pysar.cpp"],
            include_dirs=[get_pybind_include(), get_pybind_include(user=True)],
            extra_compile_args=sysconfig.get_config_var("CFLAGS").split()
            + ["-std=c++11", "-Wall", "-Wextra"],
            language="c++11",
        )
    ],
    zip_safe=False,
)
