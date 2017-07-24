import os
from setuptools import setup, find_packages


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# Read metadata from version file
classifiers = [
    "Programming Language :: Python :: 3",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Topic :: Software Development :: Libraries",
    "Topic :: Utilities",
]

setup(
    name='dataset_loading',
    version='0.0.1',
    author="Fergal Cotter",
    author_email="fbc23@cam.ac.uk",
    description=("Convenience Functions for Tensorflow"),
    license="MIT",
    keywords="image datasets, cifar, pascal, mnist",
    url="https://github.com/fbcotter/dataset_loading.git",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    long_description=read('README.rst'),
    classifiers=classifiers,
    install_requires=["numpy", "Pillow"],
    tests_require=["pytest"],
    extras_require={
        'docs': ['sphinx', 'docutils']
    }
)
