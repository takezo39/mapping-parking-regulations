#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
from codecs import open  # To use a consistent encoding
from os import path

#import dataexplorer

"""
Detecting parking regulation signs from video taken from a driving car. 
Associating  GPS coordinates with signs. Deriving parking pockets and their 
associated parking rules.
"""

# Get the long description from the README file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

packages = ['sign_detection', 'sign_detection.data',
            'sign_detection.features',
            'sign_detection.models']

#tests = [p + '.tests' for p in packages]

setup(
    name='sign_detection',
    #version=dataexplorer.__version__,
    description="Detecting parking regulation signs from video taken from a driving car. Associating  GPS coordinates with signs. Deriving parking pockets and their associated parking rules.",
    author="Mark Hashimoto",
    author_email="mark.m.hashimoto@gmail.com",
    packages=packages,
    #packages=packages + tests,
    install_requires=[
        'python-dotenv>=0.5.1',
    ],
    long_description=long_description,
    test_suite='nose.collector',
    tests_require=['nose'],
    include_package_data=True,
    #package_data={'path_to_data': [list_of_files]},
    zip_safe=False
)
