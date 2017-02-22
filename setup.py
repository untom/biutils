from setuptools import setup
from codecs import open
from os import path
import numpy as np
VERSION = '2017.02'


# Get the long description from the relevant file
with open(path.join('README.rst'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()


setup(
    name='biutils',

    version=VERSION,

    description='Collection of utilities for various Machine Learning related things',
    long_description=LONG_DESCRIPTION,
    url='https://github.com/untom/biutils',

    # Author details
    author='Thomas Unterthiner',
    author_email='thomas.unterthiner@gmx.net',

    # Choose your license
    license='GPLv2+',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=['Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
        'License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],

    keywords='Machine Learning Deep Learning Neural Nets',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=['biutils'],
)
