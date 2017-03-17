# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(
    name='sparseflow',
    version='0.1',
    description='Benchmarking of several CTR prediction models.',
    author={'Max Barkhausen'},
    packages=['sparseflow'],
    package_dir={'sparseflow': 'sparseflow'}
)
