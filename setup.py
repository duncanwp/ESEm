#!/usr/bin/env python

from setuptools import setup
from GCEm import __version__ as version

dependencies = ["scitools-iris",
                "tensorflow",
                "numpy",
                'matplotlib',
                'scipy',
                'sklearn']

optional_dependencies = {"Keras": ["keras"], "GPFlow": ["gpflow"]}


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='GCEm',
      version=version,
      description='General Climate Emulator',
      long_description=readme(),
      license='GPLv3',
      author='Duncan Watson-Parris',
      author_email='duncan.watson-parris@physics.ox.ac.uk',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Topic :: Scientific/Engineering :: Physics',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.7',
      ],
      keywords=['climate', 'machine-learning'],
      install_requires=dependencies,
      extras_require=optional_dependencies,
      tests_require=['unittest'],
      )