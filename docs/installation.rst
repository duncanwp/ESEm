
===============
Installing ESEm
===============

Using PyPi
==========

It is straightforward to install esem using pip, this will automatically include tensorflow (with GPU support)::

    $ pip install esem

Optionally also install GPFlow, keras or scikit-learn ::

    $ pip install esem[gpflow]

Or ::

    $ pip install esem[gpflow,keras,scikit-learn]

Using conda
===========

In order to make the most of the support for `Iris <https://scitools-iris.readthedocs.io/en/stable/>`_ and `CIS <http://cistools.net/>`_ creating a specific conda environment is recommended.
If you don't already have conda, you must first download and install it. Anaconda is a free conda package that includes Python and many common scientific and data analysis libraries, and is available `here <http://continuum.io/downloads>`_. Further documentation on using Anaconda and the features it provides can be found at http://docs.continuum.io/anaconda/index.html.

Having installed (mini-) conda - and ideally within a fresh environment - you can easily install CIS (and Iris) with the following command::

    $ conda install -c conda-forge cis

It is then straightforward to install esem in to this environment using pip as above.


Dependencies
============

If you choose to install the dependencies yourself, use the following command to check the required dependencies are present::

    $ python setup.py checkdep

