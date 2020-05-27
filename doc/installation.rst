
===============
Installing GCEm
===============

Once conda is installed, you can easily install CIS with the following command::

    $ conda install -c conda-forge cis


If you don't already have conda, you must first download and install it. Anaconda is a free conda package that includes Python and many common scientific and data analysis libraries, and is available `here <http://continuum.io/downloads>`_. Further documentation on using Anaconda and the features it provides can be found at http://docs.continuum.io/anaconda/index.html.

Then pip install GPflow and tensorflow-gpu::

    $ pip install gpflow tensorflow-gpu

Optionally also install keras::

    $ conda install -c conda-forge keras

Dependencies
============

If you choose to install the dependencies yourself, use the following command to check the required dependencies are present::

    $ python setup.py checkdep

