
===============
Installing esem
===============

Once conda is installed, you can easily install CIS with the following command::

    $ conda install -c conda-forge cis

If you don't already have conda, you must first download and install it. Anaconda is a free conda package that includes Python and many common scientific and data analysis libraries, and is available `here <http://continuum.io/downloads>`_. Further documentation on using Anaconda and the features it provides can be found at http://docs.continuum.io/anaconda/index.html.

In our experience the pip install of tensorflow has better hardware support than on conda. Installing esem this way will automatically include tensorflow (with GPU support)::

    $ pip install esem

Optionally also install GPFlow, keras or scikit-learn ::

    $ pip install esem[gpflow]

Dependencies
============

If you choose to install the dependencies yourself, use the following command to check the required dependencies are present::

    $ python setup.py checkdep

