
===============
Installing ESEm
===============

Currently installing ESEm is a two step process. This is because `Iris <https://scitools-iris.readthedocs.io/en/stable/>`_ cannot be installed via pip and installing tensorflow via conda doesn't provide machine level optimisations.
This is a long-standing issue that is unlikely to be quickly resolved but the steps below should provide a stable environment.

Having installed (mini-) conda, you can easily install Iris with the following command::

    $ conda install -c conda-forge iris

If you don't already have conda, you must first download and install it. Anaconda is a free conda package that includes Python and many common scientific and data analysis libraries, and is available `here <http://continuum.io/downloads>`_. Further documentation on using Anaconda and the features it provides can be found at http://docs.continuum.io/anaconda/index.html.

Now you can pip install esem, this will automatically include tensorflow (with GPU support)::

    $ pip install esem

Optionally also install GPFlow, keras or scikit-learn ::

    $ pip install esem[gpflow]

Or ::

    $ pip install esem[gpflow,keras,scikit-learn]

Dependencies
============

If you choose to install the dependencies yourself, use the following command to check the required dependencies are present::

    $ python setup.py checkdep

