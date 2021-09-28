Earth System Emulator (ESEm)
============================

[![CircleCI](https://circleci.com/gh/duncanwp/ESEm.svg?style=svg)](https://circleci.com/gh/duncanwp/ESEm)
[![codecov](https://codecov.io/gh/duncanwp/ESEm/branch/master/graph/badge.svg?token=4QI2G22Q3M)](https://codecov.io/gh/duncanwp/ESEm)
[![readthedocs](https://readthedocs.org/projects/pip/badge/?version=latest&style=plastic)](https://esem.readthedocs.io/en/latest/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5196631.svg)](https://doi.org/10.5281/zenodo.5196631)
[![PyPI version](https://badge.fury.io/py/ESEm.svg)](https://pypi.org/project/ESEm/)

A tool for emulating geophysical datasets including (but not limited to!) Earth System Models.

Why ESEm?
---------

While excellent tools exist for regression and emulation, and similarly for efficient calibration, there isn't a single package that makes it easy for Earth scientists to emulate and calibrate their models. ESEm provides a simple interface to do so, with a thin wrapper around familiar emulation engines and efficient sampling tools. 

ESEm can use [Iris](https://scitools-iris.readthedocs.io/en/stable/) Cubes or [xarray](http://xarray.pydata.org/en/stable/) DataArrays to retain useful geophysical information about the data being emulated and also streamlines the typical task of co-locating models and observations for comparison using e.g. [CIS](https://cis.readthedocs.io/).

These tasks aren't just restricted to emulating and calibrating models though and can be used in any situation where regression of Earth system data is needed.

Documentation
-------------

Detailed instructions and example notebooks can be found in our official documentation at https://esem.readthedocs.io/en/latest/

Installation
------------

ESEm can be easily installed using pip, including tensorflow (with GPU support):

    $ pip install esem

Optionally also install GPFlow, keras or scikit-learn e.g.,:

    $ pip install esem[gpflow]

For more detailed instructions, including using conda to install alongside iris or xarray see https://esem.readthedocs.io/en/latest/installation.html

Citation
--------

If you use ESEm in your research please be sure to cite our [paper](https://gmd.copernicus.org/preprints/gmd-2021-267/):

    @Article{gmd-2021-267,
    AUTHOR = {Watson-Parris, D. and Williams, A. and Deaconu, L. and Stier, P.},
    TITLE = {Model calibration using ESEm v1.0.0 -- an open, scalable Earth System Emulator},
    JOURNAL = {Geoscientific Model Development Discussions},
    VOLUME = {2021},
    YEAR = {2021},
    PAGES = {1--24},
    URL = {https://gmd.copernicus.org/preprints/gmd-2021-267/},
    DOI = {10.5194/gmd-2021-267}
    }

Contributing
------------

Contributions to ESEm of any size are very welcome, please see our [Contributing](https://github.com/duncanwp/ESEm/blob/master/CONTRIBUTING.md) page for more details.


Get in touch
------------
 - Ask general installation and usage questions ("How do I?") in our [Discussions](https://github.com/duncanwp/ESEm/discussions) tab.
 - Report bugs and suggest features in the [Issues](https://github.com/duncanwp/ESEm/issues) tab


License
-------

   Copyright 2019-2021 Duncan Watson-Parris

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
