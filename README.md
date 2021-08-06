Earth System Emulator (ESEm)
============================

[![CircleCI](https://circleci.com/gh/duncanwp/ESEm.svg?style=svg)](https://circleci.com/gh/duncanwp/ESEm)
[![codecov](https://codecov.io/gh/duncanwp/ESEm/branch/master/graph/badge.svg?token=4QI2G22Q3M)](https://codecov.io/gh/duncanwp/ESEm)
[![readthedocs](https://readthedocs.org/projects/pip/badge/?version=latest&style=plastic)](https://esem.readthedocs.io/en/latest/)

A tool for emulating geophysical datasets including (but not limited to!) Earth System Models.

Why ESEm?
---------

While excellent tools exist for regression and emulation, and similarly for efficient calibration, there isn't a single package that makes it easy for Earth scientists to emulate and calibrate their models. ESEm provides a simple interface to do so, with a thin wrapper around familiar emulation engines and efficient sampling tools. 

ESEm uses [Iris](https://scitools-iris.readthedocs.io/en/stable/) Cubes to retain useful geophysical information about the data being emulated and also streamlines the typical task of co-locating models and observations for comparison using [CIS](https://cis.readthedocs.io/).

These tasks aren't just restricted to emulating and calibrating models though and can be used in any situation where regression of Earth system data is needed.

Documentation
-------------

Detailed instructions and example notebooks can be found in our official doccumentation at https://esem.readthedocs.io/en/latest/

Citation
--------

If you use ESEm in your research please be sure to cite our paper: [Coming soon!]

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
