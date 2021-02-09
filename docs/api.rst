.. currentmodule:: GCEm

#############
API reference
#############

This page provides an auto-generated summary of xarray's API. For more details
and examples, refer to the relevant chapters in the main part of the
documentation.


Top-level functions
===================

.. autosummary::
   :toctree: generated/

   gp_model
   cnn_model
   rf_model

Emulator
========

.. autosummary::
   :toctree: generated/

   Emulator
   Emulator.train
   Emulator.predict
   Emulator.batch_stats

Sampler
=======

.. currentmodule:: GCEm.sampler
.. autosummary::
   :toctree: generated/

   Sampler
   Sampler.sample

ABCSampler
----------

.. currentmodule:: GCEm.abc_sampler
.. autosummary::
   :toctree: generated/

   ABCSampler.get_implausibility
   ABCSampler.batch_constrain

CubeWrapper
===========

.. currentmodule:: GCEm.cube_wrapper
.. autosummary::
   :toctree: generated/

   CubeWrapper
   CubeWrapper.name
   CubeWrapper.data
   CubeWrapper.dtype
   CubeWrapper.wrap

ModelAdaptor
============

.. currentmodule:: GCEm.model_adaptor
.. autosummary::
   :toctree: generated/

   ModelAdaptor


DataProcessor
=============

.. currentmodule:: GCEm.data_processors
.. autosummary::
   :toctree: generated/

   DataProcessor
   Log
   Whiten
   Normalise
   Flatten
   Reshape
   Recast

Utilities
=========

.. currentmodule:: GCEm.utils
.. autosummary::
   :toctree: generated/

   plot_results
   validation_plot
   plot_parameter_space
   get_uniform_params
   get_random_params
   ensemble_collocate
   LeaveOneOut