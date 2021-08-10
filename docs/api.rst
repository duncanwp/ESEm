.. currentmodule:: esem

#############
API reference
#############

This page provides an auto-generated summary of xarray's API. For more details
and examples, refer to the relevant chapters in the main part of the
documentation.


Top-level functions
===================

This provides the main interface for ESEm and should be the starting point for most users.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   gp_model
   cnn_model
   rf_model

Emulator
========

.. currentmodule:: esem.emulator
.. autosummary::
   :nosignatures:
   :toctree: generated/

   Emulator
   Emulator.train
   Emulator.predict
   Emulator._predict
   Emulator.batch_stats

Sampler
=======

This class defines the sampling interface currently used by the ABC and MCMC
sampling implementations.

.. currentmodule:: esem.sampler
.. autosummary::
   :nosignatures:
   :toctree: generated/

   Sampler
   Sampler.sample

MCMCSampler
-----------

.. autosummary::
   :nosignatures:
   :toctree: generated/

   MCMCSampler
   MCMCSampler.sample

ABCSampler
----------

.. currentmodule:: esem.abc_sampler
.. autosummary::
   :nosignatures:
   :toctree: generated/

   ABCSampler
   ABCSampler.sample
   ABCSampler.get_implausibility
   ABCSampler.batch_constrain

CubeWrapper
===========

.. currentmodule:: esem.cube_wrapper
.. autosummary::
   :nosignatures:
   :toctree: generated/

   DataWrapper
   CubeWrapper
   CubeWrapper.name
   CubeWrapper.data
   CubeWrapper.dtype
   CubeWrapper.wrap

ModelAdaptor
============

.. currentmodule:: esem.model_adaptor
.. autosummary::
   :nosignatures:
   :toctree: generated/

   ModelAdaptor
   SKLearnModel
   KerasModel
   GPFlowModel


DataProcessor
=============

.. currentmodule:: esem.data_processors
.. autosummary::
   :nosignatures:
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

A collection of associated utilities which might be of use when performing typical ESEm workflows.

.. currentmodule:: esem.utils
.. autosummary::
   :nosignatures:
   :toctree: generated/

   plot_results
   validation_plot
   plot_parameter_space
   get_uniform_params
   get_random_params
   ensemble_collocate
   leave_one_out
   get_param_mask
