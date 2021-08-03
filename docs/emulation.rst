
===================
Emulating with ESEm
===================

ESEm provides a simple and streamlined interface to emulate earth system datasets stored as iris Cubes.
The corresponding predictors can be provided as a numpy array or pandas DataFrame.
This emulation is essentially just a regression and can be performed using a variety of techniques using the same API.


Gaussian processes emulation
============================

The ESEm Gaussian process emulation module provides a wrapper around the `GPFlow <https://gpflow.readthedocs.io/en/master/#>`_ implementation.


Feature selection
=================

ESEm includes a simple utility function that wraps the scikit-learn LassoLarsIC regression tool in order to enable an
initial feature (parameter) selection. This can be useful to reduce the dimensionality of the input space. Either the
Akaike information criterion (AIC) or the Bayes Information criterion (BIC) can be used, although BIC is the default.

For example,

.. code-block:: python

    from esem import gp_model
    from esem.utils import get_param_mask

    # X and Y are our model parameters and outputs respectively.
    active_params = get_param_mask(X, y)

    # The model parameters can then be subsampled either directly
    X_sub = X[:, active_params]

    # Or by specifying the GP active_dims
    active_dims, = np.where(active_params)
    model = gp_model(X, y, active_dims=active_dims)


Note, this estimate only applies to one-dimensional outputs. Feature selection for higher dimension outputs is a much
harder task beyond the scope of this package.
