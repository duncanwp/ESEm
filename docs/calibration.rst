
=====================
Calibrating with ESEm
=====================

Having trained a fast, robust emulator this can be used to calibrate our model against available observations.
The following description closely follows that in our model description paper but with explicit links to the ESEm algorithms to aid clarity.
Generally, this problem involves estimating the model parameters which could give rise to, or best match, the available observations.

In other words, we would like to know the posterior probability distribution of the input parameters: :math:`p\left(\theta\middle| Y^0\right)`.

Using Bayes' theorem, we can write this as:

.. math::
    p\left(\theta\middle| Y^0\right)=p\left(Y^0\middle|\theta\right)p(θ)p(Y0)
    :label: Bayes

Where the probability of an output given the input parameters, :math:`p\left(Y^0\middle|\theta\right)`, is referred to as the likelihood.
While the model is capable of sampling this distribution, generally the full distribution is unknown and intractable, and we must approximate this likelihood.
Depending on the purpose of the calibration and assumptions about the form of :math:`p\left(Y^0\middle| Y\right)`, different techniques can be used.
In order to determine a (conservative) estimate of the parametric uncertainty in the model for example, we can use approximate Bayesian computation (ABC) to determine those parameters which are plausible given a set of observations.
Alternatively, we may wish to know the optimal parameters to best match a set of observations and Markov-Chain Monte-Carlo based techniques might be more appropriate.
Both of these sampling strategies are available in ESEm and we describe each of them here.


Using approximate Bayesian computation (ABC)
============================================

The simplest ABC approach seeks to approximate the likelihood using only samples from the simulator and a discrepancy function :math:`\rho`:

.. math::
    p\left(\theta\middle| Y^0\right)\propto p\left(Y^0\middle| Y\right)p\left(Y\middle|\theta\right)p(\theta)\approx\int{\mathbb{I}(\rho\left(Y^0,Y\right)\le\epsilon)\ \ p\left(Y\middle|\theta\right)\ p(\theta)\ dY}
    :label: Eq 2

where the indicator function :math:`\mathbb{I}(x) = 1` if x is true and :math:`\mathbb{I}(x) = 0` if x is false, and :math:`\epsilon` is a small discrepancy.
This can then be integrated numerically using e.g., Monte-Carlo sampling of :math:`p(\theta)`.
Any of those parameters for which :math:`\rho\left(Y^0,Y\right)\le\epsilon` are accepted and those which do not are rejected.
As :math:`\epsilon\rightarrow\infty` therefore, all parameters are accepted and we recover :math:`p(\theta)`.
For :math:`\epsilon=0`, it can be shown that we generate samples from the posterior :math:`p\left(\theta\middle| Y^0\right)` exactly.

In practice however the simulator proposals will never exactly match the observations and we must make a pragmatic choice for both :math:`\rho` and :math:`\epsilon`.
ESEm includes an implementation of the 'implausibility metric' which defines the discrepancy in terms of the standardized Cartesian distance:

.. math::
    \rho(Y^0,Y(\theta))=\frac{\left|Y^0-Y(\theta)\right|}{\sqrt{\sigma_E^2+\sigma_Y^2+\sigma_R^2+\sigma_S^2}}=\rho(Y^0,\theta)
    :label: Eq. 3

where the total standard deviation is taken to be the squared sum of the emulator variance :math:`(\sigma_E^2)` and the uncertainty in the observations :math:`(\sigma_Y^2)` and due to representation :math:`(\sigma_R^2)` and structural model uncertainties :math:`(\sigma_S^2)` as described in the paper.

Framed in this way, :math:`\epsilon`, can be thought of as representing the number of standard deviations the (emulated) model value is from the observations.
While this can be treated as a free parameter and may be specified in ESEm, it is common to choose :math:`\epsilon=3` since it can be shown that for unimodal distributions values of :math:`3\sigma` correspond to a greater than 95% confidence bound.
This approach is implemented in the :class:`esem.abc_sampler.ABCSampler` class where :math:`\epsilon` is referred to as a threshold since it defines the cut-off for acceptance.

In the general case, multiple :math:`(\mathcal{N})` observations can be used and :math:`\rho` can be written as a vector of implausibilities, :math:`\rho(Y_i^O,\theta)` or simply :math:`\rho_i(\theta)`, and a modified method of rejection or acceptance must be used.
A simple choice is to require :math:`\rho_i<\epsilon\ \forall\ i\ \in\mathcal{N}`, however this can become restrictive for large :math:`\mathcal{N}` due to the curse of dimensionality.
The first step should be to reduce :math:`\mathcal{N}` through the use of summary statistics, such as averaging over regions, or stations, or by performing an e.g. Principle Component Analysis (PCA) decomposition.

An alternative is to introduce a tolerance (T) such that only some proportion of :math:`\rho_i` need to be smaller than :math:`\epsilon: \sum_{i=0}^{\mathcal{N}}{H(}\rho_i\ -\ \epsilon)<T`, where H is the Heaviside function.
This tolerance can be specified when sampling using the :class:`esem.abc_sampler.ABCSampler`.
An efficient implementation of this approach whereby the acceptance is calculated in batches on the GPU can be particularly useful when dealing with high-dimensional outputs :class:`esem.abc_sampler.ABCSampler`.
It is recommended however, to choose :math:`T=0` as a first approximation and then identify any particular observations which generate a very large implausibilities, since this provides a mechanism for identifying potential structural (or observational) errors.

A useful way of identifying such observations is using the :meth:`esem.abc_sampler.ABCSampler.get_implausibility` method which returns the full implausibility matrix :math:`\rho_i`. Note this may be very large (N-samples x N-observations) so it is recommended that only a subset of the full sample space be requested.
The offending observations can then be removed and noted for further investigation.

Examples of the ABC sampling can be found in `Calibrating_GPs_using_ABC.ipynb <examples/Calibrating_GPs_using_ABC.html>`_.


Using Markov-chain Monte-Carlo
==============================

The ABC method described above is simple and powerful, but somewhat inefficient as it repeatedly samples from the same prior.
In reality each rejection or acceptance of a set of parameters provides us with extra information about the ‘true’ form of :math:`p\left(\theta\middle| Y^0\right)` so that the sampler could spend more time in plausible regions of the parameter space.
This can then allow us to use smaller values of :math:`\epsilon` and hence find better approximations of  :math:`p\left(\theta\middle| Y^0\right)`.

Given the joint probability distribution described by Eq. 2 and an initial choice of parameters :math:`\theta'` and (emulated) output :math:`Y'`, the acceptance probability :math:`r` of a new set of parameters :math:`(\theta)` is given by:

.. math::
    r=\frac{p\left(Y^0\middle| Y'\right)p\left(\theta'\middle|\theta\right)p(\theta')}{p\left(Y^0\middle| Y\right)p\left(\theta\middle|\theta'\right)p(\theta)}
    :label: Eq. 4

The :class:`esem.sampler.MCMCSampler` class uses the TensorFlow-probability implementation of Hamiltonian Monte-Carlo (HMC) which uses the gradient information automatically calculated by TensorFlow to inform the proposed new parameters :math:`\theta`.
For simplicity, we assume that the proposal distribution is symmetric: :math:`p\left(\theta'\middle|\theta\right)\ =\ p\left(\theta\middle|\theta'\right)`, which is implemented as a zero log-acceptance correction in the initialisation of the TensorFlow target distribution.
The target log probability provided to the TensorFlow HMC algorithm is then:

.. math::
    log(r)=log(p\left(Y^0\middle| Y'\right))\ +\ log(p(\theta'))\ -\ log(p\left(Y^0\middle| Y\right))\ -\ log(p(\theta))
    :labeL: Eq. 5

Note, that for this implementation the distance metric :math:`\rho` must be cast as a probability distribution with values [0, 1].
We therefore assume that this discrepancy can be approximated as a normal distribution centred about zero, with standard deviation equal to the sum of the squares of the variances as described in Eq. 3:

.. math::
    p\left(Y^0\middle| Y\right)\approx{\frac{1}{\sigma_t\sqrt{2\pi}}e}^{-\frac{1}{2}\left(\frac{Y^0-Y}{\sigma_t}\right)^2},\ \ \sigma_t=\sqrt{\sigma_E^2+\sigma_Y^2+\sigma_R^2+\sigma_S^2}
    :label: Eq. 6

The :meth:`esem.sampler.MCMCSampler.sample` method will then return the requested number of accepted samples as well as reporting the acceptance rate, which provides a useful metric for tuning the algorithm.
It should be noted that MCMC algorithms can be sensitive to a number of key parameters, including the number of burn-in steps used (and discarded) before sampling occurs and the step size. Each of these can be controlled via keyword arguments to the :meth:`esem.sampler.MCMCSampler.sample` method.

This approach can provide much more efficient sampling of the emulator and provide improved parameter estimates, especially when used with informative priors which can guide the sampler.

Examples of the MCMC sampling can be found in `Calibrating_GPs_using_MCMC.ipynb <examples/Calibrating_GPs_using_MCMC.html>`_ and `CMIP6_emulator.ipynb <examples/CMIP6_emulator.html>`_.
