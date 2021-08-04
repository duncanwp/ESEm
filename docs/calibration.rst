
=====================
Calibrating with ESEm
=====================

Having trained a fast, robust emulator this can be used to calibrate our model against available observations.
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
where the indicator function :math:`\mathbb{I}(x)=1,&x is true0,&x is false`, and :math:`\epsilon` is a small discrepancy.
This can then be integrated numerically using e.g., Monte-Carlo sampling of p(\theta).
Any of those parameters for which \rho\left(Y^0,Y\right)\le\epsilon are accepted and those which do not are rejected.
As \epsilon\rightarrow\infty therefore, all parameters are accepted and we recover p(\theta).
For \epsilon=0, it can be shown that we generate samples from the posterior p\left(\theta\middle| Y^0\right) exactly.

In practice however the simulator proposals will never exactly match the observations and we must make a pragmatic choice for both \rho and \epsilon. ESEm includes an implementation of the ‘implausibility metric’ (Williamson et al., 2013; Craig et al., 1996; Vernon et al., 2010)(Williamson et al., 2013; Craig et al., 1996; Vernon et al., 2010) which defines the discrepancy in terms of the standardized Cartesian distance:
\rho(Y^0,Y(\theta))=\frac{\left|Y^0-Y(\theta)\right|}{\sqrt{\sigma_E^2+\sigma_Y^2+\sigma_R^2+\sigma_S^2}}=\rho(Y^0,\theta)	Eq. 3
where the total standard deviation is taken to be the squared sum of the emulator variance (\sigma_E^2)\ and the uncertainty in the observations (\sigma_Y^2) and due to representation (\sigma_R^2) and structural model uncertainties (\sigma_S^2). As described above, the representation uncertainty represents the degree to which observations at a particular time and location can be expected to match the (typically aggregate) model output  (Schutgens et al., 2016a, b). While reasonable approximates can often be made of this and the observational uncertainties, the model structural uncertainties are typically unknown. In some cases, a multi-model ensemble may be available which can provide an indication of the structural uncertainties for particular observables (Sexton et al., 1995), but these are likely to underestimate true structural uncertainties as models typically share many key processes and assumptions (Knutti et al., 2013). Indeed, one benefit of a comprehensive analysis of the parametric uncertainty of a model is that this structural uncertainty can be explored and determined (Williamson et al., 2015).
Framed in this way, \epsilon, can be thought of as representing the number of standard deviations the (emulated) model value is from the observations. While this can be treated as a free parameter and may be specified in ESEm, it is common to choose \epsilon=3\ since it can be shown that for unimodal distributions values of 3\sigma correspond to a greater than 95% confidence bound (Vysochanskij and Petunin, 1980).
This approach is closely related to the approach of ‘history matching’ (Williamson et al., 2013) and can be shown to be identical in the case of fixed \epsilon and uniform priors (Holden et al., 2015b). The key difference being that history matching may result in an empty posterior distribution, that is, it may find no plausible model configurations which match the observations. With ABC on the other hand the epsilon is typically treated as a hyper-parameter which can be tuned in order to return a suitably large number of posterior samples. Both \epsilon and the prior distributions can be specified in ESEm and it can thus be used to perform either analysis. The speed at which samples can typically be generated from the emulator means we can keep \epsilon fixed as in history matching and generate as many samples as is required to estimate the posterior distribution.
When multiple (\mathcal{N})\ observations are used (as is often the case) \rho can be written as a vector of implausibilities, \rho(Y_i^O,\theta) or simply \rho_i(\theta), and a modified method of rejection or acceptance must be used. A simple choice is to require \rho_i<\epsilon\ \forall\ i\ \in\mathcal{N}, however this can become restrictive for large \mathcal{N} due to the curse of dimensionality. The first step should be to reduce \mathcal{N} through the use of summary statistics as described above. An alternative is to introduce a tolerance (T) such that only some proportion of \rho_i need to be smaller than \epsilon: \sum_{i=0}^{\mathcal{N}}{H(}\rho_i\ -\ \epsilon)<T, where H is the Heaviside function (Johnson et al. 2019), although this is a somewhat unsatisfactory approach that can hide potential structural uncertainties. On the other hand, choosing T=0 as a first approximation and then identifying any particular observations which generate a very large implausibility provides a mechanism for identifying potential structural (or observational) errors. These can then be removed and noted for further investigation.


Using Markov-chain Monte-Carlo
==============================

The ABC method described above is simple and powerful, but somewhat inefficient as it repeatedly samples from the same prior. In reality each rejection or acceptance of a set of parameters provides us with extra information about the ‘true’ form of p\left(\theta\middle| Y^0\right) so that the sampler could spend more time in plausible regions of the parameter space. This can then allow us to use smaller values of \epsilon and hence find better approximations of  p\left(\theta\middle| Y^0\right).
Given the joint probability distribution described by Eq. 2 and an initial choice of parameters \theta\prime and (emulated) output Y\prime, the acceptance probability r of a new set of parameters (\theta) is given by:
r=\frac{p\left(Y^0\middle| Y\prime\right)p\left(\theta\prime\middle|\theta\right)p(\theta\prime)}{p\left(Y^0\middle| Y\right)p\left(\theta\middle|\theta\prime\right)p(\theta)}	Eq. 4
In the default implementation of MCMC calibration ESEm uses the TensorFlow-probability implementation of Hamiltonian Monte-Carlo (HMC) (Neal, 2011) which uses the gradient information automatically calculated by TensorFlow to inform the proposed new parameters \theta. For simplicity, we assume that the proposal distribution is symmetric: p\left(\theta\prime\middle|\theta\right)\ =\ p\left(\theta\middle|\theta\prime\right), which is implemented as a zero log-acceptance correction in the initialisation of the TensorFlow target distribution. The target log probability provided to the TensorFlow HMC algorithm is then:
log(r)=log(p\left(Y^0\middle| Y\prime\right))\ +\ log(p(\theta\prime))\ -\ log(p\left(Y^0\middle| Y\right))\ -\ log(p(\theta))\ 	Eq. 5
Note, that for this implementation the distance metric \rho must be cast as a probability distribution with values [0, 1]. We therefore assume that this discrepancy can be approximated as a normal distribution centred about zero, with standard deviation equal to the sum of the squares of the variances as described in Eq. 3:
p\left(Y^0\middle| Y\right)\approx{\frac{1}{\sigma_t\sqrt{2\pi}}e}^{-\frac{1}{2}\left(\frac{Y^0-Y}{\sigma_t}\right)^2},\ \ \sigma_t=\sqrt{\sigma_E^2+\sigma_Y^2+\sigma_R^2+\sigma_S^2}\ 	Eq. 5
The implementation will then return the requested number of accepted samples as well as reporting the acceptance rate, which provides a useful metric for tuning the algorithm. It should be noted that MCMC algorithms can be sensitive to a number of key parameters, including the number of burn-in steps used (and discarded) before sampling occurs and the step size. Each of these can be controlled via keyword arguments to the sampler.
This approach can provide much more efficient sampling of the emulator and provide improved parameter estimates, especially when used with informative priors which can guide the sampler.

