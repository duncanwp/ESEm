from abc import ABC
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class Sampler(ABC):
    """
    A class that efficiently samples a Model object for posterior inference
    """

    def __init__(self, model, obs,
                 obs_uncertainty=0., interann_uncertainty=0.,
                 repres_uncertainty=0., struct_uncertainty=0.,
                 abs_obs_uncertainty=0., abs_interann_uncertainty=0.,
                 abs_repres_uncertainty=0., abs_struct_uncertainty=0.,
                 ):
        """

        Parameters
        ----------
        model: esem.emulator.Emulator
        obs: iris.cube.Cube or array-like
            The objective
        obs_uncertainty: float
            Fractional, relative (1 sigma) uncertainty in observations
        repres_uncertainty: float
            Fractional, relative (1 sigma) uncertainty due to the spatial and temporal
            representitiveness of the observations
        interann_uncertainty: float
            Fractional, relative (1 sigma) uncertainty introduced when using a model run
            for a year other than that the observations were measured in.
        struct_uncertainty: float
            Fractional, relative (1 sigma) uncertainty in the model itself.
        abs_obs_uncertainty: float
            Fractional, absolute (1 sigma)  uncertainty in observations
        abs_repres_uncertainty: float
            Fractional, absolute (1 sigma)  uncertainty due to the spatial and temporal
            representitiveness of the observations
        abs_interann_uncertainty: float
            Fractional, absolute (1 sigma)  uncertainty introduced when using a model run
            for a year other than that the observations were measured in.
        abs_struct_uncertainty: float
            Fractional, absolute (1 sigma)  uncertainty in the model itself.

        """
        self.model = model

        # This tests for 'cube like' objects including CIS ungridded data
        #  I don't want to have to depend on CIS though to check explicitly
        if hasattr(obs, 'data') and isinstance(obs.data, np.ndarray):
            obs = obs.data

        self.obs = obs.astype(model.dtype)

        def _is_specified(uncertainty):
            # If it's anything other than a float 0. (e.g. and array) it must have been specified
            return (type(uncertainty) != float) or (uncertainty != 0.)

        if _is_specified(obs_uncertainty) and _is_specified(abs_obs_uncertainty):
            raise ValueError("Only one of the absolute and relative observational uncertainties should be specified")
        elif _is_specified(abs_obs_uncertainty):
            # Broadcast the square of the absolute uncertainty
            abs_observational_var = np.broadcast_to(np.square(abs_obs_uncertainty), self.obs.shape)
        else:  # obs_uncertainty can be zero
            # Get the square of the absolute uncertainty and add a batch dimension
            abs_observational_var = np.square(self.obs * obs_uncertainty)[np.newaxis, ...]

        if _is_specified(repres_uncertainty) and _is_specified(abs_repres_uncertainty):
            raise ValueError("Only one of the absolute and relative representivity uncertainties should be specified")
        elif _is_specified(abs_repres_uncertainty):
            abs_respres_var = np.broadcast_to(np.square(abs_repres_uncertainty), self.obs.shape)
        else:  # obs_uncertainty can be zero
            abs_respres_var = np.square(self.obs * repres_uncertainty)[np.newaxis, ...]

        if _is_specified(interann_uncertainty) and _is_specified(abs_interann_uncertainty):
            raise ValueError("Only one of the absolute and relative interannual uncertainties should be specified")
        elif _is_specified(abs_interann_uncertainty):
            abs_interann_var = np.broadcast_to(np.square(abs_interann_uncertainty), self.obs.shape)
        else:  # obs_uncertainty can be zero
            abs_interann_var = np.square(self.obs * interann_uncertainty)[np.newaxis, ...]

        if _is_specified(struct_uncertainty) and _is_specified(abs_struct_uncertainty):
            raise ValueError("Only one of the absolute and relative structural uncertainties should be specified")
        elif _is_specified(abs_struct_uncertainty):
            abs_struct_var = np.broadcast_to(np.square(abs_struct_uncertainty), self.obs.shape)
        else:  # obs_uncertainty can be zero
            abs_struct_var = np.square(self.obs * struct_uncertainty)[np.newaxis, ...]

        self.total_var = sum([abs_observational_var, abs_respres_var, abs_interann_var, abs_struct_var])

    def sample(self, prior_x=None, n_samples=1):
        """
        This is the call that does the actual inference.

        It should call model.sample over the prior, compare with the objective, and then output samples
        from the posterior distribution

        Parameters
        ----------
        prior_x: tensorflow_probability.distribution
            The distribution to sample parameters from.
            By default it will uniformly sample the unit N-D hypercube
        n_samples: int
            The number of samples to draw

        Returns
        -------
        np.array
            Array of samples
        """
        pass


class MCMCSampler(Sampler):
    """
    Sample from the posterior using the TensorFlow Markov-Chain Monte-Carlo (MCMC)
    sampling tools. It uses a HamiltonianMonteCarlo kernel.

    Notes
    -----
    Note that NaN observations will create ill-defined likelihoods.
    """

    def __init__(self, model, obs, **kwargs):
        super().__init__(model, obs, **kwargs)
        if np.isnan(self.obs).any():
            print("WARNING: Obs contains some NaNs which will invalidate the sampling results!")

    def sample(self, prior_x=None, n_samples=1, kernel_kwargs=None, mcmc_kwargs=None):
        """
        This is the call that does the actual inference.

        It should call model.sample over the prior, compare with the objective, and then output a posterior
        distribution

        Parameters
        ----------
        prior_x: tensorflow_probability.distribution
            The distribution to sample parameters from.
            By default it will uniformly sample the unit N-D hypercube
        n_samples: int
            The number of samples to draw
        kernel_kwargs: dict
            kwargs for the MCMC kernel
        mcmc_kwargs: dict
            kwargs for the MCMC sampler

        Returns
        -------
        np.array
            Array of samples
        """
        if prior_x is None:
            prior_x = tfd.Uniform(low=tf.zeros(self.model.n_params, dtype=tf.float64),
                                  high=tf.ones(self.model.n_params, dtype=tf.float64))

        prior_x = tfd.Independent(prior_x, reinterpreted_batch_ndims=1, name='model')

        if kernel_kwargs is None:
            kernel_kwargs = dict()
        kernel_kwargs.setdefault('step_size', 0.01)
        kernel_kwargs.setdefault('num_leapfrog_steps', 5)

        if mcmc_kwargs is None:
            mcmc_kwargs = dict()
        mcmc_kwargs.setdefault('num_burnin_steps', 1000)
        mcmc_kwargs.setdefault('parallel_iterations', 1)

        samples, log_accept_ratio = _tf_sample(self.model, prior_x, self.obs, self.total_var,
                                               n_samples, mcmc_kwargs, kernel_kwargs)

        print("Acceptance rate: {}".format(tf.math.exp(tf.minimum(log_accept_ratio, 0.)).numpy().mean()))
        return samples.numpy()[:, 0, :]


@tf.function
def _tf_sample(model, prior_x, obs, obs_var, n_samples, mcmc_kwargs, kernel_kwargs):

    def target(x):
        emulator_mean, emulator_var = model._predict(x)
        total_sd = tf.sqrt(tf.add(emulator_var, obs_var))
        diff = tf.subtract(obs, emulator_mean)
        return _target_log_likelihood(prior_x, x, diff, total_sd)

    def trace_log_accept_ratio(states, previous_kernel_results):
        return previous_kernel_results.log_accept_ratio

    samples, log_accept_ratio = tfp.mcmc.sample_chain(
        num_results=n_samples,
        current_state=tf.ones((1, model.n_params), dtype='float64') * 0.5,
        kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target, **kernel_kwargs),
        trace_fn=trace_log_accept_ratio, **mcmc_kwargs
    )
    return samples, log_accept_ratio


@tf.function
def _target_log_likelihood(prior_x, x, diff, total_sd):
    # I think doing this inside the tf_function is slowing down the sampling, but I can't
    #  see any other way of incorporating the emulator uncertainty
    diff_dist = tfd.Independent(
        tfd.Normal(loc=tf.zeros(diff.shape[0], dtype=tf.float64), scale=total_sd),
        reinterpreted_batch_ndims=1, name='model')

    return prior_x.log_prob(x) + diff_dist.log_prob(diff)


@tf.function
def _target_log_likelihood_non_independent(prior_x, x, diff, total_sd):

    # Sum the probabilities from this (multivariate) distributions
    diff_dist = tfd.Normal(loc=tf.zeros(diff.shape[0], dtype=tf.float64),
                           scale=total_sd)
    return prior_x.log_prob(x) + tf.reduce_sum(diff_dist.log_prob(diff))
