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
                 repres_uncertainty=0., struct_uncertainty=0.):
        """
        :param GCEm.model.Model model:
        :param iris.cube.Cube obs: The objective
        :param float obs_uncertainty: Fractional, relative (1 sigma) uncertainty in observations
        :param float repres_uncertainty: Fractional, relative (1 sigma) uncertainty due to the spatial and temporal
         representitiveness of the observations
        :param float interann_uncertainty: Fractional, relative (1 sigma) uncertainty introduced when using a model run
         for a year other than that the observations were measured in.
        :param float struct_uncertainty: Fractional, relative (1 sigma) uncertainty in the model itself.
        """
        self.model = model
        self.obs = obs

        # TODO: Could add an absolute uncertainty term here
        # Get the square of the absolute uncertainty and broadcast it across the batch
        #  (since it's the same for each sample)
        observational_var = np.reshape(np.square(obs.data * obs_uncertainty), (1, obs.shape[0]))
        respres_var = np.reshape(np.square(obs.data * repres_uncertainty), (1, obs.shape[0]))
        interann_var = np.reshape(np.square(obs.data * interann_uncertainty), (1, obs.shape[0]))
        struct_var = np.reshape(np.square(obs.data * struct_uncertainty), (1, obs.shape[0]))
        self.total_var = sum([observational_var, respres_var, interann_var, struct_var])

    def sample(self, prior_x=None, n_samples=1):
        """
        This is the call that does the actual inference.

        It should call model.sample over the prior, compare with the objective, and then output samples
        from the posterior distribution

        :param tensorflow_probability.distribution prior_x: The distribution to sample parameters from.
         By default it will uniformly sample the unit N-D hypercube
        :param int n_samples: The number of samples to draw
        :return np.array : Array of samples
        """
        pass


class MCMCSampler(Sampler):
    """
    Sample from the posterior using the TensorFlow Markov-Chain Monte-Carlo (MCMC)
     sampling tools. It uses a HamiltonianMonteCarlo kernel.
    """

    def sample(self, prior_x=None, n_samples=1, kernel_kwargs=None, mcmc_kwargs=None):
        """
        This is the call that does the actual inference.

        It should call model.sample over the prior, compare with the objective, and then output a posterior
        distribution

        :param tensorflow_probability.distribution prior_x: The distribution to sample parameters from.
         By default it will uniformly sample the unit N-D hypercube
        :param int n_samples: The number of samples to draw
        :param dict kernel_kwargs: kwargs for the MCMC kernel
        :param dict mcmc_kwargs: kwargs for the MCMC sampler
        :return:
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

        samples, log_accept_ratio = _tf_sample(self.model, prior_x, self.obs.data, self.total_var,
                                               n_samples, mcmc_kwargs, kernel_kwargs)

        print("Acceptance rate: {}".format(tf.math.exp(tf.minimum(log_accept_ratio, 0.)).numpy().mean()))
        return samples.numpy()[:, 0, :]


@tf.function
def _tf_sample(model, prior_x, obs, obs_var, n_samples, mcmc_kwargs, kernel_kwargs):

    def target(x):
        emulator_mean, emulator_var = model._tf_predict(x)
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
