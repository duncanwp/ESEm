import tensorflow as tf

from GCEm import Sampler
from GCEm.utils import tf_tqdm


class ABCSampler(Sampler):
    """
    Sample from the posterior using Approximate Bayesian Computation (ABC).
     This is a style of rejection sampling
    """

    def sample(self, prior_x=None, n_samples=1, tolerance=0., threshold=3.):
        """
        This is the call that does the actual inference.

        It should call model.sample over the prior, compare with the objective, and then output a posterior
        distribution

        :param tensorflow_probability.distribution prior_x: The distribution to sample parameters from.
         By default it will uniformly sample the unit N-D hypercube
        :param int n_samples: The number of samples to draw
        :param float tolerance: The fraction of samples which are allowed to be over the threshold
        :param float threshold: The number of standard deviations a sample is allowed to be away from the obs
        :return:
        """
        import tensorflow_probability as tfp
        tfd = tfp.distributions

        if prior_x is None:
            prior_x = tfd.Uniform(low=tf.zeros(self.model.n_params, dtype=tf.float64),
                                  high=tf.ones(self.model.n_params, dtype=tf.float64))
        with self.model.tf_device_context:
            n_iterations, samples = _tf_sample(self.model, self.obs, prior_x, n_samples,
                              self.total_var, tolerance, threshold)
        print("Acceptance rate: {}".format(n_samples/n_iterations.numpy()))
        return samples.numpy()

    def get_implausibility(self, sample_points, batch_size=1):
        """

        :param sample_points:
        :param int batch_size:
        :return:
        """
        import pandas as pd
        if isinstance(sample_points, pd.DataFrame):
            sample_points = sample_points.to_numpy()
        else:
            sample_points = sample_points

        with self.model.tf_device_context:
            implausibility = _tf_implausibility(self.model, self.obs, sample_points,
                                                self.total_var, batch_size=batch_size,
                                                pbar=tf_tqdm(batch_size=batch_size,
                                                             total=sample_points.shape[0])
                                                )

        return self.model.training_data.wrap(implausibility.numpy(), name_prefix='Implausibility in emulated ')

    def batch_constrain(self, sample_points, tolerance=0., threshold=3.0, batch_size=1):
        """

        :param sample_points:
        :param float tolerance: The fraction of samples which are allowed to be over the threshold
        :param float threshold: The number of standard deviations a sample is allowed to be away from the obs
        :param int batch_size:
        :return:
        """
        import pandas as pd
        if isinstance(sample_points, pd.DataFrame):
            sample_points = sample_points.to_numpy()
        else:
            sample_points = sample_points

        with self.model.tf_device_context:
            valid_samples = _tf_constrain(self.model, self.obs, sample_points,
                                          self.total_var,
                                          tolerance=tolerance, threshold=threshold,
                                          batch_size=batch_size,
                                          pbar=tf_tqdm(batch_size=batch_size,
                                                       total=sample_points.shape[0]))

        return valid_samples.numpy()


@tf.function
def constrain(implausibility, tolerance=0., threshold=3.0):
    """
        Return a boolean array indicating if each sample meets the implausibility criteria:

            I < T

        Return True (for a sample) if the number of implausibility measures greater
         than the threshold is less than or equal to the tolerance

    :param np.array implausibility: Distance of each sample from each observation (in S.Ds)
    :param float tolerance: The fraction of samples which are allowed to be over the threshold
    :param float threshold: The number of standard deviations a sample is allowed to be away from the obs
    :return np.array: Boolean array of samples which meet the implausibility criteria
    """
    total_obs = tf.cast(tf.reduce_prod(tf.shape(implausibility)[1:]), dtype=implausibility.dtype)
    # Calculate the absolute tolerance
    abs_tolerance = tf.multiply(tf.constant(tolerance, dtype=implausibility.dtype), total_obs)
    threshold = tf.constant(threshold, dtype=implausibility.dtype)

    # Count the number of implausible observations against the threshold
    n_implausible = tf.reduce_sum(tf.cast(tf.greater(implausibility, threshold), dtype=implausibility.dtype),
                                  # Reduce over all dims except the first
                                  axis=(range(1, len(implausibility.shape))))
    # Compare with the tolerance
    return tf.less_equal(n_implausible, abs_tolerance)


@tf.function
def _calc_implausibility(emulator_mean, obs, tot_sd):
    return tf.divide(tf.abs(tf.subtract(emulator_mean, obs)), tot_sd)


@tf.function
def _tf_constrain(model, obs, sample_points, total_variance,
                  tolerance, threshold, batch_size, pbar):
    """

    :param model:
    :param Tensor obs:
    :param Tensor sample_points:
    :param Tensor total_variance: Total variance in observational comparison
    :param int batch_size:
    :return:
    """
    sample_T = tf.data.Dataset.from_tensor_slices(sample_points)
    dataset = sample_T.batch(batch_size)

    all_valid = tf.zeros((0, ), dtype=tf.bool)

    for data in pbar(dataset):
        # Get batch prediction
        emulator_mean, emulator_var = model._predict(data)

        tot_sd = tf.sqrt(tf.add(emulator_var, total_variance))
        implausibility = _calc_implausibility(emulator_mean, obs, tot_sd)

        valid_samples = constrain(implausibility, tolerance, threshold)
        all_valid = tf.concat([all_valid, valid_samples], 0)

    return all_valid


@tf.function
def _tf_implausibility(model, obs, sample_points, total_variance,
                       batch_size, pbar):
    """

    :param model:
    :param Tensor obs:
    :param Tensor sample_points:
    :param Tensor total_variance: Total variance in observational comparison
    :param int batch_size:
    :return:
    """
    sample_T = tf.data.Dataset.from_tensor_slices(sample_points)

    dataset = sample_T.batch(batch_size)

    all_implausibility = tf.zeros((0, ) + obs.shape, dtype=sample_points.dtype)

    for data in pbar(dataset):
        # Get batch prediction
        emulator_mean, emulator_var = model._predict(data)

        tot_sd = tf.sqrt(tf.add(emulator_var, total_variance))
        implausibility = _calc_implausibility(emulator_mean, obs, tot_sd)

        all_implausibility = tf.concat([all_implausibility, implausibility], 0)

    return all_implausibility


@tf.function
def _tf_sample(model, obs, dist, n_sample_points, total_variance,
                  tolerance, threshold):
    """

    :param model:
    :param Tensor obs:
    :param Tensor sample_points:
    :param Tensor total_variance: Total variance in observational comparison
    :param int batch_size:
    :return:
    """
    # Empty array in which to hold the samples (and the number of iterations)
    samples_i = tf.zeros((0, model.n_params+1), dtype=tf.float64)
    i0 = tf.constant(0)

    _, res = tf.while_loop(
        lambda i, m: i < n_sample_points,
        lambda i, m: [i + 1,
                      tf.concat([m, get_valid_sample(model, obs, dist, threshold, tolerance, total_variance)],
                                axis=0)],
        loop_vars=[i0, samples_i],
        shape_invariants=[i0.get_shape(), tf.TensorShape([None, model.n_params+1])]
    )

    all_samples, iterations = res[:, :-1], res[:, -1:]
    n_iterations = tf.reduce_sum(iterations)
    return n_iterations, all_samples


@tf.function()
def get_valid_sample(model, obs, dist, threshold, tolerance, total_variance):
    """
    Given a distribution keep sampling it until it returns a valid sample

    :param model:
    :param obs:
    :param dist:
    :param threshold:
    :param tolerance:
    :param total_variance:
    :return Tensor(n_params + 1): the valid parameters and the number of iterations it took to find them
    """
    valid = dist.sample()
    count = tf.constant((1,), dtype=tf.float64)
    valid = tf.while_loop(
        lambda x, i: tf.math.logical_not(is_valid_sample(model, obs, x, threshold, tolerance, total_variance)),
        lambda x, i: (dist.sample(), i+1.),
        loop_vars=(valid, count)
    )
    return tf.reshape(tf.concat(valid, 0), (1, -1))


@tf.function
def is_valid_sample(model, obs, sample, threshold, tolerance, total_variance):
    """
     Given a sample determine if it is 'valid' or not

    :param model:
    :param obs:
    :param sample:
    :param threshold:
    :param tolerance:
    :param total_variance:
    :return bool:
    """
    emulator_mean, emulator_var = model._predict(tf.reshape(sample, (1, -1)))
    tot_sd = tf.sqrt(tf.add(emulator_var, total_variance))
    implausibility = _calc_implausibility(emulator_mean, obs, tot_sd)
    valid = constrain(implausibility, tolerance, threshold)[0]
    return valid
