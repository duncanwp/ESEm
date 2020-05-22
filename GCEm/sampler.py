from abc import ABC
import tensorflow as tf
import numpy as np


# TODO: I need to define some distance metrics (including uncertainty?) Should these be functions, or objects?
#  Should this be passed to __init__, or calibrate?

# TODO: I'm not yet sure how the MCMC sampling works so this might need adjusting
from tqdm import tqdm


class Sampler(ABC):
    """
    A class that efficiently samples a Model object for posterior inference
    """

    def __init__(self, model, obs, var_obs=0., log_obs=False):
        """

        """
        self.model = model
        self.log_obs = log_obs
        self.obs = obs
        self.var_obs = var_obs

    def calibrate(self, objective, prior=None):
        """
        This is the call that does the actual inference.

        It should call model.sample over the prior, compare with the objective, and then output a posterior
        distribution

        :param objective: This is an Iris cube of observations
        :param prior: Ideally this would either be a numpy array or a tf.probability.distribution, could default to
        uniforms
        :return:
        """
        pass


class tf_tqdm(object):
    """
    A progress bar suitable for reporting on progress iterating over a TF dataset
    """
    def __init__(self, unit='sample', batch_size=1, total=None):
        import io
        self.unit = unit
        self.batch_size = batch_size
        self.total = total
        self.bar = tqdm(file=io.StringIO(), unit=unit, total=int(total))

    def __call__(self, ds, *args, **kwargs):
        def advance_tqdm(e):
            def fn():
                self.bar.update(self.batch_size)
                # Print the status update manually.
                print('\r', end='')
                print(repr(self.bar), end='')

            tf.py_function(fn, [], [])
            return e

        return ds.map(advance_tqdm)


@tf.function
def constrain(implausibility, tolerance=0., threshold=3.0):
    """
        Return a boolean array indicating if each sample meets the implausibility criteria:

            I < T

    :param np.array implausibility: Distance of each sample from each observation (in S.Ds)
    :param float tolerance: The fraction of samples which are allowed to be over the threshold
    :param float threshold: The number of standard deviations a sample is allowed to be away from the obs
    :return np.array: Boolean array of samples which meet the implausibility criteria
    """
    # Return True (for a sample) if the number of implausibility measures greater
    #  than the threshold is less than or equal to the tolerance
    tolerance = tf.constant(tolerance, dtype=implausibility.dtype)
    threshold = tf.constant(threshold, dtype=implausibility.dtype)
    return tf.less_equal(
                tf.reduce_sum(tf.cast(tf.greater(implausibility, threshold), dtype=implausibility.dtype), axis=1),
                tf.multiply(tolerance, tf.cast(tf.shape(implausibility)[1], dtype=tolerance.dtype))
           )


def get_implausibility(model, obs, sample_points,
                       obs_uncertainty=0., interann_uncertainty=0.,
                       repres_uncertainty=0., struct_uncertainty=0., batch_size=1):
    """

    Each of the specified uncertainties are assumed to be normal and are added in quadrature

    NOTE - each of the uncertaintes are 1 sigma as compared to previous versions which assumed 2 sigma

    :param model:
    :param obs:
    :param sample_points:
    :param float obs_uncertainty: Fractional, relative (1 sigma) uncertainty in obervations
    :param float repres_uncertainty: Fractional, relative (1 sigma) uncertainty due to the spatial and temporal
     representitiveness of the observations
    :param float interann_uncertainty: Fractional, relative (1 sigma) uncertainty introduced when using a model run
     for a year other than that the observations were measured in.
    :param float struct_uncertainty: Fractional, relative (1 sigma) uncertainty in the model itself.
    :param int batch_size:
    :return:
    """

    #TODO: Could add an absolute uncertainty term here

    # Get the square of the absolute uncertainty and broadcast it across the batch (since it's the same for each sample)
    observational_var = np.broadcast_to(np.square(obs.data * obs_uncertainty), (batch_size, obs.shape[0]))
    respres_var = np.broadcast_to(np.square(obs.data * repres_uncertainty), (batch_size, obs.shape[0]))
    interann_var = np.broadcast_to(np.square(obs.data * interann_uncertainty), (batch_size, obs.shape[0]))
    struct_var = np.broadcast_to(np.square(obs.data * struct_uncertainty), (batch_size, obs.shape[0]))

    implausibility = _tf_implausibility(model, obs.data, sample_points,
                                        observational_var, interann_var,
                                        respres_var, struct_var, batch_size=batch_size,
                                        pbar=tf_tqdm(batch_size=batch_size,
                                                     total=sample_points.shape[0])
                                        )

    return model._post_process(implausibility, name_prefix='Implausibility in emulated ')


def batch_constrain(model, obs, sample_points,
                       obs_uncertainty=0., interann_uncertainty=0.,
                       repres_uncertainty=0., struct_uncertainty=0.,
                       tolerance=0., threshold=3.0, batch_size=1):
    """

    Each of the specified uncertainties are assumed to be normal and are added in quadrature

    NOTE - each of the uncertaintes are 1 sigma as compared to previous versions which assumed 2 sigma

    :param model:
    :param obs:
    :param sample_points:
    :param float obs_uncertainty: Fractional, relative (1 sigma) uncertainty in obervations
    :param float repres_uncertainty: Fractional, relative (1 sigma) uncertainty due to the spatial and temporal
     representitiveness of the observations
    :param float interann_uncertainty: Fractional, relative (1 sigma) uncertainty introduced when using a model run
     for a year other than that the observations were measured in.
    :param float struct_uncertainty: Fractional, relative (1 sigma) uncertainty in the model itself.
    :param int batch_size:
    :return:
    """

    #TODO: Could add an absolute uncertainty term here

    # Get the square of the absolute uncertainty and broadcast it across the batch (since it's the same for each sample)
    observational_var = np.broadcast_to(np.square(obs.data * obs_uncertainty), (batch_size, obs.shape[0]))
    respres_var = np.broadcast_to(np.square(obs.data * repres_uncertainty), (batch_size, obs.shape[0]))
    interann_var = np.broadcast_to(np.square(obs.data * interann_uncertainty), (batch_size, obs.shape[0]))
    struct_var = np.broadcast_to(np.square(obs.data * struct_uncertainty), (batch_size, obs.shape[0]))

    valid_samples = _tf_constrain(model, obs.data, sample_points,
                                  observational_var, interann_var,
                                  respres_var, struct_var,
                                  tolerance=tolerance, threshold=threshold,
                                  batch_size=batch_size,
                                  pbar=tf_tqdm(batch_size=batch_size,
                                               total=sample_points.shape[0]))

    return valid_samples


@tf.function
def _tf_constrain(model, obs, sample_points,
                  observational_var, interann_var, respres_var, struct_var,
                  tolerance, threshold, batch_size, pbar):
    """

    Each of the specified uncertainties are assumed to be normal and are added in quadrature

    NOTE - each of the uncertaintes are 1 sigma as compared to previous versions which assumed 2 sigma

    :param model:
    :param Tensor obs:
    :param Tensor sample_points:
    :param Tensor observational_var: Variance in obervations
    :param Tensor respres_var: Fractional, relative (1 sigma) uncertainty due to the spatial and temporal
     representitiveness of the observations
    :param Tensor interann_var: Fractional, relative (1 sigma) uncertainty introduced when using a model run
     for a year other than that the observations were measured in.
    :param Tensor struct_var: Fractional, relative (1 sigma) uncertainty in the model itself.
    :param int batch_size:
    :return:
    """
    with tf.device('/gpu:{}'.format(model._GPU)):

        sample_T = tf.data.Dataset.from_tensor_slices(sample_points)
        dataset = sample_T.batch(batch_size)

        all_valid = tf.zeros((0, ), dtype=tf.bool)

        for data in pbar(dataset):
            # Get batch prediction
            emulator_mean, emulator_var = model._tf_predict(data)

            implausibility = _calc_implausibility(emulator_mean, obs,
                                                  emulator_var, interann_var,
                                                  observational_var, respres_var,
                                                  struct_var)

            valid_samples = constrain(implausibility, tolerance, threshold)
            all_valid = tf.concat([all_valid, valid_samples], 0)

    return all_valid


@tf.function
def _calc_implausibility(emulator_mean, obs, emulator_var, interann_var, observational_var, respres_var, struct_var):
    tot_sd = tf.sqrt(tf.add_n([emulator_var, observational_var, respres_var, interann_var, struct_var]))
    implausibility = tf.divide(tf.abs(tf.subtract(emulator_mean, obs)), tot_sd)
    return implausibility


@tf.function
def _tf_implausibility(model, obs, sample_points,
                       observational_var, interann_var,
                       respres_var, struct_var, batch_size, pbar):
    """

    Each of the specified uncertainties are assumed to be normal and are added in quadrature

    NOTE - each of the uncertaintes are 1 sigma as compared to previous versions which assumed 2 sigma

    :param model:
    :param Tensor obs:
    :param Tensor sample_points:
    :param Tensor observational_var: Variance in obervations
    :param Tensor respres_var: Fractional, relative (1 sigma) uncertainty due to the spatial and temporal
     representitiveness of the observations
    :param Tensor interann_var: Fractional, relative (1 sigma) uncertainty introduced when using a model run
     for a year other than that the observations were measured in.
    :param Tensor struct_var: Fractional, relative (1 sigma) uncertainty in the model itself.
    :param int batch_size:
    :return:
    """
    with tf.device('/gpu:{}'.format(model._GPU)):

        sample_T = tf.data.Dataset.from_tensor_slices(sample_points)

        dataset = sample_T.batch(batch_size)

        all_implausibility = tf.zeros((0, obs.shape[0]), dtype=sample_points.dtype)

        for data in pbar(dataset):
            # Get batch prediction
            emulator_mean, emulator_var = model._tf_predict(data)

            implausibility = _calc_implausibility(emulator_mean, obs,
                                                  emulator_var, interann_var,
                                                  observational_var, respres_var,
                                                  struct_var)

            all_implausibility = tf.concat([all_implausibility, implausibility], 0)

    return all_implausibility


@tf.function
def _tf_stats(model, sample_points, batch_size, pbar):
    with tf.device('/gpu:{}'.format(model._GPU)):
        sample_T = tf.data.Dataset.from_tensor_slices(sample_points)
        dataset = sample_T.batch(batch_size)
        n_samples = tf.shape(sample_points)[0]

        tot_s = tf.constant(0., dtype=model.dtype)  # Proportion of valid samples required
        tot_s2 = tf.constant(0., dtype=model.dtype)  # Proportion of valid samples required

        for data in pbar(dataset):
            # Get batch prediction
            emulator_mean, _ = model._tf_predict(data)

            # Get sum of x and sum of x**2
            tot_s += tf.reduce_sum(emulator_mean, axis=0)
            tot_s2 += tf.reduce_sum(tf.square(emulator_mean), axis=0)

    n_samples = tf.cast(n_samples, dtype=model.dtype)  # Make this a float to allow division
    # Calculate the resulting first two moments
    mean = tot_s / n_samples
    sd = tf.sqrt((tot_s2 - (tot_s * tot_s) / n_samples) / (n_samples - 1))

    return mean, sd


def batch_stats(model, sample_points, batch_size=1):
    mean, sd = _tf_stats(model, tf.constant(sample_points),
                         tf.constant(batch_size, dtype=tf.int64),
                         pbar=tf_tqdm(batch_size=batch_size,
                                      total=sample_points.shape[0]))
    # Wrap the results in a cube (but pop off the sample dim which will always
    #  only be one in this case
    return (model._post_process(mean, 'Ensemble mean ')[0],
            model._post_process(sd, 'Ensemble standard deviation in ')[0])
