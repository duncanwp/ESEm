import numpy as np
import tensorflow as tf

from GCEm.cube_wrapper import CubeWrapper
from GCEm.model_adaptor import ModelAdaptor


class Emulator:
    """
    A class wrapping a statistical emulator

    Attributes
    ----------

    training_data : iris.cube.Cube
        The Iris cube representing the training data
    name : str
        A human-readable name for the model

    """

    def __init__(self, model, training_params, training_data, name='', gpu=0):
        """

        :param ModelAdaptor model: The (compiled) model to be wrapped
        :param pd.DataFrame training_params: The training parameters
        :param GCEm.cube_wrapper.CubeWrapper training_data: The training data - the leading dimension should represent training samples
        :param str name: Human readable name for the model
        :param int gpu: The machine GPU to assign this model to
        :param None or list of DataProcessors data_processors: A list of data processors to apply to the data
        """
        from contextlib import nullcontext
        from iris.cube import Cube
        import pandas as pd

        assert isinstance(model, ModelAdaptor), "Model must be an instance of type ModelAdaptor"
        self.model = model

        if isinstance(training_data, CubeWrapper):
            self.training_data = training_data
        elif isinstance(training_data, np.ndarray) or isinstance(training_data, Cube):
            self.training_data = CubeWrapper(training_data)
        else:
            raise ValueError("Training data must be a cube, numpy array or CubeWrapper instance")

        self.name = name or training_data.name()

        if isinstance(training_params, pd.DataFrame):
            self.training_params = training_params.to_numpy()
        elif isinstance(training_params, np.ndarray):
            self.training_params = training_params
        else:
            raise ValueError("Training parameters must be a numpy array or pd.DataFrame instance")
        self.n_params = training_params.shape[1]

        # Set the GPU to use (if provided)
        self.tf_device_context = tf.device('/gpu:{}'.format(gpu)) if gpu is not None else nullcontext

        # Store the training data dtype (after pre-processing in case it has changed)
        self.dtype = self.training_data.dtype

    def train(self, verbose=False, *args, **kwargs):
        """
        Train on the training data
        :return:
        """
        self.model.train(self.training_params, self.training_data.data_wrapper.data, verbose=verbose, *args, **kwargs)

    def predict(self, *args, **kwargs):
        """
        Make a prediction using a trained emulator

        :param args:
        :param kwargs:
        :return iris.Cube: Emulated results
        """
        # TODO: Add a warning if .train hasn't been called?
        mean, var = self._predict(*args, **kwargs)

        return (self.training_data.wrap(mean, 'Emulated '),
                self.training_data.wrap(var, 'Variance in emulated '))

    def _predict(self, *args, **kwargs):
        """
        The (internal) predict interface used by e.g., a sampler. It is still in tf but has been post-processed
         to allow comparison with obs.

        :param args:
        :param kwargs:
        :return:
        """
        with self.tf_device_context:
            mean, var = self.model.predict(*args, **kwargs)
        # Left un-nested for readability
        return self.training_data.data_wrapper.post_process(mean, var)

    def batch_stats(self, sample_points, batch_size=1):
        """
        Return mean and standard deviation in model predictions over samples,
         without storing the intermediate predicions in memory to enable
         evaluating large models over more samples than could fit in memory

        :param sample_points:
        :param int batch_size:
        :return:
        """
        from GCEm.utils import tf_tqdm
        with self.tf_device_context:
            # TODO: Make sample points optional and just sample from a uniform distribution if not provided
            mean, sd = _tf_stats(self, tf.constant(sample_points),
                                 tf.constant(batch_size, dtype=tf.int64),
                                 pbar=tf_tqdm(batch_size=batch_size,
                                              total=sample_points.shape[0]))
        # Wrap the results in a cube (but pop off the sample dim which will always
        #  only be one in this case
        return (self.training_data.wrap(mean.numpy(), 'Ensemble mean ')[0],
                self.training_data.wrap(sd.numpy(), 'Ensemble standard deviation in ')[0])


@tf.function
def _tf_stats(model, sample_points, batch_size, pbar):
    sample_T = tf.data.Dataset.from_tensor_slices(sample_points)
    dataset = sample_T.batch(batch_size)
    n_samples = tf.shape(sample_points)[0]

    tot_s = tf.constant(0., dtype=model.dtype)  # Proportion of valid samples required
    tot_s2 = tf.constant(0., dtype=model.dtype)  # Proportion of valid samples required

    for data in pbar(dataset):
        # Get batch prediction
        emulator_mean, _ = model._predict(data)

        # Get sum of x and sum of x**2
        tot_s += tf.reduce_sum(emulator_mean, axis=0)
        tot_s2 += tf.reduce_sum(tf.square(emulator_mean), axis=0)

    n_samples = tf.cast(n_samples, dtype=model.dtype)  # Make this a float to allow division
    # Calculate the resulting first two moments
    mean = tot_s / n_samples
    sd = tf.sqrt((tot_s2 - (tot_s * tot_s) / n_samples) / (n_samples - 1))

    return mean, sd