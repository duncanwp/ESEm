from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np


class Model(ABC):
    """
    A class representing a statistical emulator

    Attributes
    ----------

    training_data : iris.cube.Cube
        The Iris cube representing the training data
    name : str
        A human-readable name for the model

    """

    def __init__(self, training_params, training_data, name='', gpu=0, *args, **kwargs):
        """

        :param pd.DataFrame training_params: The training parameters
        :param iris.cube.Cube training_data: The training data - the leading dimension should represent training samples
        :param str name: Human readable name for the model
        :param int gpu: The machine GPU to assign this model to
        """
        import iris.cube
        from contextlib import nullcontext
        import pandas as pd

        if isinstance(training_data, iris.cube.Cube):
            self.training_cube = training_data
            self.training_data = training_data.data
            self.name = name or training_data.name()
        else:
            self.training_cube = None
            self.training_data = training_data
            self.name = name

        if isinstance(training_params, pd.DataFrame):
            self.training_params = training_params.to_numpy()
        else:
            self.training_params = training_params
        self.n_params = training_params.shape[1]
        self.dtype = training_data.dtype
        # Set the GPU to use (if provided)
        self.tf_device_context = tf.device('/gpu:{}'.format(gpu)) if gpu is not None else nullcontext

        # Useful for whitening the training data
        self.training_std_dev = np.std(self.training_data.data, axis=0, keepdims=True)
        self.training_mean = np.mean(self.training_data.data, axis=0, keepdims=True)

        # Useful for normalising the training data
        self.training_min = np.min(self.training_data.data, axis=0, keepdims=True)
        self.training_max = np.max(self.training_data.data, axis=0, keepdims=True)

        # Perform any pre-processing of the data the model might require
        self._pre_process()

        # Construct the model
        self.model = self._construct(*args, **kwargs)

    def scale(self, data):
        return (data - self.training_min) / (self.training_max - self.training_min)

    def rescale(self, data):
        return data * (self.training_max - self.training_min) + self.training_min

    def whiten(self, data):
        # Add a tiny epsilon to avoid division by zero
        return (data - self.training_mean) / (self.training_std_dev + 1e-12)

    def un_whiten(self, data):
        return (data * self.training_std_dev) + self.training_mean

    @abstractmethod
    def _construct(self, *args, **kwargs):
        """
        Construct the model and compile if necessary (but don't train it yet)
        :return:
        """
        pass

    def _pre_process(self):
        """
         Any necessary rescaling or weightings are performed here
        :return:
        """
        pass

    def _post_process(self, data, name_prefix='Emulated '):
        """
        Reshape output if needed and wrap back in a cube if one was provided
         for training

        :param np.array data: Model output to post-process
        :param args:
        :param kwargs:
        :return:
        """
        from iris.cube import Cube
        from iris.coords import DimCoord

        if isinstance(data, tf.Tensor):
            data = data.numpy()

        if (data is not None) and (data.size > 0) and (self.training_cube is not None):

            # Ensure we have a leading sample dimension
            data = data.reshape((-1,) + self.training_cube.shape[1:])

            # Create a coordinate for the sample dimension (which could be a different length to the original)
            sample_coord = [(DimCoord(np.arange(data.shape[0]), long_name="sample"), 0)]
            # Pull out the other coordinates - we can't rely on these being in order unfortunately, but we know the
            #  member dimension was the 0th
            other_coords = [(c, self.training_cube.coord_dims(c)) for c in self.training_cube.dim_coords if
                            self.training_cube.coord_dims(c) != (0,)]
            out = Cube(data,
                       long_name=name_prefix + self.training_cube.name(),
                       units=self.training_cube.units,
                       dim_coords_and_dims=other_coords + sample_coord,
                       aux_coords_and_dims=self.training_cube._aux_coords_and_dims)
        else:
            out = data
        return out

    @abstractmethod
    def train(self, verbose=False):
        """
        Train on the training data
        :return:
        """
        pass

    def predict(self, *args, **kwargs):
        mean, var = self._tf_predict(*args, **kwargs)

        return (self._post_process(mean, 'Emulated '),
                self._post_process(var, 'Variance in emulated '))

    @property
    @abstractmethod
    def _tf_predict(self):
        """
        This is either the tf model which I can then call, or a generator over the model.predict (in tf, so it's quick)

        The sampler (using either tf.probability.mcmc and my ABC method) can then just call this to get samples

        :return:
        """
        pass

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
        return (self._post_process(mean.numpy(), 'Ensemble mean ')[0],
                self._post_process(sd.numpy(), 'Ensemble standard deviation in ')[0])


@tf.function
def _tf_stats(model, sample_points, batch_size, pbar):
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

