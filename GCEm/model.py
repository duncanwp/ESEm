from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np


class ModelAdaptor(ABC):

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def train(self, training_params, training_data, verbose=False, *args, **kwargs):
        """
        Train on the training data
        :return:
        """
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        """
        This is either the tf model which I can then call, or a generator over the model.predict (in tf, so it's quick).
        This function

        The sampler (using either tf.probability.mcmc and my ABC method) can then just call this to get samples

        :return:
        """
        pass


class DataWrapper:

    def __init__(self, data, data_processors=None):
        self.data_processors = data_processors if data_processors is not None else []
        self._raw_data = data
        self._data = None

    @property
    def data(self):
        if self._data is not None:
            data = self._data
        else:
            data = self.pre_process(self._raw_data)
            self._data = data
        return data

    def __call__(self, *args, **kwargs):
        return self.post_process(*args, **kwargs)

    def pre_process(self, data):
        """
         Any necessary rescaling or weightings are performed here
        :return:
        """
        # Apply each in turn
        for processor in self.data_processors:
            data = processor.process(data)
        return data

    def post_process(self, mean, variance):
        """
         Any necessary reshaping or un-weightings are performed here

        :param np.array or tf.Tensor mean: Model mean output to post-process
        :param np.array or tf.Tensor variance: Model variance output to post-process
        :return:
        """
        # Check we were actually given some data to process
        if variance is None:
            variance = tf.ones_like(mean) * tf.constant([float('NaN')], dtype=mean.dtype)
        # Loop through the processors, undoing each process in reverse order
        for processor in self.data_processors[::-1]:
            mean, variance = processor.unprocess(mean, variance)

        return mean, variance


class CubeWrapper:

    def __init__(self, possible_cube, data_processors=None):
        import iris.cube

        if isinstance(possible_cube, iris.cube.Cube):
            self.cube = possible_cube
            data = possible_cube.data
        else:
            self.cube = None
            data = possible_cube

        self.data_wrapper = DataWrapper(data, data_processors)

    def name(self):
        return self.cube.name() if self.cube is not None else ''

    @property
    def dtype(self):
        return self.data_wrapper.data.dtype

    def wrap(self, data, name_prefix='Emulated '):
        """
        Wrap back in a cube if one was provided

        :param np.array data: Model output to wrap
        :param str name_prefix:
        :return:
        """
        from iris.cube import Cube
        from iris.coords import DimCoord

        if isinstance(data, tf.Tensor):
            data = data.numpy()

        if (data is not None) and (data.size > 0) and (self.cube is not None):

            # Ensure we have a leading sample dimension
            data = data.reshape((-1,) + self.cube.shape[1:])

            # Create a coordinate for the sample dimension (which could be a different length to the original)
            sample_coord = [(DimCoord(np.arange(data.shape[0]), long_name="sample"), 0)]
            # Pull out the other coordinates - we can't rely on these being in order unfortunately, but we know the
            #  member dimension was the 0th
            other_coords = [(c, self.cube.coord_dims(c)) for c in self.cube.dim_coords if
                            self.cube.coord_dims(c) != (0,)]
            out = Cube(data,
                       long_name=name_prefix + self.cube.name(),
                       units=self.cube.units,
                       dim_coords_and_dims=other_coords + sample_coord,
                       aux_coords_and_dims=self.cube._aux_coords_and_dims)
        else:
            out = data
        return out


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
        :param CubeWrapper training_data: The training data - the leading dimension should represent training samples
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

