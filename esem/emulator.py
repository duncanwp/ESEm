import numpy as np
import tensorflow as tf

from esem.wrappers import wrap_data
from esem.model_adaptor import ModelAdaptor


class Emulator:
    """
    A class wrapping a statistical emulator

    Attributes
    ----------

    training_data : esem.wrappers.DataWrapper
        A wrapped representation of the training data
    model: ModelAdaptor
        The underlying model which performs the emulation
    name : str
        A human-readable name for the model

    """

    def __init__(self, model, training_params, training_data, name='', gpu=0):
        """

        Parameters
        ----------
        model: ModelAdaptor
            The (compiled but not trained) model to be wrapped
        training_params: pd.DataFrame or array-like
            The training parameters (X)
        training_data: esem.wrappers.DataWrapper or xarray.DataArray or iris.Cube or array-like
            The training data - the leading dimension should represent training samples (Y)
        name: str
            Human readable name for the model
        gpu: int
            The machine GPU to assign this model to
        """
        from contextlib import nullcontext
        import pandas as pd

        assert isinstance(model, ModelAdaptor), "Model must be an instance of type ModelAdaptor"
        self.model = model

        self.training_data = wrap_data(training_data)

        self.name = name or self.training_data.name()

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

    def train(self, verbose=False, **kwargs):
        """
        Train on the training data

        Parameters
        ----------
        verbose: bool
            Print verbose training output to screen
        """
        self.model.train(self.training_params, self.training_data.data, verbose=verbose, **kwargs)

    def predict(self, x, *args, **kwargs):
        """
        Make a prediction using a trained emulator

        Parameters
        ----------
        x: pd.DataFrame or array-like
            The points at which to make predictions from the model
        args:
            The specific arguments needed for prediction with this model
        kwargs:
            Any keyword arguments that might need to be passed through to the model

        Returns
        -------
        Emulated prediction and variance with the same type as `self.training_data`
        """
        import pandas as pd

        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()

        # TODO: Add a warning if .train hasn't been called?
        mean, var = self._predict(x, *args, **kwargs)

        return (self.training_data.wrap(mean, 'Emulated '),
                self.training_data.wrap(var, 'Variance in emulated '))

    def _predict(self, x, *args, **kwargs):
        """
        The (internal) predict interface used by e.g., a sampler. It is still in tf but has been post-processed
        to allow comparison with obs.

        Parameters
        ----------
        x: array-like
            The points at which to make predictions from the model
        args:
            The specific arguments needed for prediction with this model
        kwargs:
            Any keyword arguments that might need to be passed through to the model

        Returns
        -------
        Emulated prediction and variance as either np.ndarray or tf.Tensor
        """
        with self.tf_device_context:
            mean, var = self.model.predict(x, *args, **kwargs)
        # Left un-nested for readability
        return self.training_data.process_wrapper(mean, var)

    def batch_stats(self, sample_points, batch_size=1):
        """
        Return mean and standard deviation in model predictions over samples,
        without storing the intermediate predicions in memory to enable
        evaluating large models over more samples than could fit in memory

        Parameters
        ----------
        sample_points: pd.DataFrame or array-like
            The parameter values at which to sample the emulator
        batch_size: int
            The number of samples to calculate in each batch. This can be optimised to fill the available (GPU) memory

        Returns
        -------
        The batch mean and standard deviation with the same type as `self.training_data`
        """
        import pandas as pd
        from esem.utils import tf_tqdm

        if isinstance(sample_points, pd.DataFrame):
            sample_points = sample_points.to_numpy()

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