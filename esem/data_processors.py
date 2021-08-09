from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf


class DataProcessor(ABC):

    """
    A utility class for transparently processing (transforming) numpy arrays and
    un-processing TensorFlow Tensors to aid in emulation.

    See the `API documentation <../api.html#dataprocessor>`_ for a list of concrete
    classes implementing this interface.
    """

    @abstractmethod
    def process(self, data):
        pass

    @abstractmethod
    def unprocess(self, mean, variance):
        pass


class Whiten(DataProcessor):
    """
    Scale the data to have zero mean and unit variance
    """

    def process(self, data):
        # Useful for whitening the training data
        self.std_dev = np.std(data, axis=0, keepdims=True)
        self.mean = np.mean(data, axis=0, keepdims=True)

        # Add a tiny epsilon to avoid division by zero
        return (data - self.mean) / (self.std_dev + 1e-12)

    def unprocess(self, mean, variance):
        return (mean * self.std_dev) + self.mean, variance * self.std_dev


class Normalise(DataProcessor):
    """
    Linearly scale the data to lie between [0, 1]
    """

    def process(self, data):
        # Useful for normalising the training data
        self.min = np.min(data, axis=0, keepdims=True)
        self.max = np.max(data, axis=0, keepdims=True)

        return (data - self.min) / (self.max - self.min)

    def unprocess(self, mean, variance):
        return mean * (self.max - self.min) + self.min, variance * (self.max - self.min)


class Log(DataProcessor):
    """
    Return log(x + c) where c can be specified.
    """

    def __init__(self, constant=0.):
        self.constant = constant

    def process(self, data):
        return np.log(data + self.constant)

    def unprocess(self, mean, variance):
        mean_res = tf.exp(mean) - tf.constant(self.constant, dtype=mean.dtype)
        return mean_res, tf.exp(mean) * variance


class Flatten(DataProcessor):
    """
    Flatten all dimensions except the leading one
    """

    def process(self, data):
        # Flatten the data
        self.original_shape = data.shape
        return data.reshape((data.shape[0], -1))

    def unprocess(self, mean, variance):
        # Reshape the output to the original shape, with a leading ensemble
        #  dimension in case we're outputting a batch of samples
        return (tf.reshape(mean, (-1,) + self.original_shape[1:]),
                tf.reshape(variance, (-1,) + self.original_shape[1:]))


class Reshape(DataProcessor):
    """
    Ensure the training data is the right shape for the ConvNet
    """

    def process(self, data):
        self.add_newaxis = False

        if data.ndim < 3:
            raise ValueError("Training data must have at least three "
                             "dimensions (including the sample dimension)")
        elif data.ndim == 3:
            self.add_newaxis = True
            data = data[..., np.newaxis]
        elif data.ndim > 4:
            raise ValueError("Training data must have at most four dimensions"
                             "(including the sample dimension)")
        return data

    def unprocess(self, mean, variance):
        if self.add_newaxis:
            mean = mean[..., 0]
            variance = variance[..., 0]
        return mean, variance


class Recast(DataProcessor):
    """
    Cast the data to a given type
    """

    def __init__(self, new_type):
        self.new_type = new_type

    def process(self, data):
        self.old_type = data.dtype
        return data.astype(self.new_type)

    def unprocess(self, mean, variance):
        # I just leave this currently
        return mean, variance
