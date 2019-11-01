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

    def __init__(self, training_data, name='', GPU=0):
        """

        :param iris.cube.Cube training_data: The training data - the leading dimension should represent training samples
        :param str name: Human readable name for the model
        :param int GPU: The machine GPU to assign this model to
        """

        self.training_data = training_data
        self.name = name
        self._GPU = GPU

        self._NP_TYPE = np.float64
        self._tf_graph = tf.Graph()
        self._tf_sess = tf.Session(graph=self._tf_graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                               log_device_placement=True))

        with self._tf_sess.as_default(), self._tf_sess.graph.as_default():
            self._TF_TYPE = tf.float64

    @abstractmethod
    def fit(self):
        """
        Train on the training data
        :return:
        """
        pass

    @abstractmethod
    def predict(self):
        """
        Basically just a wrapper around 'predict' and 'predict_y'

        :return:
        """
        pass

    @property
    @abstractmethod
    def _sample(self):
        """
        This is either the tf model which I can then call, or a generator over the model.predict (in tf, so it's quick)

        The sampler (using either tf.probability.mcmc and my ABC method) can then just call this to get samples

        :return:
        """
        pass
