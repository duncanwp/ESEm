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

    def __init__(self, training_data, name='', GPU=0, dtype=None):
        """

        :param iris.cube.Cube training_data: The training data - the leading dimension should represent training samples
        :param str name: Human readable name for the model
        :param int GPU: The machine GPU to assign this model to
        """
        import iris.cube

        if isinstance(training_data, iris.cube.Cube):
            self.training_cube = training_data
            self.training_data = training_data.data
            self.name = name or training_data.name()
        else:
            self.training_cube = None
            self.training_data = training_data
            self.name = name

        self.dtype = dtype if dtype is not None else training_data.dtype
        self._GPU = GPU

    def _post_process(self, data, name_prefix='Emulated '):
        """
        Reshape output if needed and wrap back in a cube if one was provided
         for training

        :param tf.Tensor data: Model output to post-process
        :param args:
        :param kwargs:
        :return:
        """
        from iris.cube import Cube
        from iris.coords import DimCoord
        # Reshape the output to the original shape, with a leading ensemble
        #  dimension in case we're outputting a batch of samples
        out = data.numpy().reshape((-1,) + self.training_data.shape[1:])
        if self.training_cube is not None:
            # Create a coordinate for the sample dimension (which could be a different length to the original)
            sample_coord = [(DimCoord(np.arange(out.shape[0]), long_name="sample"), 0)]
            # Pull out the other coordinates - we can't rely on these being in order unfortunately, but we know the
            #  member dimension was the 0th
            other_coords = [(c, self.training_cube.coord_dims(c)) for c in self.training_cube.dim_coords if
                            self.training_cube.coord_dims(c) != (0,)]
            out = Cube(out,
                       long_name=name_prefix + self.training_cube.name(),
                       units=self.training_cube.units,
                       dim_coords_and_dims=other_coords + sample_coord,
                       aux_coords_and_dims=self.training_cube._aux_coords_and_dims)

        return out

    @abstractmethod
    def train(self, X, params=None, verbose=False):
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
    def _tf_predict(self):
        """
        This is either the tf model which I can then call, or a generator over the model.predict (in tf, so it's quick)

        The sampler (using either tf.probability.mcmc and my ABC method) can then just call this to get samples

        :return:
        """
        pass
