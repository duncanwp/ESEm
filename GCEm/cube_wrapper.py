import numpy as np
import tensorflow as tf


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
    def data(self):
        return self.data_wrapper.data

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
