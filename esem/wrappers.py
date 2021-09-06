import numpy as np
import tensorflow as tf


class ProcessWrapper:
    """
    This class handles applying any data pre- and post-processing by any provided DataProcessor
    """

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


class DataWrapper:
    """
    Provide a unified interface for numpy arrays, Iris Cube's and xarray DataArrays.
    Emulation outputs will be provided based on the provided input type, preserving appropriate
    metadata.
    """

    def __init__(self, data, data_processors=None):

        self.process_wrapper = ProcessWrapper(data, data_processors)

    def name(self):
        return ''

    @property
    def data(self):
        return self.process_wrapper.data

    @property
    def dtype(self):
        return self.process_wrapper.data.dtype

    def wrap(self, data, name_prefix='Emulated '):
        """
        Wrap back in a cube if one was provided

        :param np.array data: Model output to wrap
        :param str name_prefix:
        :return:
        """
        if isinstance(data, tf.Tensor):
            data = data.numpy()
        return data


class CubeWrapper(DataWrapper):

    def __init__(self, cube, data_processors=None):
        self.cube = cube
        data = cube.data

        super(CubeWrapper, self).__init__(data, data_processors)

    def name(self):
        return self.cube.name()

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

        if (data is not None) and (data.size > 0):

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


class DataArrayWrapper(DataWrapper):

    def __init__(self, dataarray, data_processors=None):

        self.dataarray = dataarray
        data = dataarray.values

        super(DataArrayWrapper, self).__init__(data, data_processors)

    def name(self):
        return self.dataarray.name

    def wrap(self, data, name_prefix='Emulated '):
        """
        Wrap back in a xr.DataArray if one was provided

        :param np.array data: Model output to wrap
        :param str name_prefix:
        :return:
        """
        from xarray import DataArray

        if isinstance(data, tf.Tensor):
            data = data.numpy()

        if (data is not None) and (data.size > 0):

            # Ensure we have a leading sample dimension
            data = data.reshape((-1,) + self.dataarray.shape[1:])

            # Get the coordinates associated with all dimensions except the first one
            coords = {c: self.dataarray.coords[c].values for c in self.dataarray.dims[1:]}
            # Create a new sample dimension coordinate
            coords['sample'] = np.arange(data.shape[0])
            dims = ['sample'] + list(self.dataarray.dims[1:])
            out = DataArray(data, coords=coords, dims=dims, name=name_prefix + self.name(),
                            attrs=self.dataarray.attrs)

        else:
            out = data
        return out


def wrap_data(data, data_processors=None):
    """
    Utility function for wrapping different data types in their appropriate DataWrapper class.
    This allows for easy handling of different input data-types which can then return the same rich types
    for emulation results.

    Parameters
    ----------
    data: xarray.DataArray or iris.Cube or array-like
        The input data to wrap
    data_processors: list of ProcessWrapper
        Any ProcessWrapper data processors to apply to the data

    Returns
    -------
    wrapped_data: DataWrapper
        The data wrapped in a DataWrapper class

    """
    # Optional Iris import
    try:
        from iris.cube import Cube
    except ImportError:
        iris_installed = False
    else:
        iris_installed = True
    # Optional xarray import
    try:
        from xarray import DataArray
    except ImportError:
        xarray_installed = False
    else:
        xarray_installed = True

    if isinstance(data, DataWrapper):
        wrapped_data = data
    elif iris_installed and isinstance(data, Cube):
        wrapped_data = CubeWrapper(data, data_processors=data_processors)
    elif xarray_installed and isinstance(data, DataArray):
        wrapped_data = DataArrayWrapper(data, data_processors=data_processors)
    elif isinstance(data, np.ndarray):
        wrapped_data = DataWrapper(data, data_processors=data_processors)
    else:
        raise ValueError("Training data must be a Cube, DataArray, numpy array or DataWrapper instance")

    return wrapped_data
