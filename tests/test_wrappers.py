import numpy as np
from numpy.testing import assert_array_equal
from esem.utils import get_uniform_params
from tests.mock import get_1d_two_param_cube
from esem.wrappers import CubeWrapper, DataArrayWrapper, DataWrapper, wrap_data


def test_wrap_cube():
    params = get_uniform_params(2, 6)
    ensemble = get_1d_two_param_cube(params)

    wrapped_cube = CubeWrapper(ensemble)
    result = wrapped_cube.wrap(ensemble.data)
    assert_array_equal(ensemble.data, result.data)
    assert ensemble.units == result.units
    assert "Emulated unknown" == result.name()


def test_wrap_cube_aux_sample_dim():
    from iris.util import demote_dim_coord_to_aux_coord
    # If you slice a dim coord in the middle somewhere so it becomes non-monotonic
    #  then it gets turned in to an AuxCoord. (It might also just be an AuxCoord sometimes)

    params = get_uniform_params(2, 6)
    ensemble = get_1d_two_param_cube(params)
    # Demote the job dim to an AuxCoord
    demote_dim_coord_to_aux_coord(ensemble, 'job')

    wrapped_cube = CubeWrapper(ensemble)
    result = wrapped_cube.wrap(ensemble.data)
    assert_array_equal(ensemble.data, result.data)
    assert ensemble.units == result.units
    assert "Emulated unknown" == result.name()


def test_wrap_dataarray():
    import xarray as xr
    params = get_uniform_params(2, 6)
    ensemble = xr.DataArray.from_iris(get_1d_two_param_cube(params))
    # Set sensible defaults
    ensemble.attrs['units'] = 'kg s-2'
    ensemble.name = 'unknown'

    wrapped_array = DataArrayWrapper(ensemble)
    result = wrapped_array.wrap(ensemble.data)
    assert_array_equal(ensemble.data, result.data)
    assert ensemble.attrs['units'] == result.attrs['units']
    assert "Emulated unknown" == result.name


def test_wrap_numpy():
    params = get_uniform_params(2, 6)
    ensemble = get_1d_two_param_cube(params).data

    wrapped_array = DataWrapper(ensemble)
    result = wrapped_array.wrap(ensemble)
    assert isinstance(result, np.ndarray)
    assert_array_equal(ensemble, result)


def test_wrap_data():
    from unittest.mock import patch
    import sys
    import xarray as xr
    params = get_uniform_params(2, 6)

    iris_data = get_1d_two_param_cube(params)
    numpy_data = iris_data.data
    xarray_data = xr.DataArray.from_iris(iris_data)

    with patch.dict(sys.modules, {'iris': None}):
        # Check we can still wrap numpy and xarray without iris
        assert isinstance(wrap_data(numpy_data), DataWrapper)
        assert isinstance(wrap_data(xarray_data), DataArrayWrapper)
    with patch.dict(sys.modules, {'xarray': None}):
        # Check we can still wrap numpy and iris without xarray
        assert isinstance(wrap_data(numpy_data), DataWrapper)
        assert isinstance(wrap_data(iris_data), CubeWrapper)
        with patch.dict(sys.modules, {'iris': None}):
            # Check we can still wrap numpy without either
            assert isinstance(wrap_data(numpy_data), DataWrapper)
