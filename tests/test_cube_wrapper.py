from numpy.testing import assert_array_equal
from esem.utils import get_uniform_params
from tests.mock import get_1d_two_param_cube
from esem.cube_wrapper import CubeWrapper


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
