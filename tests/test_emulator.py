from esem.emulator import Emulator
from esem.utils import get_uniform_params
from esem.cube_wrapper import CubeWrapper
from tests.mock import get_mock_model, get_1d_two_param_cube
from numpy.testing import assert_array_equal
from contextlib import nullcontext
import pandas as pd
import pytest


@pytest.mark.parametrize(
    "training_data,expectation",
    [
        (get_1d_two_param_cube(get_uniform_params(3)), nullcontext()),
        (get_1d_two_param_cube(get_uniform_params(3)).data, nullcontext()),
        (CubeWrapper(get_1d_two_param_cube(get_uniform_params(3))), nullcontext()),
        ([100., 50.], pytest.raises(ValueError)),
    ],
)
def test_construct_emulator_with_different_training_data(training_data, expectation):
    # Test the interface correctly deals with a cube training data

    with expectation:
        emulator = Emulator(get_mock_model(), get_uniform_params(3), training_data)
        assert_array_equal(emulator.training_data.data, training_data.data)


@pytest.mark.parametrize(
    "training_params,expectation",
    [
        (get_uniform_params(3), nullcontext()),
        (pd.DataFrame(get_uniform_params(3)), nullcontext()),
        ([100., 50.], pytest.raises(ValueError)),
    ],
)
def test_construct_emulator_with_different_training_params(training_params, expectation):
    # Test the interface correctly deals with a cube training data

    with expectation:
        emulator = Emulator(get_mock_model(), training_params, get_1d_two_param_cube(get_uniform_params(3)))
        assert_array_equal(emulator.training_params, training_params)


def test_emulator_prediction_cube():
    # Test the interface correctly deals with a cube training data

    params = get_uniform_params(3)
    training_ensemble = get_1d_two_param_cube(params)
    m = get_mock_model()

    emulator = Emulator(m, params, training_ensemble)

    pred_mean, pred_var = emulator.predict()

    assert pred_mean.name() == 'Emulated unknown'
    assert pred_var.name() == 'Variance in emulated unknown'
    assert pred_mean.units == training_ensemble.units
    assert_array_equal(pred_mean.data, m.predict()[0])

