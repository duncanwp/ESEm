import unittest
from GCEm.emulator import Emulator
from GCEm.utils import get_uniform_params
from tests.mock import get_mock_model, get_1d_two_param_cube
from numpy.testing import assert_allclose, assert_array_equal


class EmulatorTests(unittest.TestCase):

    def setUp(self) -> None:
        self.training_params = get_uniform_params(3)
        self.training_ensemble = get_1d_two_param_cube(self.training_params)

        self.m = get_mock_model()

    def test_construct_emulator_with_cube(self):
        # Test the interface correctly deals with a cube training data

        emulator = Emulator(self.m, self.training_params, self.training_ensemble)

        assert_array_equal(emulator.training_data.data, self.training_ensemble.data)

    def test_construct_emulator_with_numpy(self):
        # Test the interface correctly deals with numpy training data

        emulator = Emulator(self.m, self.training_params, self.training_ensemble.data)

        assert_array_equal(emulator.training_data.data, self.training_ensemble.data)

    def test_construct_emulator_with_CubeWrapper(self):
        from GCEm.cube_wrapper import CubeWrapper
        # Test the interface correctly deals with CubeWrapper training data

        emulator = Emulator(self.m, self.training_params, CubeWrapper(self.training_ensemble))

        assert_array_equal(emulator.training_data.data, self.training_ensemble.data)

    def test_construct_emulator_with_data_type(self):
        # Test the interface correctly deals with invalid training data

        emulator = Emulator(self.m, self.training_params, [100., 10.])

        assert_array_equal(emulator.training_data.data, self.training_ensemble.data)

    def test_construct_emulator_with_numpy_params(self):
        # Test the interface correctly deals with numpy training params

        emulator = Emulator(self.m, self.training_params.numpy(), self.training_ensemble)

        assert_array_equal(emulator.training_data.data, self.training_ensemble.data)

    def test_construct_emulator_with_invalid_params(self):
        # Test the interface correctly deals with invalid training params

        emulator = Emulator(self.m, [100., 10.], self.training_ensemble)

        assert_array_equal(emulator.training_data.data, self.training_ensemble.data)

    def test_emulator_prediction_cube(self):
        # Test the interface correctly deals with a cube training data

        emulator = Emulator(self.m, self.training_params, self.training_ensemble)

        pred_mean, pred_var = emulator.predict()

        self.assert_(pred_mean.name, 'Emulated ')
        self.assert_(pred_mean.name, 'Variance in emulated ')
        self.assert_(pred_mean.units, self.training_ensemble.units)
        assert_array_equal(pred_mean.data, self.m.predict()[0])

