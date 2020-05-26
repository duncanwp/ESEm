import unittest
from GCEm.gp_model import GPModel
from GCEm.utils import get_uniform_params
from tests.mock import *
from numpy.testing import assert_allclose


class GPTest(object):
    """
    Tests on the GPModel class and its methods. The actual model is setup
     independently in the concrete test classes below. This abstracts the
     different test cases out and allows the model to only be created once
     for each test case.
    """

    def test_simple_predict(self):

        # Get the actual test data
        #  Use the class method `eval_fn` so 'self' doesn't get passed
        expected = type(self).eval_fn(self.test_params[0])

        pred_m, pred_var = self.model._tf_predict(self.test_params[0:1])

        assert_allclose(expected.data, pred_m.numpy(), rtol=1e-3)

        # Assert that the mean is within the variance
        # TODO: I'm not sure exactly how to test this...
        # assert_allclose((expected.data-pred_m), pred_var, rtol=1e-4)

    def test_predict_interface(self):

        # Get the actual test data
        #  Use the class method `eval_fn` so 'self' doesn't get passed
        expected = type(self).eval_fn(self.test_params[0])

        pred_m, pred_var = self.model.predict(self.test_params[0:1])

        assert_allclose(expected.data, pred_m.data, rtol=1e-3)
        assert pred_m.name() == 'Emulated ' + expected.name()
        assert pred_var.name() == 'Variance in emulated ' + expected.name()
        assert pred_m.units == expected.units
        assert pred_var.units == expected.units

    def test_predict_interface_multiple_samples(self):
        from iris.cube import CubeList
        # Get the actual test data
        #  Use the class method `eval_fn` so 'self' doesn't get passed
        expected = CubeList([type(self).eval_fn(p, job_n=i) for i, p in enumerate(self.test_params)])
        expected = expected.concatenate_cube()

        pred_m, pred_var = self.model.predict(self.test_params)

        assert_allclose(expected.data, pred_m.data, rtol=1e-3)
        assert pred_m.name() == 'Emulated ' + (expected.name() or 'data')
        assert pred_var.name() == 'Variance in emulated ' + (expected.name() or 'data')
        assert pred_m.units == expected.units
        assert pred_var.units == expected.units

    def test_batch_stats(self):
        from iris.cube import CubeList
        from GCEm.utils import get_random_params
        # Test that the sample_mean function returns the mean of the sample

        sample_params = get_random_params(self.params.shape[1], 10)

        expected = CubeList([type(self).eval_fn(p, job_n=i) for i, p in enumerate(sample_params)])
        expected_ensemble = expected.concatenate_cube()

        mean, std_dev = self.model.batch_stats(sample_params)

        assert_allclose(mean.data, expected_ensemble.data.mean(axis=0), rtol=1e-1)
        # This is a really loose test but it needs to be because of the
        #  stochastic nature of the model and the ensemble points
        assert_allclose(std_dev.data, expected_ensemble.data.std(axis=0), rtol=.5)


class Simple1DTest(unittest.TestCase, GPTest):
    """
    Setup for the simple 1D 2 parameter test case
    """

    @classmethod
    def setUpClass(cls) -> None:
        params, test = pop_elements(get_uniform_params(2), 10, 12)

        ensemble = get_1d_two_param_cube(params)
        m = GPModel(params, ensemble)
        m.train()

        cls.model = m
        cls.params = params
        cls.test_params = test
        cls.eval_fn = eval_1d_cube


class Simple2DTest(unittest.TestCase, GPTest):
    """
    Setup for the simple 2D 3 parameter test case.
    """

    @classmethod
    def setUpClass(cls) -> None:
        params, test = pop_elements(get_uniform_params(3), 50)

        ensemble = get_three_param_cube(params)
        m = GPModel(params, ensemble)
        m.train()

        cls.model = m
        cls.params = params
        cls.test_params = test
        cls.eval_fn = eval_cube


class Simple32bitTest(unittest.TestCase, GPTest):
    """
    Setup for the simple 2D 3 parameter test case with 32bit data
    """

    @classmethod
    def setUpClass(cls) -> None:
        params, test = pop_elements(get_uniform_params(3), 50)

        ensemble = get_three_param_cube(params)
        # Create a new, ensemble at lower precision
        ensemble = ensemble.copy(data=ensemble.data.astype('float32'))
        m = GPModel(params, ensemble)
        m.train()

        cls.model = m
        cls.params = params
        cls.test_params = test
        cls.eval_fn = eval_cube


if __name__ == '__main__':
    unittest.main()
