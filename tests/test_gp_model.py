import unittest
from GCEm.gp_model import GPModel
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
        expected = type(self).eval_fn(self.test_params[0])[0]

        pred_m, pred_var = self.model.predict(self.test_params)

        assert_allclose(expected.data, pred_m, rtol=1e-3)

        # Assert that the mean is within the variance
        # TODO: I'm not sure exactly how to test this...
        # assert_allclose((expected.data-pred_m), pred_var, rtol=1e-4)


class Simple1DTest(unittest.TestCase, GPTest):
    """
    Setup for the simple 1D 2 parameter test case
    """

    @classmethod
    def setUpClass(cls) -> None:
        params, test = pop_param(get_uniform_params(2), 10)

        ensemble = get_1d_two_param_cube(params)
        m = GPModel(ensemble)
        m.train(params)

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
        params, test = pop_param(get_uniform_params(3), 50)

        ensemble = get_2d_three_param_cube(params)
        m = GPModel(ensemble)
        m.train(params)

        cls.model = m
        cls.params = params
        cls.test_params = test
        cls.eval_fn = eval_2d_cube


class SampleTest(unittest.TestCase):

    def test_sample_mean(self):
        # Test that the sample_mean function returns the mean of the sample
        from GCEm.gp_model import sample_mean

        training_params = get_uniform_params(2)

        training_ensemble = get_1d_two_param_cube(training_params)

        m = GPModel(training_ensemble)
        m.train(training_params, verbose=True)

        sample_params = get_random_params(2)
        expected_ensemble = get_1d_two_param_cube(sample_params)

        mean, std_dev = sample_mean(m, sample_params)

        assert_allclose(mean, expected_ensemble.data.mean(axis=0), rtol=1e-1)
        assert_allclose(std_dev, expected_ensemble.data.std(axis=0), rtol=1e-1)

    def test_implausibility(self):
        # Test the implausibility is correct
        from GCEm.gp_model import get_implausibility

    def test_constrain(self):
        # Test that constrain returns the correct boolean array for the given implausibiltiy and params
        pass


if __name__ == '__main__':
    unittest.main()
