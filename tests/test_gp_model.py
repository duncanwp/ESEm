import unittest
from GCEm.gp_model import GPModel
from tests.mock import *
from numpy.testing import assert_allclose


class GPTest(unittest.TestCase):
    def test_sample_1d_simple(self):
        params, test = pop_param(get_params(2), 10)

        ensemble = get_1d_two_param_cube(params)
        expected = eval_1d_cube(test[0])[0]

        m = GPModel(ensemble)
        m.train(params)
        pred_m, pred_var = m.predict(test)

        assert_allclose(expected.data, pred_m, rtol=1e-4)

        # Assert that the mean is within the variance
        # TODO: I'm not sure exactly how to test this...
        # assert_allclose((expected.data-pred_m), pred_var, rtol=1e-4)

    def test_sample_2d_simple(self):
        params, test = pop_param(get_params(3), 10)

        ensemble = get_2d_three_param_cube(params)
        expected = eval_2d_cube(test[0])[0]

        m = GPModel(ensemble)
        m.train(params)
        pred_m, pred_var = m.predict(test)

        assert_allclose(expected.data, pred_m, rtol=1e-4)

    def test_constrain(self):
        from GCEm.gp_model import constrain
        params, test = pop_param(get_params(2), 10)

        ensemble = get_1d_two_param_cube(params)
        expected = eval_1d_cube(test[0])[0]

        # TODO: It might be good to cache this somehow for testing?
        m = GPModel(ensemble)
        m.train(params)

        sample_params = get_params(2, 10)
        _, _, _, _, _, all_valid_samples_arr = constrain(m, expected, sample_params)

        assert_allclose(sample_params[all_valid_samples_arr].mean(), test[0])


if __name__ == '__main__':
    unittest.main()
