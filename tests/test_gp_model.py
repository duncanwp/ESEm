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

    def setUp(self) -> None:
        self.training_params = get_uniform_params(2)
        self.training_ensemble = get_1d_two_param_cube(self.training_params)

        self.m = GPModel(self.training_ensemble)
        self.m.train(self.training_params)

    def test_batch_stats(self):
        # Test that the sample_mean function returns the mean of the sample
        from GCEm.gp_model import batch_stats

        sample_params = get_random_params(2)
        expected_ensemble = get_1d_two_param_cube(sample_params)

        mean, std_dev = batch_stats(self.m, sample_params)

        assert_allclose(mean, expected_ensemble.data.mean(axis=0), rtol=1e-1)
        assert_allclose(std_dev, expected_ensemble.data.std(axis=0), rtol=1e-1)

    def test_implausibility_scalar_uncertainty(self):
        # Test the implausibility is correct
        from GCEm.gp_model import get_implausibility

        obs_uncertainty = 5.
        # Perturbing the obs by one sd should lead to an implausibility of 1.
        obs = self.training_ensemble[10].copy() + obs_uncertainty

        # Calculate the implausbility of the training points from a perturbed
        #  training point. The emulator variance should be zero making testing
        #  easier.
        implausibility = get_implausibility(self.m, obs, self.training_params,
                                            obs_uncertainty=obs_uncertainty/obs.data.mean(),
                                            interann_uncertainty=0.,
                                            repres_uncertainty=0.,
                                            struct_uncertainty=0.)

        # The implausibility for the 10th sample (the one we perturbed around)
        #  should be one - on average
        assert_allclose(implausibility.numpy()[10, :].mean(), 1., rtol=1e-2)

    def test_implausibility_vector_uncertainty(self):
        # Test with a vector obs uncertainty
        from GCEm.gp_model import get_implausibility
        obs_uncertainty = self.training_ensemble.data.std(axis=0)

        # Perturbing the obs by one sd should lead to an implausibility of 1.
        obs = self.training_ensemble[10].copy() + obs_uncertainty

        # Calculate the implausbility of the training points from a perturbed
        #  training point. The emulator variance should be zero making testing
        #  easier.
        implausibility = get_implausibility(self.m, obs, self.training_params,
                                            obs_uncertainty=obs_uncertainty/obs.data,
                                            interann_uncertainty=0.,
                                            repres_uncertainty=0.,
                                            struct_uncertainty=0.)

        # The implausibility for the 10th sample (the one we perturbed around)
        #  should be one.
        expected = np.ones((100,))
        # The first element has zero variance so nan implausibility
        expected[0] = np.nan
        assert_allclose(implausibility.numpy()[10, :], expected, rtol=1e-1)

    def test_calc_implausibility(self):
        # Test the implausibility is correct
        from GCEm.gp_model import _calc_implausibility

        # Test a bunch of simple cases
        imp = _calc_implausibility(np.asarray([1., 1., 2., 1., -2.]),  # Emulator Mean
                                   np.asarray([1., 1., 1., 2., 1.]),  # Obs
                                   np.asarray([1., 2., 1., 1., 1.]),  # Emulator var
                                   np.asarray([1., 1., 1., 1., 1.]),  # Interann var
                                   np.asarray([1., 1., 1., 1., 1.]),  # Obs var
                                   np.asarray([1., 1., 1., 1., 1.]),  # respres var
                                   np.asarray([1., 1., 1., 1., 1.]),  # Struct var
                                   )
        assert_allclose(imp, np.asarray([0., 0., 1./np.sqrt(5.), 1./np.sqrt(5.), 3./np.sqrt(5.)]))

        # Test single value inputs
        imp = _calc_implausibility(np.asarray([1., ]),  # Emulator Mean
                                   np.asarray([1., ]),  # Obs
                                   np.asarray([1., ]),  # Emulator var
                                   np.asarray([1., ]),  # Interann var
                                   np.asarray([1., ]),  # Obs var
                                   np.asarray([1., ]),  # respres var
                                   np.asarray([1., ]),  # Struct var
                                   )
        assert_allclose(imp, np.asarray([0.]))

        # Test invalid inputs
        imp = _calc_implausibility(np.asarray([1., ]),  # Emulator Mean
                                   np.asarray([1., ]),  # Obs
                                   np.asarray([0., ]),  # Emulator var
                                   np.asarray([0., ]),  # Interann var
                                   np.asarray([0., ]),  # Obs var
                                   np.asarray([0., ]),  # respres var
                                   np.asarray([0., ]),  # Struct var
                                   )
        assert_allclose(imp, np.asarray([np.nan]))

    def test_constrain(self):
        # Test that constrain returns the correct boolean array for the given implausibiltiy and params
        from GCEm.gp_model import get_implausibility, constrain

    def test_batch_constrain(self):
        # Test that constrain returns the correct boolean array for the given implausibiltiy and params
        from GCEm.gp_model import get_implausibility, batch_constrain


if __name__ == '__main__':
    unittest.main()
