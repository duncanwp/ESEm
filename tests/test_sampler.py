import unittest
from GCEm.gp_model import GPModel
from tests.mock import *
from numpy.testing import assert_allclose, assert_array_equal


class SampleTest(unittest.TestCase):

    def setUp(self) -> None:
        self.training_params = get_uniform_params(2)
        self.training_ensemble = get_1d_two_param_cube(self.training_params)

        self.m = GPModel(self.training_ensemble)
        self.m.train(self.training_params)

    def test_batch_stats(self):
        # Test that the sample_mean function returns the mean of the sample
        from GCEm.sampler import batch_stats

        sample_params = get_random_params(2)
        expected_ensemble = get_1d_two_param_cube(sample_params)

        mean, std_dev = batch_stats(self.m, sample_params)

        assert_allclose(mean.data, expected_ensemble.data.mean(axis=0), rtol=1e-1)
        # This is a really loose test but it needs to be because of the
        #  stochastic nature of the model and the ensemble points
        assert_allclose(std_dev.data, expected_ensemble.data.std(axis=0), rtol=.5)

    def test_implausibility_scalar_uncertainty(self):
        # Test the implausibility is correct
        from GCEm.sampler import get_implausibility

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
        assert_allclose(implausibility.data[10, :].mean(), 1., rtol=1e-2)

    def test_implausibility_interann(self):
        # Test the implausibility is correct
        from GCEm.sampler import get_implausibility

        obs_uncertainty = 5.
        # Perturbing the obs by one sd should lead to an implausibility of 1.
        obs = self.training_ensemble[10].copy() + obs_uncertainty

        # Calculate the implausbility of the training points from a perturbed
        #  training point. The emulator variance should be zero making testing
        #  easier.
        implausibility = get_implausibility(self.m, obs, self.training_params,
                                            obs_uncertainty=0.,
                                            interann_uncertainty=obs_uncertainty/obs.data.mean(),
                                            repres_uncertainty=0.,
                                            struct_uncertainty=0.)

        # The implausibility for the 10th sample (the one we perturbed around)
        #  should be one - on average
        assert_allclose(implausibility.data[10, :].mean(), 1., rtol=1e-2)

    def test_implausibility_repres(self):
        # Test the implausibility is correct
        from GCEm.sampler import get_implausibility

        obs_uncertainty = 5.
        # Perturbing the obs by one sd should lead to an implausibility of 1.
        obs = self.training_ensemble[10].copy() + obs_uncertainty

        # Calculate the implausbility of the training points from a perturbed
        #  training point. The emulator variance should be zero making testing
        #  easier.
        implausibility = get_implausibility(self.m, obs, self.training_params,
                                            obs_uncertainty=0.,
                                            interann_uncertainty=0.,
                                            repres_uncertainty=obs_uncertainty/obs.data.mean(),
                                            struct_uncertainty=0.)

        # The implausibility for the 10th sample (the one we perturbed around)
        #  should be one - on average
        assert_allclose(implausibility.data[10, :].mean(), 1., rtol=1e-2)

    def test_implausibility_struct(self):
        # Test the implausibility is correct
        from GCEm.sampler import get_implausibility

        obs_uncertainty = 5.
        # Perturbing the obs by one sd should lead to an implausibility of 1.
        obs = self.training_ensemble[10].copy() + obs_uncertainty

        # Calculate the implausbility of the training points from a perturbed
        #  training point. The emulator variance should be zero making testing
        #  easier.
        implausibility = get_implausibility(self.m, obs, self.training_params,
                                            obs_uncertainty=0.,
                                            interann_uncertainty=0.,
                                            repres_uncertainty=0.,
                                            struct_uncertainty=obs_uncertainty/obs.data.mean())

        # The implausibility for the 10th sample (the one we perturbed around)
        #  should be one - on average
        assert_allclose(implausibility.data[10, :].mean(), 1., rtol=1e-2)

    def test_implausibility_vector_uncertainty(self):
        # Test with a vector obs uncertainty
        from GCEm.sampler import get_implausibility
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
        assert_allclose(implausibility.data[10, :], expected, rtol=1e-1)

    def test_calc_implausibility(self):
        # Test the implausibility is correct
        from GCEm.sampler import _calc_implausibility

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
        # Test that constrain returns the correct boolean array for the given implausibility and params
        from GCEm.sampler import constrain

        implausibility = np.asarray([[0., 0., 0., 0., 0.],
                                     [0., 1., 1., 1., 0.],
                                     [0., 0., 1., 0., 0.]])
        assert_array_equal(constrain(implausibility, tolerance=0., threshold=3.0),
                           np.asarray([True, True, True]))
        assert_array_equal(constrain(implausibility, tolerance=0., threshold=0.5),
                           np.asarray([True, False, False]))
        assert_array_equal(constrain(implausibility, tolerance=0., threshold=1.0),
                           np.asarray([True, True, True]))

        assert_array_equal(constrain(implausibility, tolerance=2./5., threshold=0.5),
                           np.asarray([True, False, True]))
        assert_array_equal(constrain(implausibility, tolerance=1./5., threshold=0.5),
                           np.asarray([True, False, True]))

    def test_batch_constrain(self):
        # Test that batch constrain returns the correct boolean array for
        #  the given model, obs and params
        from GCEm.sampler import batch_constrain
        obs_uncertainty = self.training_ensemble.data.std(axis=0)

        # Perturbing the obs by one sd should lead to an implausibility of 1.
        obs = self.training_ensemble[10].copy() + obs_uncertainty

        # Calculate the implausbility of the training points from a perturbed
        #  training point. The emulator variance should be zero making testing
        #  easier.
        valid_samples = batch_constrain(self.m, obs, self.training_params,
                                        obs_uncertainty=obs_uncertainty/obs.data,
                                        interann_uncertainty=0.,
                                        repres_uncertainty=0.,
                                        struct_uncertainty=0.,
                                        tolerance=0., threshold=2.)

        # The implausibility for the 10th sample (the one we perturbed around)
        #  should be around one (and hence valid), some neighbouring points are
        #  also valid, the rest should be invalid
        expected = np.asarray([True, False, False, True, False,
                               True, False, False, True, False,
                               True, False, False, True, True,
                               True, True, True, True, True,
                               False, False, False, False, False])

        assert_array_equal(valid_samples.numpy(), expected)
