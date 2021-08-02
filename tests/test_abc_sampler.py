from esem import gp_model
from esem.utils import get_uniform_params
from esem.abc_sampler import ABCSampler, constrain, _calc_implausibility
from tests.mock import *
from numpy.testing import assert_allclose, assert_array_equal
import pytest


class TestABCSampler:

    def setup_method(self):
        self.training_params = get_uniform_params(2)
        self.training_ensemble = get_1d_two_param_cube(self.training_params)

        self.m = gp_model(self.training_params, self.training_ensemble)
        self.m.train()

    def test_implausibility_scalar_uncertainty(self):
        # Test the implausibility is correct

        obs_uncertainty = 5.
        # Perturbing the obs by one sd should lead to an implausibility of 1.
        obs = self.training_ensemble[10].copy() + obs_uncertainty

        sampler = ABCSampler(self.m, obs,
                             obs_uncertainty=obs_uncertainty/obs.data.mean(),
                             interann_uncertainty=0.,
                             repres_uncertainty=0.,
                             struct_uncertainty=0.)

        # Calculate the implausbility of the training points from a perturbed
        #  training point. The emulator variance should be zero making testing
        #  easier.
        implausibility = sampler.get_implausibility(self.training_params)

        # The implausibility for the 10th sample (the one we perturbed around)
        #  should be one - on average
        assert_allclose(implausibility.data[10, :].mean(), 1., rtol=1e-2)

    def test_implausibility_interann(self):
        # Test the implausibility is correct

        obs_uncertainty = 5.
        # Perturbing the obs by one sd should lead to an implausibility of 1.
        obs = self.training_ensemble[10].copy() + obs_uncertainty

        sampler = ABCSampler(self.m, obs,
                             obs_uncertainty=0.,
                             interann_uncertainty=obs_uncertainty/obs.data.mean(),
                             repres_uncertainty=0.,
                             struct_uncertainty=0.)

        # Calculate the implausbility of the training points from a perturbed
        #  training point. The emulator variance should be zero making testing
        #  easier.
        implausibility = sampler.get_implausibility(self.training_params)

        # The implausibility for the 10th sample (the one we perturbed around)
        #  should be one - on average
        assert_allclose(implausibility.data[10, :].mean(), 1., rtol=1e-2)

    def test_implausibility_repres(self):
        # Test the implausibility is correct
        obs_uncertainty = 5.
        # Perturbing the obs by one sd should lead to an implausibility of 1.
        obs = self.training_ensemble[10].copy() + obs_uncertainty

        sampler = ABCSampler(self.m, obs,
                             obs_uncertainty=0.,
                             interann_uncertainty=0.,
                             repres_uncertainty=obs_uncertainty/obs.data.mean(),
                             struct_uncertainty=0.)

        # Calculate the implausbility of the training points from a perturbed
        #  training point. The emulator variance should be zero making testing
        #  easier.
        implausibility = sampler.get_implausibility(self.training_params)

        # The implausibility for the 10th sample (the one we perturbed around)
        #  should be one - on average
        assert_allclose(implausibility.data[10, :].mean(), 1., rtol=1e-2)

    def test_implausibility_struct(self):
        # Test the implausibility is correct

        obs_uncertainty = 5.
        # Perturbing the obs by one sd should lead to an implausibility of 1.
        obs = self.training_ensemble[10].copy() + obs_uncertainty

        sampler = ABCSampler(self.m, obs,
                             obs_uncertainty=0.,
                             interann_uncertainty=0.,
                             repres_uncertainty=0.,
                             struct_uncertainty=obs_uncertainty/obs.data.mean())

        # Calculate the implausbility of the training points from a perturbed
        #  training point. The emulator variance should be zero making testing
        #  easier.
        implausibility = sampler.get_implausibility(self.training_params)

        # The implausibility for the 10th sample (the one we perturbed around)
        #  should be one - on average
        assert_allclose(implausibility.data[10, :].mean(), 1., rtol=1e-2)

    def test_implausibility_vector_uncertainty(self):
        # Test with a vector obs uncertainty
        obs_uncertainty = self.training_ensemble.data.std(axis=0)

        # Perturbing the obs by one sd should lead to an implausibility of 1.
        obs = self.training_ensemble[10].copy() + obs_uncertainty

        sampler = ABCSampler(self.m, obs,
                             obs_uncertainty=obs_uncertainty/obs.data,
                             interann_uncertainty=0.,
                             repres_uncertainty=0.,
                             struct_uncertainty=0.)

        # Calculate the implausbility of the training points from a perturbed
        #  training point. The emulator variance should be zero making testing
        #  easier.
        implausibility = sampler.get_implausibility(self.training_params)

        # The implausibility for the 10th sample (the one we perturbed around)
        #  should be one.
        expected = np.ones((100,))
        # The first element has zero variance so nan implausibility
        expected[0] = np.nan
        assert_allclose(implausibility.data[10, :], expected, rtol=1e-1)

    def test_absolute_obs_uncertainty(self):
        # Test with a vector obs uncertainty
        obs_uncertainty = self.training_ensemble.data.std(axis=0)

        # Perturbing the obs by one sd should lead to an implausibility of 1.
        obs = self.training_ensemble[10].copy() + obs_uncertainty

        sampler = ABCSampler(self.m, obs,
                             abs_obs_uncertainty=obs_uncertainty,
                             interann_uncertainty=0.,
                             repres_uncertainty=0.,
                             struct_uncertainty=0.)

        # Calculate the implausbility of the training points from a perturbed
        #  training point. The emulator variance should be zero making testing
        #  easier.
        implausibility = sampler.get_implausibility(self.training_params)

        # The implausibility for the 10th sample (the one we perturbed around)
        #  should be one.
        expected = np.ones((100,))
        # The first element has zero variance
        expected[0] = 0.
        assert_allclose(implausibility.data[10, :], expected, rtol=1e-1)

    def test_absolute_interann_uncertainty(self):
        # Test with a vector obs uncertainty
        obs_uncertainty = self.training_ensemble.data.std(axis=0)

        # Perturbing the obs by one sd should lead to an implausibility of 1.
        obs = self.training_ensemble[10].copy() + obs_uncertainty

        sampler = ABCSampler(self.m, obs,
                             abs_obs_uncertainty=0.,
                             abs_interann_uncertainty=obs_uncertainty,
                             repres_uncertainty=0.,
                             struct_uncertainty=0.)

        # Calculate the implausbility of the training points from a perturbed
        #  training point. The emulator variance should be zero making testing
        #  easier.
        implausibility = sampler.get_implausibility(self.training_params)

        # The implausibility for the 10th sample (the one we perturbed around)
        #  should be one.
        expected = np.ones((100,))
        # The first element has zero variance
        expected[0] = 0.
        assert_allclose(implausibility.data[10, :], expected, rtol=1e-1)

    def test_absolute_repress_uncertainty(self):
        # Test with a vector obs uncertainty
        obs_uncertainty = self.training_ensemble.data.std(axis=0)

        # Perturbing the obs by one sd should lead to an implausibility of 1.
        obs = self.training_ensemble[10].copy() + obs_uncertainty

        sampler = ABCSampler(self.m, obs,
                             abs_obs_uncertainty=0.,
                             interann_uncertainty=0.,
                             abs_repres_uncertainty=obs_uncertainty,
                             struct_uncertainty=0.)

        # Calculate the implausbility of the training points from a perturbed
        #  training point. The emulator variance should be zero making testing
        #  easier.
        implausibility = sampler.get_implausibility(self.training_params)

        # The implausibility for the 10th sample (the one we perturbed around)
        #  should be one.
        expected = np.ones((100,))
        # The first element has zero variance
        expected[0] = 0.
        assert_allclose(implausibility.data[10, :], expected, rtol=1e-1)

    def test_absolute_struct_uncertainty(self):
        # Test with a vector obs uncertainty
        obs_uncertainty = self.training_ensemble.data.std(axis=0)

        # Perturbing the obs by one sd should lead to an implausibility of 1.
        obs = self.training_ensemble[10].copy() + obs_uncertainty

        sampler = ABCSampler(self.m, obs,
                             abs_obs_uncertainty=0.,
                             interann_uncertainty=0.,
                             repres_uncertainty=0.,
                             abs_struct_uncertainty=obs_uncertainty)

        # Calculate the implausbility of the training points from a perturbed
        #  training point. The emulator variance should be zero making testing
        #  easier.
        implausibility = sampler.get_implausibility(self.training_params)

        # The implausibility for the 10th sample (the one we perturbed around)
        #  should be one.
        expected = np.ones((100,))
        # The first element has zero variance
        expected[0] = 0.
        assert_allclose(implausibility.data[10, :], expected, rtol=1e-1)

    def test_obs_uncertainty_specified_twice(self):
        # Test with a vector obs uncertainty
        obs_uncertainty = self.training_ensemble.data.std(axis=0)

        # Perturbing the obs by one sd should lead to an implausibility of 1.
        obs = self.training_ensemble[10].copy() + obs_uncertainty

        with pytest.raises(ValueError):
            sampler = ABCSampler(self.m, obs,
                                 obs_uncertainty=obs_uncertainty,
                                 abs_obs_uncertainty=obs_uncertainty)

    def test_interann_uncertainty_specified_twice(self):
        # Test with a vector obs uncertainty
        obs_uncertainty = self.training_ensemble.data.std(axis=0)

        # Perturbing the obs by one sd should lead to an implausibility of 1.
        obs = self.training_ensemble[10].copy() + obs_uncertainty

        with pytest.raises(ValueError):
            sampler = ABCSampler(self.m, obs,
                                 interann_uncertainty=obs_uncertainty,
                                 abs_interann_uncertainty=obs_uncertainty)

    def test_repres_uncertainty_specified_twice(self):
        # Test with a vector obs uncertainty
        obs_uncertainty = self.training_ensemble.data.std(axis=0)

        # Perturbing the obs by one sd should lead to an implausibility of 1.
        obs = self.training_ensemble[10].copy() + obs_uncertainty

        with pytest.raises(ValueError):
            sampler = ABCSampler(self.m, obs,
                                 repres_uncertainty=obs_uncertainty,
                                 abs_repres_uncertainty=obs_uncertainty)

    def test_struct_uncertainty_specified_twice(self):
        # Test with a vector obs uncertainty
        obs_uncertainty = self.training_ensemble.data.std(axis=0)

        # Perturbing the obs by one sd should lead to an implausibility of 1.
        obs = self.training_ensemble[10].copy() + obs_uncertainty

        with pytest.raises(ValueError):
            sampler = ABCSampler(self.m, obs,
                                 struct_uncertainty=obs_uncertainty,
                                 abs_struct_uncertainty=obs_uncertainty)

    def test_calc_implausibility(self):
        # Test the implausibility is correct

        # Test a bunch of simple cases
        imp = _calc_implausibility(np.asarray([1., 1., 2., 1., -2.]),  # Emulator Mean
                                   np.asarray([1., 1., 1., 2., 1.]),  # Obs
                                   np.asarray([1., 2., 1., 1., 1.]),  # Tot Std
                                   )
        assert_allclose(imp, np.asarray([0., 0., 1., 1., 3.]))

        # Test single value inputs
        imp = _calc_implausibility(np.asarray([1., ]),  # Emulator Mean
                                   np.asarray([1., ]),  # Obs
                                   np.asarray([1., ]),  # Tot var
                                   )
        assert_allclose(imp, np.asarray([0.]))

        # Test invalid inputs
        imp = _calc_implausibility(np.asarray([1., ]),  # Emulator Mean
                                   np.asarray([1., ]),  # Obs
                                   np.asarray([0., ]),  # Tot var
                                   )
        assert_allclose(imp, np.asarray([np.nan]))

    def test_constrain(self):
        # Test that constrain returns the correct boolean array for the given implausibility and params

        implausibility = np.asarray([[0., 0., 0., 0., 0.],
                                     [0., 1., 1., 1., 0.],
                                     [0., 0., 1., np.NaN, 0.]])
        # Note that the NaN should be ignored for the following calculations

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

    def test_constrain_2d(self):
        # Test that constrain returns the correct boolean array for 2d obs

        implausibility = np.asarray([[[0., 0., 0., 0., 0.],
                                     [0., 1., 1., 1., 0.],
                                     [0., 0., 1., np.NaN, 0.]],
                                    [[0., 2., 0., 0., 0.],
                                     [0., 2., 1., 1., 0.],
                                     [0., 0., 1., 0., 0.]]]
                                    )

        assert_array_equal(constrain(implausibility, tolerance=0., threshold=3.0),
                           np.asarray([True, True]))
        assert_array_equal(constrain(implausibility, tolerance=0., threshold=0.5),
                           np.asarray([False, False]))
        assert_array_equal(constrain(implausibility, tolerance=0., threshold=1.0),
                           np.asarray([True, False]))

        assert_array_equal(constrain(implausibility, tolerance=3./15., threshold=0.5),
                           np.asarray([False, False]))
        assert_array_equal(constrain(implausibility, tolerance=4./15., threshold=0.5),
                           np.asarray([True, False]))
        assert_array_equal(constrain(implausibility, tolerance=5./15., threshold=0.5),
                           np.asarray([True, True]))

    def test_batch_constrain(self):
        # Test that batch constrain returns the correct boolean array for
        #  the given model, obs and params
        obs_uncertainty = self.training_ensemble.data.std(axis=0)

        # Perturbing the obs by one sd should lead to an implausibility of 1.
        obs = self.training_ensemble[10].copy() + obs_uncertainty

        sampler = ABCSampler(self.m, obs,
                             obs_uncertainty=obs_uncertainty/obs.data,
                             interann_uncertainty=0.,
                             repres_uncertainty=0.,
                             struct_uncertainty=0.)

        # Calculate the implausbility of the training points from a perturbed
        #  training point. The emulator variance should be zero making testing
        #  easier.
        valid_samples = sampler.batch_constrain(self.training_params,
                                                tolerance=0., threshold=2.)

        # The implausibility for the 10th sample (the one we perturbed around)
        #  should be around one (and hence valid), some neighbouring points are
        #  also valid, the rest should be invalid
        expected = np.asarray([True, False, False, True, False,
                               True, False, False, True, False,
                               True, False, False, True, True,
                               True, True, True, True, True,
                               False, False, False, False, False])

        assert_array_equal(valid_samples, expected)

    def test_sample(self):
        # Test that batch constrain returns the correct boolean array for
        #  the given model, obs and params
        obs_uncertainty = self.training_ensemble.data.std(axis=0)

        # Perturbing the obs by one sd should lead to an implausibility of 1.
        obs = self.training_ensemble[10].copy() + obs_uncertainty

        sampler = ABCSampler(self.m, obs,
                             obs_uncertainty=obs_uncertainty/obs.data,
                             interann_uncertainty=0.,
                             repres_uncertainty=0.,
                             struct_uncertainty=0.)

        # Generate only valid samples
        valid_samples = sampler.sample(n_samples=100, tolerance=0., threshold=2.)

        assert valid_samples.shape == (100, 2)

        # Constrain them all - they should all be valid
        are_valid = sampler.batch_constrain(valid_samples,
                                            tolerance=0., threshold=2.)

        assert are_valid.all()


class TestABCSampler2D:

    def setup_method(self):
        self.training_params = get_uniform_params(3)
        self.training_ensemble = get_three_param_cube(self.training_params)

        self.m = gp_model(self.training_params, self.training_ensemble)
        self.m.train()

    def test_implausibility_scalar_uncertainty_interface(self):
        # Test the interface correctly deals with 2d model and obs

        obs_uncertainty = 5.
        # Perturbing the obs by one sd should lead to an implausibility of 1.
        obs = self.training_ensemble[10].copy() + obs_uncertainty

        sampler = ABCSampler(self.m, obs,
                             obs_uncertainty=obs_uncertainty/obs.data.mean(),
                             interann_uncertainty=0.,
                             repres_uncertainty=0.,
                             struct_uncertainty=0.)

        implausibility = sampler.get_implausibility(self.training_params)

        expected_shape = (len(self.training_params), ) + obs.shape
        assert implausibility.shape == expected_shape

    def test_implausibility_vector_uncertainty(self):
        # Test with a vector obs uncertainty
        obs_uncertainty = self.training_ensemble.data.std(axis=0)

        # Perturbing the obs by one sd should lead to an implausibility of 1.
        obs = self.training_ensemble[10].copy() + obs_uncertainty

        sampler = ABCSampler(self.m, obs,
                             obs_uncertainty=obs_uncertainty/obs.data,
                             interann_uncertainty=0.,
                             repres_uncertainty=0.,
                             struct_uncertainty=0.)

        # Calculate the implausbility of the training points from a perturbed
        #  training point. The emulator variance should be zero making testing
        #  easier.
        implausibility = sampler.get_implausibility(self.training_params)

        expected_shape = (len(self.training_params),) + obs.shape
        assert implausibility.shape == expected_shape

    def test_batch_constrain(self):
        # Test that batch constrain returns the correct boolean array for
        #  the given model, obs and params
        obs_uncertainty = self.training_ensemble.data.std(axis=0)

        # Perturbing the obs by one sd should lead to an implausibility of 1.
        obs = self.training_ensemble[10].copy() + obs_uncertainty

        sampler = ABCSampler(self.m, obs,
                             obs_uncertainty=obs_uncertainty/obs.data,
                             interann_uncertainty=0.,
                             repres_uncertainty=0.,
                             struct_uncertainty=0.)

        # Calculate the implausbility of the training points from a perturbed
        #  training point. The emulator variance should be zero making testing
        #  easier.
        valid_samples = sampler.batch_constrain(self.training_params,
                                                tolerance=0., threshold=2.)

        expected_shape = (len(self.training_params),)
        assert valid_samples.shape == expected_shape

    def test_sample(self):
        # Test that batch constrain returns the correct boolean array for
        #  the given model, obs and params
        obs_uncertainty = self.training_ensemble.data.std(axis=0)

        # Perturbing the obs by one sd should lead to an implausibility of 1.
        obs = self.training_ensemble[10].copy() + obs_uncertainty

        sampler = ABCSampler(self.m, obs,
                             obs_uncertainty=obs_uncertainty/obs.data,
                             interann_uncertainty=0.,
                             repres_uncertainty=0.,
                             struct_uncertainty=0.)

        # Generate only valid samples
        valid_samples = sampler.sample(n_samples=100, tolerance=0., threshold=2.)

        assert valid_samples.shape == (100, 3)
