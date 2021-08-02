from esem import gp_model
from esem.utils import get_uniform_params
from esem.sampler import MCMCSampler, _target_log_likelihood
from tests.mock import *
import pytest
from numpy.testing import assert_allclose
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


def test_calc_likelihood():
    # Test the likelihood is correct
    prior_x = tfd.Uniform(low=tf.zeros(2, dtype=tf.float64),
                          high=tf.ones(2, dtype=tf.float64))

    prior_x = tfd.Independent(prior_x, reinterpreted_batch_ndims=1, name='model')

    # Test prob of x
    imp = _target_log_likelihood(prior_x,
                                 np.asarray([1.]),  # x
                                 np.asarray([0.]),  # diff
                                 np.asarray([1.]),  # Tot Std
                                 )
    # Prob at center of a Normal distribution of sigma=1
    expected = 1. / np.sqrt(2. * np.pi)
    assert_allclose(imp, np.log(np.asarray([expected])))

    imp = _target_log_likelihood(prior_x,
                                 np.asarray([0., 0.]),  # x
                                 np.asarray([0.]),  # diff
                                 np.asarray([1.]),  # Tot Std
                                 )
    assert_allclose(imp, np.log(np.asarray([expected])))

    imp = _target_log_likelihood(prior_x,
                                 np.asarray([2., 1]),  # x
                                 np.asarray([0., 0.]),  # diff
                                 np.asarray([1., 1.]),  # Tot Std
                                 )
    assert_allclose(imp, np.log(np.asarray([0.])))

    # Test a bunch of simple cases
    imp = _target_log_likelihood(prior_x,
                                 np.asarray([0.5, 0.5]),  # x
                                 np.asarray([0.]),  # diff
                                 np.asarray([1.]),  # Tot Std
                                 )
    assert_allclose(imp, np.log(np.asarray([expected])))

    imp = _target_log_likelihood(prior_x,
                                 np.asarray([0.5, 0.5]),  # x
                                 np.asarray([1.]),  # diff
                                 np.asarray([1.]),  # Tot Std
                                 )
    # Prob at 1-sigma of a Normal distribution (of sigma=1)
    expected = np.exp(-0.5) / np.sqrt(2. * np.pi)
    assert_allclose(imp, np.log(np.asarray([expected])))

    imp = _target_log_likelihood(prior_x,
                                 np.asarray([0.5, 0.5]),  # x
                                 np.asarray([1., 1.]),  # diff
                                 np.asarray([1., 1.]),  # Tot Std
                                 )
    assert_allclose(imp, np.log(np.asarray([expected*expected])))


def test_sample():
    training_params = get_uniform_params(2)
    training_ensemble = get_1d_two_param_cube(training_params)

    m = gp_model(training_params, training_ensemble)
    m.train()

    # Test that sample returns the correct shape array for
    #  the given model, obs and params.
    obs_uncertainty = training_ensemble.data.std(axis=0)

    # Perturbing the obs by one sd should lead to an implausibility of 1.
    obs = training_ensemble[10].copy() + obs_uncertainty

    sampler = MCMCSampler(m, obs,
                          obs_uncertainty=obs_uncertainty/obs.data,
                          interann_uncertainty=0.,
                          repres_uncertainty=0.,
                          struct_uncertainty=0.)

    # Generate only valid samples, don't bother with burn-in
    valid_samples = sampler.sample(n_samples=10,
                                   mcmc_kwargs=dict(num_burnin_steps=0))

    # Just check the shape. We test the actual probabilities above
    #  and we don't need to test the tf mcmc code
    assert valid_samples.shape == (10, 2)


@pytest.mark.skip  # This doesn't work now - we just print a loud warning instead.
def test_nan_obs_are_ignored():
    prior_x = tfd.Uniform(low=tf.zeros(2, dtype=tf.float64),
                          high=tf.ones(2, dtype=tf.float64))

    prior_x = tfd.Independent(prior_x, reinterpreted_batch_ndims=1, name='model')

    # If an obs is NaN then the difference (distance) must be NaN too
    likelihood = _target_log_likelihood(prior_x, 0.5, diff=np.asarray([[0.0, np.NaN]]),
                                        total_sd=np.asarray([[1.0, 1.0]]))

    # The likelihood should not be NaN though
    assert not np.isnan(likelihood.numpy())


@pytest.mark.slow
def test_simple_sample():
    from iris.cube import Cube
    X = get_uniform_params(2)
    z = simple_polynomial_fn_two_param(*X.T)

    m = gp_model(X, z)
    m.train()

    sampler = MCMCSampler(m, Cube(np.asarray([2.])),
                          obs_uncertainty=0.1,
                          interann_uncertainty=0.,
                          repres_uncertainty=0.,
                          struct_uncertainty=0.)

    # Use as few burn-in steps as we can get away with to speed up the test
    samples = sampler.sample(n_samples=500,
                             mcmc_kwargs=dict(num_burnin_steps=50))
    Zs = simple_polynomial_fn_two_param(*samples.T)
    assert_allclose(Zs.mean(), 2., rtol=0.1)
