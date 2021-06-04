import numpy as np
from numpy.testing import assert_array_equal
from GCEm.utils import get_uniform_params, get_param_mask
from tests.mock import simple_polynomial_fn_two_param


def test_get_param_mask():
    X = get_uniform_params(n_params=2, n_samples=5)
    y = simple_polynomial_fn_two_param(*X.T)

    # Test that the two valid parameters get picked out
    mask = get_param_mask(X, y)
    assert_array_equal(mask, np.asarray([True, True]))

    # Test that if we add a nonsense parameter this gets rejected
    X2 = np.hstack([X, np.ones((25, 1))])
    mask = get_param_mask(X2, y)
    assert_array_equal(mask, np.asarray([True, True, False]))

    # Test that you can use the mask as in the documentation
    assert_array_equal(X, X2[:, mask])
