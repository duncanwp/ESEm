import unittest
from GCEm.rf_model import RFModel
from GCEm.utils import get_uniform_params
from tests.mock import *
from numpy.testing import assert_allclose


class RFTest(object):
    """
    Tests on the RFModel class and its methods. The actual model is setup
     independently in the concrete test classes below. This abstracts the
     different test cases out and allows the model to only be created once
     for each test case.

    N.B. Fixing random_seed=0 when initialising model, else no guarantee
    that the tolerance will always pass. (RFs use bootstrapping)
    """

    def test_simple_predict(self):

        # Get the actual test data
        #  Use the class method `eval_fn` so 'self' doesn't get passed
        expected = type(self).eval_fn(self.test_params[0])

        pred_m, pred_var = self.model._predict(self.test_params[0:1])

        assert_allclose(expected.data, pred_m, rtol=1e-2)

    def test_predict_interface(self):

        # Get the actual test data
        #  Use the class method `eval_fn` so 'self' doesn't get passed
        expected = type(self).eval_fn(self.test_params[0])

        pred_m, pred_var = self.model.predict(self.test_params[0:1])

        assert_allclose(expected.data, pred_m.data, rtol=1e-2)
        assert pred_m.name() == 'Emulated ' + expected.name()
        assert pred_var is None
        assert pred_m.units == expected.units

    def test_predict_interface_multiple_samples(self):
        from iris.cube import CubeList
        # Get the actual test data
        #  Use the class method `eval_fn` so 'self' doesn't get passed
        expected = CubeList([type(self).eval_fn(p, job_n=i) for i, p in enumerate(self.test_params)])
        expected = expected.concatenate_cube()

        pred_m, pred_var = self.model.predict(self.test_params)

        # For some reason the relative tolerance has to be
        # higher here than in the other tests???
        assert_allclose(expected.data, pred_m.data, rtol=1e-1)
        assert pred_m.name() == 'Emulated ' + (expected.name() or 'data')
        assert pred_var is None
        assert pred_m.units == expected.units


class Simple1DTest(unittest.TestCase, RFTest):
    """
    Setup for the simple 1D 2 parameter test case
    """

    @classmethod
    def setUpClass(cls) -> None:
        params, test = pop_elements(get_uniform_params(2), 10, 12)
        ensemble = get_1d_two_param_cube(params)
        m = RFModel(training_params=params,
                    training_data=ensemble,
                    random_state=0)
        m.train()

        cls.model = m
        cls.params = params
        cls.test_params = test
        cls.eval_fn = eval_1d_cube


class Simple2DTest(unittest.TestCase, RFTest):
    """
    Setup for the simple 2D 3 parameter test case.
    """

    @classmethod
    def setUpClass(cls) -> None:
        params, test = pop_elements(get_uniform_params(3), 50)
        ensemble = get_three_param_cube(params)
        m = RFModel(training_params=params,
                    training_data=ensemble,
                    random_state=0)
        m.train()

        cls.model = m
        cls.params = params
        cls.test_params = test
        cls.eval_fn = eval_cube


class Simple32bitTest(unittest.TestCase, RFTest):
    """
    Setup for the simple 2D 3 parameter test case with 32bit data
    """

    @classmethod
    def setUpClass(cls) -> None:
        params, test = pop_elements(get_uniform_params(3), 50)

        ensemble = get_three_param_cube(params)
        # Create a new, ensemble at lower precision
        ensemble = ensemble.copy(data=ensemble.data.astype('float32'))
        m = RFModel(training_params=params,
                    training_data=ensemble,
                    random_state=0)
        m.train()

        cls.model = m
        cls.params = params
        cls.test_params = test
        cls.eval_fn = eval_cube


if __name__ == '__main__':
    unittest.main()
