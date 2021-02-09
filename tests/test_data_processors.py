import unittest
from GCEm.data_processors import *
from tests.mock import *
from numpy.testing import assert_allclose, assert_array_equal


class DataProcessorTest(object):

    def test_reversibility(self):
        # Check that unprocess(process(x) == x (this is only true for the mean)
        mean, var = self.processor.unprocess(self.processor.process(self.test_data),
                                             self.test_data)
        assert_allclose(mean, self.test_data, rtol=1e-3)

    def test_no_side_effects(self):
        # ensure the processing doesn't actually change the data
        copy = self.test_data.copy()
        d = self.processor.process(self.test_data)
        assert_array_equal(copy, self.test_data)


class WhitenTest(unittest.TestCase, DataProcessorTest):
    """
    Setup for the Whiten test case
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_data = np.arange(1, 101, dtype=float).reshape((10, 10))
        cls.processor = Whiten()


class NormaliseTest(unittest.TestCase, DataProcessorTest):
    """
    Setup for the Whiten test case
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_data = np.arange(1, 101, dtype=float).reshape((10, 10))
        cls.processor = Normalise()


class LogTest(unittest.TestCase, DataProcessorTest):
    """
    Setup for the Whiten test case
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_data = np.arange(1, 101, dtype=float).reshape((10, 10))
        cls.processor = Log()


class LogP1Test(unittest.TestCase, DataProcessorTest):
    """
    Setup for the Whiten test case
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_data = np.arange(1, 101, dtype=float).reshape((10, 10))
        cls.processor = Log(plus_one=True)


class FlattenTest(unittest.TestCase, DataProcessorTest):
    """
    Setup for the Whiten test case
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_data = np.arange(1, 126, dtype=float).reshape((5, 5, 5))
        cls.processor = Flatten()


class ReshapeTest(unittest.TestCase, DataProcessorTest):
    """
    Setup for the Whiten test case
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_data = np.arange(1, 126, dtype=float).reshape((5, 5, 5, 1))
        cls.processor = Reshape()
