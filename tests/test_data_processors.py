from esem.data_processors import *
from tests.mock import *
from numpy.testing import assert_allclose, assert_array_equal


class DataProcessorTest:

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


class TestWhiten(DataProcessorTest):
    """
    Setup for the Whiten test case
    """

    @classmethod
    def setup_class(cls) -> None:
        cls.test_data = np.arange(1, 101, dtype=float).reshape((10, 10))
        cls.processor = Whiten()


class TestNormalise(DataProcessorTest):
    """
    Setup for the Whiten test case
    """

    @classmethod
    def setup_class(cls) -> None:
        cls.test_data = np.arange(1, 101, dtype=float).reshape((10, 10))
        cls.processor = Normalise()


class TestLog(DataProcessorTest):
    """
    Setup for the Whiten test case
    """

    @classmethod
    def setup_class(cls) -> None:
        cls.test_data = np.arange(1, 101, dtype=float).reshape((10, 10))
        cls.processor = Log()


class TestLogP1(DataProcessorTest):
    """
    Setup for the Whiten test case
    """

    @classmethod
    def setup_class(cls) -> None:
        cls.test_data = np.arange(1, 101, dtype=float).reshape((10, 10))
        cls.processor = Log(constant=1.0)


class TestFlatten(DataProcessorTest):
    """
    Setup for the Whiten test case
    """

    @classmethod
    def setup_class(cls) -> None:
        cls.test_data = np.arange(1, 126, dtype=float).reshape((5, 5, 5))
        cls.processor = Flatten()


class TestReshape(DataProcessorTest):
    """
    Setup for the Whiten test case
    """

    @classmethod
    def setup_class(cls) -> None:
        cls.test_data = np.arange(1, 126, dtype=float).reshape((5, 5, 5, 1))
        cls.processor = Reshape()
