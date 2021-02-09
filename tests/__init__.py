import tensorflow as tf
import pytest
import os

# Force tf.functions to be run eagerly to properly profile coverage
tf.config.experimental_run_functions_eagerly(True)
# tf.config.run_functions_eagerly(True)

skip_on_ci = pytest.mark.skipif(
    os.getenv("CI", 'false').lower() == 'true', reason="Skipping test when run on Continuous Integration server."
)
