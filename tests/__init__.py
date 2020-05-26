import tensorflow as tf

# Force tf.functions to be run eagerly to properly profile coverage
tf.config.experimental_run_functions_eagerly(True)
# tf.config.run_functions_eagerly(True)
