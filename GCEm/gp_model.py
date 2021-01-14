import tensorflow as tf
from .model import Model, DataProcessor, Flatten
import gpflow
from gpflow.config import default_float


class Recast(DataProcessor):

    def __init__(self, new_type):
        self.new_type = new_type

    def process(self, data):
        self.old_type = data.dtype
        return data.astype(self.new_type)

    def unprocess(self, mean, variance):
        # I just leave this currently
        return mean, variance


class GPModel(Model):
    """
    Simple Gaussian Process (GP) regression emulator which assumes independent
    inputs (and outputs). Different kernels can be specified.
    """

    def __init__(self, *args, data_processors=None, **kwargs):
        # Add recasting and reshaping processors
        # Ensure the training data is the same as the GPFlow default (float64)
        #  We can't just reduce the precision of GPFlow to that of the data
        #  because this leads to unstable optimization
        data_processors = data_processors if data_processors is not None else []
        data_processors.extend([Recast(default_float()), Flatten()])

        super(GPModel, self).__init__(data_processors=data_processors, *args, **kwargs)

    def _construct(self, *args, noise_variance=1., **kwargs):
        # TODO: Look at the noise_variance term here - what does it represent?
        # TODO: A lot of this needs to be optional somehow
        k = gpflow.kernels.RBF(lengthscales=[0.5]*self.n_params, variance=0.01) + \
            gpflow.kernels.Linear(variance=[1.]*self.n_params) + \
            gpflow.kernels.Polynomial(variance=[1.]*self.n_params) + \
            gpflow.kernels.Bias()

        return gpflow.models.GPR(data=(self.training_params, self.training_data),
                                 kernel=k,
                                 noise_variance=tf.constant(noise_variance,
                                                            dtype=self.dtype))

    def train(self, verbose=False, **kwargs):
        with self.tf_device_context:

            # Uses L-BFGS-B by default
            opt = gpflow.optimizers.Scipy()
            opt.minimize(self.model.training_loss,
                         variables=self.model.trainable_variables,
                         options=dict(disp=verbose, maxiter=100), **kwargs)

    def _raw_predict(self, *args, **kwargs):
        with self.tf_device_context:
            return self.model.predict_y(*args, **kwargs)

