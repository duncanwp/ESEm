import tensorflow as tf
from .model import Model
import gpflow


class GPModel(Model):
    """
    Simple Gaussian Process (GP) regression emulator which assumes independent
    inputs (and outputs). Different kernels can be specified.
    """

    def __init__(self, *args, **kwargs):
        from gpflow.config import default_float
        super(GPModel, self).__init__(*args, **kwargs)

        # Ensure the training data is the same as the GPFlow default (float64)
        #  We can't just reduce the precision of GPFlow to that of the data
        #  because this leads to unstable optimization
        self.dtype = default_float()
        self.training_data = self.training_data.astype(self.dtype)

    def _construct(self, *args, **kwargs):
        # TODO: A lot of this needs to be optional somehow
        return gpflow.kernels.RBF(lengthscales=[0.5]*self.n_params, variance=0.01) + \
            gpflow.kernels.Linear(variance=[1.]*self.n_params) + \
            gpflow.kernels.Polynomial(variance=[1.]*self.n_params) + \
            gpflow.kernels.Bias()

    def train(self, X, verbose=False):
        with tf.device('/gpu:{}'.format(self._GPU)):

            # Uses L-BFGS-B by default
            opt = gpflow.optimizers.Scipy()

            Y_flat = self.training_data.reshape((self.training_data.shape[0], -1))
            # TODO: Look at the noise_variance term here
            self.model = gpflow.models.GPR(data=(X, Y_flat), kernel=self._structure, noise_variance=1e-05)

            opt.minimize(self.model.training_loss,
                         variables=self.model.trainable_variables,
                         options=dict(disp=verbose, maxiter=100))
    
    def predict(self, *args, **kwargs):
        mean, var = self._tf_predict(*args, **kwargs)
        # Reshape the output to the original shape (neglecting the param dim)
        return (self._post_process(mean, 'Emulated '),
                self._post_process(var, 'Variance in emulated '))

    def _tf_predict(self, *args, **kwargs):
        return self.model.predict_y(*args, **kwargs)

