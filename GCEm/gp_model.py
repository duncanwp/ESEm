import tensorflow as tf
from .model import ModelAdaptor, CubeWrapper, Emulator
from .data_processors import Flatten, Recast


def gp_model(training_params, training_data, noise_variance=1., data_processors=None,
             name='', gpu=0, *args, **kwargs):
    import gpflow
    from gpflow.config import default_float

    n_params = training_params.shape[1]

    data_processors = data_processors if data_processors is not None else []
    data_processors.extend([Recast(default_float()), Flatten()])
    data = CubeWrapper(training_data, data_processors=data_processors)

    k = gpflow.kernels.RBF(lengthscales=[0.5] * n_params, variance=0.01) + \
        gpflow.kernels.Linear(variance=[1.] * n_params) + \
        gpflow.kernels.Polynomial(variance=[1.] * n_params) + \
        gpflow.kernels.Bias()

    # TODO: Look at the noise_variance term here - what does it represent?
    model = GPFlowModel(gpflow.models.GPR(data=(training_params, data.data_wrapper.data),
                                          kernel=k,
                                          noise_variance=tf.constant(noise_variance,
                                          dtype=data.dtype)))

    return Emulator(model, training_params, data, name=name, gpu=gpu)


class GPFlowModel(ModelAdaptor):
    """
    Simple Gaussian Process (GP) regression emulator which assumes independent
    inputs (and outputs). Different kernels can be specified.
    """

    def train(self, training_params, training_data, verbose=False, maxiter=100, **kwargs):
        import gpflow
        # Uses L-BFGS-B by default
        opt = gpflow.optimizers.Scipy()
        opt.minimize(self.model.training_loss,
                     variables=self.model.trainable_variables,
                     options=dict(disp=verbose, maxiter=maxiter), **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict_y(*args, **kwargs)

