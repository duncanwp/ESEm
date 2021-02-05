"""
A package for easily emulating earth systems data

.. note ::

    The GCEm documentation has detailed usage information, including a :doc:`user guide <../index>`
    for new users.

"""
from .cube_wrapper import CubeWrapper
from .data_processors import Recast, Whiten, Reshape as Reshaper, Flatten
from .emulator import Emulator

__author__ = "Duncan Watson-Parris"
__version__ = "0.2.0"
__status__ = "Dev"


def gp_model(training_params, training_data, data_processors=None,
             kernel=None, noise_variance=1., name='', gpu=0):
    from .model_adaptor import GPFlowModel
    import tensorflow as tf
    import gpflow
    from gpflow.config import default_float

    n_params = training_params.shape[1]

    data_processors = data_processors if data_processors is not None else []
    data_processors.extend([Recast(default_float()), Flatten()])
    data = CubeWrapper(training_data, data_processors=data_processors)

    if kernel is None:
        kernel = gpflow.kernels.RBF(lengthscales=[0.5] * n_params, variance=0.01) + \
                 gpflow.kernels.Linear(variance=[1.] * n_params) + \
                 gpflow.kernels.Polynomial(variance=[1.] * n_params) + \
                 gpflow.kernels.Bias()
        print("WARNING: Using default kernel - be sure you understand the assumptions this implies")

    # TODO: Look at the noise_variance term here - what does it represent?
    model = GPFlowModel(gpflow.models.GPR(data=(training_params, data.data_wrapper.data),
                                          kernel=kernel,
                                          noise_variance=tf.constant(noise_variance,
                                          dtype=data.dtype)))

    return Emulator(model, training_params, data, name=name, gpu=gpu)


def cnn_model(training_params, training_data, data_processors=None,
              filters=12, learning_rate=1e-3, decay=0.01,
              kernel_size=(3, 5), loss='mean_squared_error',
              activation='tanh', optimizer='RMSprop', name='', gpu=0):
    from .model_adaptor import KerasModel
    from tensorflow.keras.layers import Dense, Input, Reshape, Conv2DTranspose
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow.keras.backend import floatx
    import numpy as np

    if optimizer == 'RMSprop':
        optimizer = RMSprop
    elif optimizer == 'Adam':
        optimizer = Adam
    else:
        raise ValueError("Invalid optimizer specified: {}".format(optimizer))

    # Add whitening and reshaping processors
    data_processors = data_processors if data_processors is not None else []
    data_processors.extend([Recast(floatx()), Whiten(), Reshaper()])
    data = CubeWrapper(training_data, data_processors=data_processors)

    # build a simple decoder model
    # NOTE that this assumes the data doesn't get reshaped by the data processors...
    latent_inputs = Input((training_params.shape[1],))
    intermediate_shape = data.data_wrapper.data.shape[1:-1] + (filters,)
    x = Dense(np.product(intermediate_shape), activation='relu')(latent_inputs)
    x = Reshape(intermediate_shape)(x)

    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=1,
                        padding='same', data_format='channels_last')(x)

    outputs = Conv2DTranspose(filters=data.data_wrapper.data.shape[-1],
                              kernel_size=kernel_size,
                              activation=activation,
                              strides=1,
                              padding='same', data_format='channels_last')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.compile(optimizer=optimizer(learning_rate=learning_rate, decay=decay), loss=loss)

    model = KerasModel(decoder)

    return Emulator(model, training_params, data, name=name, gpu=gpu)


def rf_model(training_params, training_data, data_processors=None, name='', gpu=0, *args, **kwargs):
    from .model_adaptor import SKLearnModel
    from sklearn.ensemble import RandomForestRegressor
    rfmodel = SKLearnModel(RandomForestRegressor(*args, **kwargs))

    # Add reshaping processor
    data_processors = data_processors if data_processors is not None else []
    data_processors.extend([Flatten()])
    data = CubeWrapper(training_data, data_processors=data_processors)

    return Emulator(rfmodel, training_params, data, name=name, gpu=gpu)
