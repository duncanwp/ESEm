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
             kernel=None, kernel_op='add', active_dims=None, noise_variance=1.,
             name='', gpu=0):
    """
    Create a Gaussian process (GP) based emulator with provided `training_params` (X) and `training_data` (Y).

    The `kernel` is a key parameter in GP emulation and care should be taken in choosing it.

    Parameters
    ----------
    training_params: pd.DataFrame
        The training parameters
    training_data: iris.cube.Cube or array_like
        The training data - the leading dimension should represent training samples
    data_processors: list of GCEm.data_processors.DataProcessor
        A list of `DataProcessor`s to apply to the data transparently before training. Model output will be
        un-transformed before being returned from the Emulator.
    kernel: gpflow.kernels.Kernel or list of str or None
        The GP kernel to use.  A GPFlow kernel can be specified directly, or a list of kernel names can be provided
        which will be initialised using the default values (of the correct shape) and combined using `kernel_op`.
        Alternatively no kernel can be specified and a default will be used.
    kernel_op : {'add', 'mul'}
        The operation to perform in order to combine the specified `kernel`s. Only used if `kernel` is a list of strings.
    noise_variance: float
        The noise variance to initialise the GP regression model
    active_dims: list of int or slice or None
        The dimensions to train the GP over (by default all of the dimensions are used)
    name: str
        An optional name for the emulator
    gpu: int
        The GPU to use (only applicable for multi-GPU) machines

    Returns
    -------
    Emulator
        A GCEm emulator object which can be trained and sampled from

    """

    from .model_adaptor import GPFlowModel
    import tensorflow as tf
    import gpflow
    from gpflow.config import default_float

    n_params = training_params.shape[1]

    data_processors = data_processors if data_processors is not None else []
    data_processors.extend([Recast(default_float()), Flatten()])
    data = CubeWrapper(training_data, data_processors=data_processors)

    if isinstance(kernel, list):
        kernel = _get_gpflow_kernel(kernel, n_params, active_dims=active_dims, operator=kernel_op)
    elif kernel is None:
        # TODO Maybe this should just use the above mechanism?
        kernel = gpflow.kernels.RBF(lengthscales=[0.5] * n_params, variance=0.01, active_dims=active_dims) + \
                 gpflow.kernels.Linear(variance=[1.] * n_params, active_dims=active_dims) + \
                 gpflow.kernels.Polynomial(variance=[1.] * n_params, active_dims=active_dims) + \
                 gpflow.kernels.Bias(active_dims=active_dims)
        print("WARNING: Using default kernel - be sure you understand the assumptions this implies")
    else:
        pass  # Use the user specified kernel

    model = GPFlowModel(gpflow.models.GPR(data=(training_params, data.data_wrapper.data),
                                          kernel=kernel,
                                          noise_variance=tf.constant(noise_variance,
                                          dtype=data.dtype)))

    return Emulator(model, training_params, data, name=name, gpu=gpu)


def _get_gpflow_kernel(names, n_params, active_dims=None, operator='add'):
    """
        Helper function for creating a single GPFlow kernel from a combination of kernel names.

    Parameters
    ----------
    names: list of str
        A list of the names of the kernels to be combined
    n_params: int
        The number of training parameters
    active_dims: list of int or slice or None
        The active dimensions to allow the kernel to fit
    operator: {'add', 'mul'}
        The operator to use to combine the kernels

    Returns
    -------
    gpflow.kernels.Kernel
        The combined GPFlow kernel
    """
    import gpflow.kernels
    from operator import add, mul
    from functools import reduce

    kernel_dict = {
        "RBF": gpflow.kernels.RBF,
        "Linear": gpflow.kernels.Linear,
        "Polynomial": gpflow.kernels.Polynomial,
        "Bias": gpflow.kernels.Bias,
        "Cosine": gpflow.kernels.Cosine,
        "Exponential": gpflow.kernels.Exponential,
        "Matern12": gpflow.kernels.Matern12,
        "Matern32": gpflow.kernels.Matern32,
        "Matern52": gpflow.kernels.Matern52,
    }

    operator_dict = {
        'add': add,
        'mul': mul
    }

    def init_kernel(k):
        """
        Initialise a GPFlow kernel with the correct shape (default) variance and lengthscales
        """
        if issubclass(k, gpflow.kernels.Constant):  # E.g., Bias
            return k(active_dims=active_dims)
        elif issubclass(k, gpflow.kernels.Linear):  # This covers polynomial too
            return k(variance=[1.] * n_params, active_dims=active_dims)
        elif issubclass(k, gpflow.kernels.Stationary):  # This covers e.g. RBF
            return k(lengthscales=[1.] * n_params, active_dims=active_dims)
        else:
            raise ValueError("Invalid Kernel: {}".format(k))

    return reduce(operator_dict[operator], (init_kernel(kernel_dict[k]) for k in names))


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
