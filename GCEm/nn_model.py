import numpy as np
from .model import ModelAdaptor, Emulator, CubeWrapper
from .data_processors import Whiten, Reshape as Reshaper, Recast


def cnn_model(training_params, training_data, data_processors=None,
              filters=12, learning_rate=1e-3, decay=0.01,
              kernel_size=(3, 5), loss='mean_squared_error',
              activation='tanh', optimizer='RMSprop', name='', gpu=0):
    # TODO: Also try 'mean_absolute_error' loss
    from tensorflow.keras.layers import Dense, Input, Reshape, Conv2DTranspose
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow.keras.backend import floatx

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


class KerasModel(ModelAdaptor):
    """
    Perform emulation using a simple two layer convolutional NN.
    Note that X should include both the train and validation data
    """

    def train(self, training_params, training_data, verbose=False, epochs=100, batch_size=8, validation_split=0.2, **kwargs):
        """
        Train the (keras) NN model.

        :param X:
        :param verbose:
        :param epochs:
        :param batch_size:
        :param float validation_split: The proportion of training data to use for validation
        :return:
        """
        self.model.fit(training_params, training_data,
                       batch_size=batch_size, epochs=epochs,
                       validation_split=validation_split, **kwargs)

    def predict(self, *args, **kwargs):
        # This only works with the tf.keras API
        return self.model(*args, **kwargs), None
