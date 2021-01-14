import numpy as np
from .model import Model, DataProcessor, Whiten
import tensorflow as tf


class Reshape(DataProcessor):

    def process(self, data):
        # Check the training data is the right shape for the ConvNet
        self.add_newaxis = False

        if data.ndim < 3:
            raise ValueError("Training data must have at least three "
                             "dimensions (including the sample dimension)")
        elif data.ndim == 3:
            self.add_newaxis = True
            data = data[..., np.newaxis]
        elif data.ndim > 4:
            raise ValueError("Training data must have at most four dimensions"
                             "(including the sample dimension)")
        return data

    def unprocess(self, mean, variance):
        if self.add_newaxis:
            mean = mean[..., 0]
            variance = variance[..., 0]
        return mean, variance


class NNModel(Model):
    """
    Perform emulation using a simple two layer convolutional NN.
    Note that X should include both the train and validation data
    """

    def __init__(self, *args, data_processors=None, **kwargs):
        # Add whitening and reshaping processors
        data_processors = data_processors if data_processors is not None else []
        data_processors.extend([Whiten(), Reshape()])

        super(NNModel, self).__init__(data_processors=data_processors, *args, **kwargs)
        self.dtype = tf.float32

    def _construct(self, filters=12, learning_rate=1e-3, decay=0.01,
                   kernel_size=(3, 5), loss='mean_squared_error',
                   activation='tanh', optimizer='RMSprop'):
        from tensorflow.keras.layers import Dense, Input, Reshape, Conv2DTranspose
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam, RMSprop

        if optimizer == 'RMSprop':
            optimizer = RMSprop
        elif optimizer == 'Adam':
            optimizer = Adam
        else:
            raise ValueError("Invalid optimizer specified: {}".format(optimizer))

        # build a simple decoder model
        latent_inputs = Input((self.n_params, ))
        intermediate_shape = self.training_data.shape[1:-1] + (filters, )
        x = Dense(np.product(intermediate_shape), activation='relu')(latent_inputs)
        x = Reshape(intermediate_shape)(x)

        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            activation='relu',
                            strides=1,
                            padding='same', data_format='channels_last')(x)

        outputs = Conv2DTranspose(filters=self.training_data.shape[-1],
                                  kernel_size=kernel_size,
                                  activation=activation,
                                  strides=1,
                                  padding='same', data_format='channels_last')(x)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')

        # TODO: Also try 'mean_absolute_error' loss
        decoder.compile(optimizer=optimizer(learning_rate=learning_rate, decay=decay), loss=loss, )
        return decoder

    def train(self, verbose=False, epochs=100, batch_size=8, validation_split=0.2, **kwargs):
        """
        Train the (keras) NN model.

        :param X:
        :param verbose:
        :param epochs:
        :param batch_size:
        :param float validation_split: The proportion of training data to use for validation
        :return:
        """
        with self.tf_device_context:
            self.model.fit(self.training_params, self.training_data,
                           batch_size=batch_size, epochs=epochs,
                           validation_split=validation_split, **kwargs)

    def _raw_predict(self, *args, **kwargs):
        with self.tf_device_context:
            # This only works with the tf.keras API
            return self.model(*args, **kwargs), None
