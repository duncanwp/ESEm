import numpy as np
from .model import Model
import tensorflow as tf


class NNModel(Model):
    """
    Perform emulation using a simple two layer convolutional NN.
    Note that X should include both the train and validation data
    """

    def __init__(self, *args, **kwargs):
        super(NNModel, self).__init__(*args, **kwargs)
        self.dtype = tf.float32

    def _pre_process(self):
        # Normalise the training data
        # self.mean_t = self.training_data.mean(axis=0)
        # self.training_data = (self.training_data - self.mean_t)
        # TODO: Make whitening, normalizing (and weighting) optional in the constructor
        # pass
        self.training_data = self.whiten(self.training_data)

        # Check the training data is the right shape for the ConvNet
        if self.training_data.ndim < 3:
            raise ValueError("Training data must have at least three "
                             "dimensions (including the sample dimension)")
        elif self.training_data.ndim == 3:
            self.training_data = self.training_data[..., np.newaxis]
        elif self.training_data.ndim > 4:
            raise ValueError("Training data must have at most four dimensions"
                             "(including the sample dimension)")

    def _post_process(self, data):
        if data is not None:
            # If the last (color) dimension is one then pop it off (we added it
            #  in pre-processing
            if data.shape[-1] == 1:
                data = data[..., 0]
            data = self.un_whiten(data)
        return super(NNModel, self)._post_process(data)

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
