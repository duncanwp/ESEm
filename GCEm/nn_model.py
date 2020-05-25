import numpy as np
from .model import Model


class NNModel(Model):
    """
    Perform emulation using a simple two layer convolutional NN.
    Note that X should include both the train and validation data
    """

    def _pre_process(self):
        # Normalise the training data
        # self.mean_t = self.training_data.mean(axis=0)
        # self.training_data = (self.training_data - self.mean_t)
        # TODO: Maybe re-weight
        # pass
        self.training_data = self.whiten(self.training_data)

    def _post_process(self, data, *args, **kwargs):
        scaled_data = self.un_whiten(data)
        return super(NNModel, self)._post_process(scaled_data, *args, **kwargs)

    def _construct(self, filters=12, learning_rate=1e-2, decay=0.01,
                   kernel_size=(3, 5), loss='mean_squared_error',
                   activation='tanh', optimizer='RMSprop'):
        from keras.layers import Dense, Input, Reshape, Conv2DTranspose
        from keras.models import Model
        from keras.optimizers import Adam, RMSprop

        if optimizer == 'RMSprop':
            optimizer = RMSprop
        elif optimizer == 'Adam':
            optimizer = Adam
        else:
            raise ValueError("Invalid optimizer specified: {}".format(optimizer))

        shape = self.training_data.shape[1:]
        # FIXME - I think the problem here is the lack of a 'color' channel
        latent_inputs = Input((self.n_params, ))
        x = Dense(np.product(shape), activation='relu')(latent_inputs)
        x = Reshape(shape)(x)

        x = Conv2DTranspose(filters=6, input_shape=shape,
                            kernel_size=kernel_size,
                            activation='relu',
                            strides=1,
                            padding='same', data_format='channels_first')(x)

        outputs = Conv2DTranspose(filters=filters, input_shape=shape,
                                  kernel_size=kernel_size,
                                  activation='sigmoid',
                                  strides=1,
                                  padding='same', data_format='channels_first')(x)

        # build simple decoder model
        # TODO: This simple dense network sort-of worked, and may have got close with lots of tuning...
        # latent_inputs = Input(shape=(self.n_params,), name='params')
        # x = Dense(np.product(shape), activation=activation)(latent_inputs)
        # x = Dense(np.product(shape), activation='linear')(x)
        # outputs = Reshape(shape)(x)
        #
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
        self.model.fit(self.training_params, self.training_data,
                       batch_size=batch_size, epochs=epochs,
                       validation_split=validation_split, **kwargs)

    def predict(self, *args, **kwargs):
        return self._post_process(self.model.predict(*args, **kwargs)), None

    def _tf_predict(self, *args, **kwargs):
        # As far as I can see Keras doesn't have a way for plugging in to this
        #  I could probably use tf.keras though
        return self.model.predict(*args, **kwargs), None

