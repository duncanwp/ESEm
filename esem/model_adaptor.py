from abc import ABC, abstractmethod


class ModelAdaptor(ABC):
    """
    Provides a unified interface for all emulation engines within ESEm.
    Concrete classes must implement both :meth:`train` and :meth:`predict` methods.

    See the `API documentation <../api.html#dataprocessor>`_ for a list of concrete
    classes implementing this interface.
    """

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def train(self, training_params, training_data, verbose=False, **kwargs):
        """
        Train on the training data
        :return:
        """

    @abstractmethod
    def predict(self, *args, **kwargs):
        """
        This is either the tf model which can be called directly, or a generator over the model.predict (in tf,
        so it's quick).

        :return:
        """


class SKLearnModel(ModelAdaptor):
    """
    A wrapper around `scikit-learn <https://scikit-learn.org>`_ models.
    """

    def train(self, training_params, training_data, verbose=False, **kwargs):
        """
        Train the RF model. Note that this scikit
        implementation can't take advantage of GPUs.
        """
        if verbose:
            self.model.verbose = 1

        self.model.fit(X=training_params, y=training_data, **kwargs)

    def predict(self, *args, **kwargs):
        # Requires X_pred to be of shape (n_samples, n_features)
        return self.model.predict(*args, **kwargs), None


class KerasModel(ModelAdaptor):
    """
    A wrapper around `Keras <https://keras.io/>`_ models
    """

    def train(self, training_params, training_data, verbose=False, epochs=100, batch_size=8, validation_split=0.2, **kwargs):
        """
        Train the Keras model.

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


class GPFlowModel(ModelAdaptor):
    """
    A wrapper around `GPFlow <https://gpflow.readthedocs.io/en/master/#>`_ regression models
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
