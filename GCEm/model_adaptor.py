from abc import ABC, abstractmethod


class ModelAdaptor(ABC):

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def train(self, training_params, training_data, verbose=False, **kwargs):
        """
        Train on the training data
        :return:
        """
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        """
        This is either the tf model which I can then call, or a generator over the model.predict (in tf, so it's quick).
        This function

        The sampler (using either tf.probability.mcmc and my ABC method) can then just call this to get samples

        :return:
        """
        pass


class SKLearnModel(ModelAdaptor):
    """
    Simple Random Forest Regression model for emulation.

    Note that because a Random Forest is just a
    recursive binary partition over the training data,
    there is no need to normalize/standardize the inputs.

    i.e. At least in theory, Random Forests are invariant
    to monotonic transformations of the independent variables
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