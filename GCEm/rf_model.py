from .model import ModelAdaptor, Emulator, CubeWrapper
from .data_processors import Flatten


def rf_model(training_params, training_data, data_processors=None, name='', gpu=0, *args, **kwargs):
    from sklearn.ensemble import RandomForestRegressor
    rfmodel = SKLearnModel(RandomForestRegressor(*args, **kwargs))

    # Add reshaping processor
    data_processors = data_processors if data_processors is not None else []
    data_processors.extend([Flatten()])
    data = CubeWrapper(training_data, data_processors=data_processors)

    return Emulator(rfmodel, training_params, data, name=name, gpu=gpu)


class SKLearnModel(ModelAdaptor):
    """
    Simple Random Forest Regression model for emulation.

    Note that because a Random Forest is just a
    recursive binary partition over the training data,
    there is no need to normalize/standardize the inputs.

    i.e. At least in theory, Random Forests are invariant
    to monotonic transformations of the independent variables
    """

    def train(self, training_params, training_data, verbose=False, *args, **kwargs):
        """
        Train the RF model. Note that this scikit
        implementation can't take advantage of GPUs.
        """
        if verbose:
            self.model.verbose = 1

        self.model.fit(X=training_params, y=training_data)

    def predict(self, *args, **kwargs):
        # Requires X_pred to be of shape (n_samples, n_features)
        return self.model.predict(*args, **kwargs), None
