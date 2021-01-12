import numpy as np
from .model import Model, Flatten
import tensorflow as tf


class RFModel(Model):
    """
    Simple Random Forest Regression model for emulation.

    Note that because a Random Forest is just a
    recursive binary partition over the training data,
    there is no need to normalize/standardize the inputs.

    i.e. At least in theory, Random Forests are invariant
    to monotonic transformations of the independent variables
    """

    def __init__(self, *args, data_processors=None, **kwargs):
        # Add reshaping processor
        data_processors = data_processors if data_processors is not None else []
        data_processors.extend([Flatten()])

        super(RFModel, self).__init__(data_processors=data_processors, *args, **kwargs)

    def _construct(self, *args, **kwargs):
        from sklearn.ensemble import RandomForestRegressor
        rfmodel = RandomForestRegressor(*args, **kwargs)
        return rfmodel

    def train(self, verbose=False):
        """
        Train the RF model. Note that this scikit
        implementation can't take advantage of GPUs.
        """
        if verbose:
            self.model.verbose = 1

        self.model.fit(X=self.training_params, y=self.training_data)

    def _raw_predict(self, *args, **kwargs):
        # Requires X_pred to be of shape (n_samples, n_features)
        return self.model.predict(*args, **kwargs), None
