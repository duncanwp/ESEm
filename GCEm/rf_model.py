import numpy as np
from .model import Model
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

    def _construct(self, *args, **kwargs):
        from sklearn.ensemble import RandomForestRegressor
        rfmodel = RandomForestRegressor(*args, **kwargs)
        return rfmodel

    def train(self):
        """
        Train the RF model. Note that this scikit
        implementation can't take advantage of GPUs.
        """
        self.model.fit(X=self.training_params, y=self.training_data)

    def _tf_predict(self):
        # Requires X_pred to be of shape (n_samples, n_features)
        m = self.model.predict(test)
        var = None
        return mean, var
