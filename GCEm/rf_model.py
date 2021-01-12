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
        Y_flat = self.training_data.reshape((self.training_data.shape[0], -1))
        self.model.fit(X=self.training_params, y=Y_flat)

    def _raw_predict(self, *args, **kwargs):
        # Requires X_pred to be of shape (n_samples, n_features)
        m = self.model.predict(*args, **kwargs)

        mean = np.reshape(m, (-1,) + self.training_data.shape[1:])
        var = None
        return mean, var
