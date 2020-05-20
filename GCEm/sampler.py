from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np


# TODO: I need to define some distance metrics (including uncertainty?) Should these be functions, or objects?
#  Should this be passed to __init__, or calibrate?

# TODO: I'm not yet sure how the MCMC sampling works so this might need adjusting

class Sampler(ABC):
    """
    A class that efficiently samples a Model object for posterior inference
    """

    def __init__(self, model, obs, var_obs=0., log_obs=False):
        """

        """
        self.model = model
        self.log_obs = log_obs
        self.obs = obs
        self.var_obs = var_obs

    def calibrate(self, objective, prior=None):
        """
        This is the call that does the actual inference.

        It should call model.sample over the prior, compare with the objective, and then output a posterior
        distribution

        :param objective: This is an Iris cube of observations
        :param prior: Ideally this would either be a numpy array or a tf.probability.distribution, could default to
        uniforms
        :return:
        """
        pass
