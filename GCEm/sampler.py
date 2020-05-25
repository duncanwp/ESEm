from abc import ABC
import numpy as np


# TODO: I need to define some distance metrics (including uncertainty?) Should these be functions, or objects?
#  Should this be passed to __init__, or calibrate?

# TODO: I'm not yet sure how the MCMC sampling works so this might need adjusting


class Sampler(ABC):
    """
    A class that efficiently samples a Model object for posterior inference
    """

    def __init__(self, model, obs,
                 obs_uncertainty=0., interann_uncertainty=0.,
                 repres_uncertainty=0., struct_uncertainty=0.):
        """
        :param GCEm.model.Model model:
        :param iris.cube.Cube obs: The objective
        :param float obs_uncertainty: Fractional, relative (1 sigma) uncertainty in observations
        :param float repres_uncertainty: Fractional, relative (1 sigma) uncertainty due to the spatial and temporal
         representitiveness of the observations
        :param float interann_uncertainty: Fractional, relative (1 sigma) uncertainty introduced when using a model run
         for a year other than that the observations were measured in.
        :param float struct_uncertainty: Fractional, relative (1 sigma) uncertainty in the model itself.
        """
        self.model = model
        self.obs = obs

        # TODO: Could add an absolute uncertainty term here
        # Get the square of the absolute uncertainty and broadcast it across the batch (since it's the same for each sample)
        observational_var = np.reshape(np.square(obs.data * obs_uncertainty), (1, obs.shape[0]))
        respres_var = np.reshape(np.square(obs.data * repres_uncertainty), (1, obs.shape[0]))
        interann_var = np.reshape(np.square(obs.data * interann_uncertainty), (1, obs.shape[0]))
        struct_var = np.reshape(np.square(obs.data * struct_uncertainty), (1, obs.shape[0]))
        self.total_var = sum([observational_var, respres_var, interann_var, struct_var])

    def sample(self, prior_x, n_samples):
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

# TODO SEPARETLY - Do this without tolerance and threshold by calculating the actual probability and accepting/rejecting against a uniform dist


