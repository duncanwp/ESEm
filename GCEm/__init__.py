"""
A package for easily emulating earth systems data

.. note ::

    The GCEm documentation has detailed usage information, including a :doc:`user guide <../index>`
    for new users.

"""
from .gp_model import GPModel
from .sampler import Sampler

__author__ = "Duncan Watson-Parris"
__version__ = "0.0.1"
__status__ = "Dev"


def calibrate(params, model_data, observational_data, model=GPModel, sampler=Sampler):
    # I think this is the basic structure. Could keep this as a high-level function?
    from utils import split_data
    # collocate?
    # TODO: think how to tidy these splits into one (use sklearn? API at least)
    train_y, val_y = split_data(model_data)
    train_x, val_y = split_data(params)

    # I don't really like passing the train separately, why do I do this?
    m = model(train_y, model_data.name())
    # There should probably be lots of (optional) arguments here, not least the optimiser
    m.train(train_x)
    # Not sure this needs to be a class?
    s = sampler(m)
    # Maybe this is just a __call__ interface?
    new_params = s.calibrate(observational_data)

