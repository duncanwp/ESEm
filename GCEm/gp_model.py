import numpy as np
import tensorflow as tf
from .model import Model
from tqdm import tqdm


class GPModel(Model):

    def train(self, X, params=None, verbose=False):
        with tf.device('/gpu:{}'.format(self._GPU)):
            import gpflow
            
            if params is None:
                n_params = X.shape[1]
                params = np.arange(n_params)
            else:
                n_params = len(params)
            
            if verbose:
                print("Fitting using dimensions: {}".format([params]))

            # Uses L-BFGS-B by default
            opt = gpflow.optimizers.Scipy()

            # TODO: A lot of this needs to be optional somehow
            k = gpflow.kernels.RBF(lengthscales=[0.5]*n_params, variance=0.01, active_dims=params) + \
                gpflow.kernels.Linear(variance=[1.]*n_params, active_dims=params) + \
                gpflow.kernels.Polynomial(variance=[1.]*n_params, active_dims=params) + \
                gpflow.kernels.Bias(active_dims=params)

            Y_flat = self.training_data.reshape((self.training_data.shape[0], -1))
            self.model = gpflow.models.GPR(data=(X, Y_flat), kernel=k)

            opt.minimize(self.model.training_loss,
                         variables=self.model.trainable_variables,
                         options=dict(disp=verbose, maxiter=100))

    @tf.function
    def _sample(self, sample_points, batch_size=1000):
        with tf.device('/gpu:{}'.format(self._GPU)):

            sample_T = tf.data.Dataset.from_generator(sample_points, (self.TF_TYPE, self.TF_TYPE))
            n_samples = sample_points.shape[0]
            
            dataset = sample_T.batch(batch_size)
            # Define iterator
            iterator = dataset.make_initializable_iterator()
            # Get one batch
            next_example = iterator.get_next()

            # Get batch prediction
            m, v = self.model._build_predict(next_example)
            my, yv = self.model.likelihood.predict_mean_and_var(m, v)
            
            # Get sum of x and sum of x**2
            s, s2 = tf.reduce_sum(my, axis=0), tf.reduce_sum(tf.square(my), axis=0)

#             sample_writer = tf.summary.FileWriter(project_path + '/log', self.sess.graph)
            self.sess.run(iterator.initializer)

            tot_s, tot_s2 = 0., 0.
            pbar = tqdm_notebook(total=n_samples)

            while True:
                try:
                    n_s, n_s2 = self.sess.run([s, s2])
                    tot_s += n_s
                    tot_s2 += n_s2
                    pbar.update(batch_size)
                except tf.errors.OutOfRangeError:
                    break

            pbar.close()

        # Calculate the resulting first two moments
        mean = tot_s / n_samples
        sd = np.sqrt((tot_s2 - (tot_s * tot_s) / n_samples) / (n_samples - 1))

        return mean, sd
    
    def constrain(self):
        raise NotImplemented()

    def predict(self, *args, **kwargs):
        m, v = self._tf_predict(*args, **kwargs)
        # Reshape the output to the original shape (neglecting the param dim)

        return (m.numpy().reshape(self.original_shape[1:]),
                v.numpy().reshape(self.original_shape[1:]))

    def _post_process(self):
        pass

    def _tf_predict(self, *args, **kwargs):
        return self.model.predict_y(*args, **kwargs)

class GPModel_IC(GPModel):

    def train(self, X, **kwargs):
        from GCEm.utils import get_param_mask
        params, = np.where(get_param_mask(X, self.flat_Y))
        super(ObsConstraint).train(X, params, **kwargs)
        

class ObsConstraint(GPModel):
    def __init__(self, training_data, obs, log_obs, *args, **kwargs):
        import iris.analysis
        super(ObsConstraint).__init__(training_data, *args, **kwargs)
        
        # Calculate the variabiltiy across the training data
        var_obs = training_data.collapsed('job', iris.analysis.STD_DEV).data.flatten()

        with self.sess.as_default(), self.sess.graph.as_default():
            self.log_obs = tf.constant(log_obs)    
            self.obs = tf.convert_to_tensor(obs, dtype=self.TF_TYPE)
            self.var_obs = tf.convert_to_tensor(var_obs, dtype=self.TF_TYPE)


def tf_tqdm(ds):
    import io
    # Suppress printing the initial status message - it creates extra newlines, for some reason.
    bar = tqdm(file=io.StringIO())

    def advance_tqdm(e):
        def fn():
            bar.update(1)
            # Print the status update manually.
            print('\r', end='')
            print(repr(bar), end='')
        tf.py_function(fn, [], [])
        return e

    return ds.map(advance_tqdm)


def constrain(implausibility, tolerance=0., threshold=3.0):
    """
        Return a boolean array indicating if each sample meets the implausibility criteria:

            I < T

    :param np.array implausibility: Distance of each sample from each observation (in S.Ds)
    :param float tolerance: The fraction of samples which are allowed to be over the threshold
    :param float threshold: The number of standard deviations a sample is allowed to be away from the obs
    :return np.array: Boolean array of samples which meet the implausibility criteria
    """
    # Return True (for a sample) if the number of implausibility measures greater
    #  than the threshold is less than or equal to the tolerance
    return np.less_equal(np.sum(np.greater(implausibility, threshold), axis=0), tolerance)


def get_implausibility(model, obs, sample_points,
                       obs_uncertainty=0., interann_uncertainty=0.,
                       repres_uncertainty=0., struct_uncertainty=0., batch_size=1000):
    """

    Each of the specified uncertainties are assumed to be normal and are added in quadrature

    NOTE - each of the uncertaintes are 1 sigma as compared to previous versions which assumed 2 sigma

    :param model:
    :param obs:
    :param sample_points:
    :param float obs_uncertainty: Fractional, relative (1 sigma) uncertainty in obervations
    :param float repres_uncertainty: Fractional, relative (1 sigma) uncertainty due to the spatial and temporal
     representitiveness of the observations
    :param float interann_uncertainty: Fractional, relative (1 sigma) uncertainty introduced when using a model run
     for a year other than that the observations were measured in.
    :param float struct_uncertainty: Fractional, relative (1 sigma) uncertainty in the model itself.
    :param int batch_size:
    :return:
    """

    #TODO: Could add an absolute uncertainty term here

    # Get the square of the absolute uncertainty and broadcast it across the batch (since it's the same for each sample)
    observational_var = np.broadcast_to(np.square(obs * obs_uncertainty), (batch_size, obs.shape[0]))
    respres_var = np.broadcast_to(np.square(obs * repres_uncertainty), (batch_size, obs.shape[0]))
    interann_var = np.broadcast_to(np.square(obs * interann_uncertainty), (batch_size, obs.shape[0]))
    struct_var = np.broadcast_to(np.square(obs * struct_uncertainty), (batch_size, obs.shape[0]))

    implausibility = _calc_implausibility(model, obs, sample_points,
                                          observational_var, interann_var,
                                          respres_var, struct_var, batch_size=batch_size)
    # TODO: I could return this as a cube for easier plotting

    return implausibility


@tf.function
def _calc_implausibility(model, obs, sample_points,
                         observational_var, interann_var,
                         respres_var, struct_var, batch_size=1000):
    """

    Each of the specified uncertainties are assumed to be normal and are added in quadrature

    NOTE - each of the uncertaintes are 1 sigma as compared to previous versions which assumed 2 sigma

    :param model:
    :param Tensor obs:
    :param Tensor sample_points:
    :param Tensor observational_var: Variance in obervations
    :param Tensor respres_var: Fractional, relative (1 sigma) uncertainty due to the spatial and temporal
     representitiveness of the observations
    :param Tensor interann_var: Fractional, relative (1 sigma) uncertainty introduced when using a model run
     for a year other than that the observations were measured in.
    :param Tensor struct_var: Fractional, relative (1 sigma) uncertainty in the model itself.
    :param int batch_size:
    :return:
    """
    with tf.device('/gpu:{}'.format(model._GPU)):

        sample_T = tf.data.Dataset.from_tensor_slices(sample_points)

        dataset = sample_T.batch(batch_size)

        all_implausibility = tf.zeros((0, ), dtype=tf.bool)

        for data in tf_tqdm(dataset):
            # Get batch prediction (Note this isn't called yet)
            m, v = model.model.predict_y(data)
            emulator_mean, emulator_var = model.model.likelihood.predict_mean_and_var(m, v)

            tot_sd = tf.sqrt(tf.add_n([emulator_var, observational_var, respres_var, interann_var, struct_var]))
            implausibility = tf.divide(tf.abs(tf.subtract(emulator_mean, obs)), tot_sd)

            all_implausibility = tf.concat([all_implausibility, implausibility], 0)

    return all_implausibility


@tf.function
def sample_mean(model, sample_points, batch_size=1):
    with tf.device('/gpu:{}'.format(model._GPU)):
        sample_T = tf.data.Dataset.from_tensor_slices(sample_points)

        dataset = sample_T.batch(batch_size)

        tot_s = tf.constant(0., dtype=model.dtype)  # Proportion of valid samples required
        tot_s2 = tf.constant(0., dtype=model.dtype)  # Proportion of valid samples required

        for data in tf_tqdm(dataset):
            # Get batch prediction (Note this isn't called yet)
            m, v = model.model.predict_y(data)
            my, yv = model.model.likelihood.predict_mean_and_var(m, v)

            # Get sum of x and sum of x**2
            tot_s += tf.reduce_sum(my, axis=0)
            tot_s2 += tf.reduce_sum(tf.square(my), axis=0)

    n_samples = tf.cast(sample_points.shape[0], dtype=model.dtype)  # Make this a float to allow division
    # Calculate the resulting first two moments
    mean = tot_s / n_samples
    sd = tf.sqrt((tot_s2 - (tot_s * tot_s) / n_samples) / (n_samples - 1))

    return mean, sd
