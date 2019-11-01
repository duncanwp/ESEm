import numpy as np
import tensorflow as tf
from .model import Model
from tqdm import tqdm_notebook


class GPModel(Model):
    
    def __init__(self, training_data, name, GPU=0):
        import iris.cube
        
        self.GPU = GPU
            
        self.NP_TYPE = np.float64
        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g, config=tf.ConfigProto(allow_soft_placement=True, 
                                                                   log_device_placement=True))
        self.name = name
        t_data = training_data.data if isinstance(training_data, iris.cube.Cube) else training_data
        
        self.flat_Y = t_data.reshape((training_data.shape[0], -1)).astype(self.NP_TYPE)

        with self.sess.as_default(), self.sess.graph.as_default():
            self.TF_TYPE = tf.float64

    def train(self, X, params=None, verbose=False):
        with self.sess.as_default(), self.sess.graph.as_default(), tf.device('/gpu:{}'.format(self.GPU)):
            import gpflow
            
            gpflow.reset_default_session(graph=self.g)
            
            if params is None:
                n_params = X.shape[1]
                params = np.arange(n_params)
            else:
                n_params = len(params)
            
            if verbose:
                print("Fitting using dimensions: {}".format([params]))
           
            opt = gpflow.train.ScipyOptimizer(method='BFGS')

            X = X.astype(self.NP_TYPE)

            k = gpflow.kernels.RBF(n_params, lengthscales=0.5, variance=0.01, ARD=True, active_dims=params) + gpflow.kernels.Linear(n_params, ARD=True, active_dims=params) + gpflow.kernels.Bias(n_params)
            self.model = gpflow.models.GPR(X, self.flat_Y, k)
            self.model._session = self.sess
            self.model.compile()
            opt.minimize(self.model, disp=True)

    def sample(self, sample_points, batch_size=1000):
        with self.sess.as_default(), self.sess.graph.as_default(), tf.device('/gpu:{}'.format(self.GPU)):

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

    def constrain(self, sample_points, tol=0.5, batch_size=1000):
        with self.sess.as_default(), self.sess.graph.as_default(), tf.device('/gpu:{}'.format(self.GPU)):
            sample_T = tf.data.Dataset.from_tensor_slices(sample_points)
            n_samples = sample_points.shape[0]
            
            PROP = tf.constant(0.8, dtype=self.TF_TYPE)  # Proportion of valid samples required
            TOL = tf.constant(tol, dtype=self.TF_TYPE)  # Proportion of valid samples required

            all_valid_samples = []

            dataset = sample_T.batch(batch_size)
            # Define iterator
            iterator = dataset.make_initializable_iterator()
            # Get one batch
            next_example = iterator.get_next()

            # Get batch prediction (Note this isn't called yet)
            m, v = self.model._build_predict(next_example)
            my, yv = self.model.likelihood.predict_mean_and_var(m, v)
            
            # TODO: Consider using yv (the emulator uncertainty) somehow as another source of variance
            valid_samples = tf.greater(tf.reduce_sum(tf.cast(tf.less(tf.abs(tf.subtract(my, self.obs)),
                                                                     tf.multiply(self.var_obs, TOL)), dtype=self.TF_TYPE), axis=1),
                                       tf.multiply(PROP, tf.cast(tf.shape(my)[1], self.TF_TYPE)))
            
            constrained_mean = tf.where(valid_samples, my, tf.zeros([batch_size, tf.shape(self.obs)[0]], dtype=self.TF_TYPE))

            # Exponentiate the fields if log obs is true so I return the un-logged means and sd's
#             my = tf.cond(self.log_obs, lambda: tf.pow(my, tf.constant(10., dtype=self.TF_TYPE)), lambda: my)
#             constrained_mean = tf.cond(self.log_obs, lambda: tf.pow(constrained_mean, tf.constant(10., dtype=self.TF_TYPE)), lambda: constrained_mean)

            # Get the global mean
        #     global_mean = tf.reduce_mean(tf.multiply(my, N48_weights_T))

            # Get sum of x and sum of x**2
            s, s2 = tf.reduce_sum(my, axis=0), tf.reduce_sum(tf.square(my), axis=0)
            cn, cs, cs2 = tf.reduce_sum(tf.cast(valid_samples, tf.int32), axis=0), tf.reduce_sum(constrained_mean, axis=0), tf.reduce_sum(tf.square(constrained_mean), axis=0)

            self.sess.run(iterator.initializer)

            tot_s, tot_s2, tot_cs, tot_cs2, tot_cn = 0., 0., 0., 0., 0
        #     global_means = []

            pbar = tqdm_notebook(total=n_samples)

            while True:
                try:
                    n_s, n_s2, n_cs, n_cs2, n_cn, batch_valid_samples = self.sess.run([s, s2, cs, cs2, cn, valid_samples])
                    tot_s += n_s
                    tot_s2 += n_s2
                    tot_cs += n_cs
                    tot_cs2 += n_cs2
                    tot_cn += n_cn
                    all_valid_samples.append(batch_valid_samples)
                    pbar.update(batch_size)
                except tf.errors.OutOfRangeError:
                    break

            pbar.close()

        # Calculate the resulting first two moments
        unconstrained_mean = tot_s / n_samples
        constrained_mean = tot_cs / tot_cn

        unconstrained_sd = np.sqrt((tot_s2 - (tot_s * tot_s) / n_samples) / (n_samples - 1))
        constrained_sd = np.sqrt((tot_cs2 - (tot_cs * tot_cs) / tot_cn) / (tot_cn - 1))

        # Squash the list of batches
        all_valid_samples_arr = np.concatenate(all_valid_samples)
        return unconstrained_mean, unconstrained_sd, constrained_mean, constrained_sd, all_valid_samples_arr

