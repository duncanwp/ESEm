import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
import pandas as pd


def add_121_line(ax):
    import numpy as np
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)


def prettify_plot(ax):
    """utility function for making plots prettier"""
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_results(ax, truth, pred, title):
    """ Validation plot for LeaveOneOut """
    from sklearn.metrics import median_absolute_error, r2_score, mean_squared_error
    from sklearn.model_selection import train_test_split
    ax.scatter(y=pred, x=truth, zorder=2.6, s=20, alpha=0.5)
    props = {'boxstyle':'round', 'facecolor':'wheat'}
    ax.text(x=0.05, y=0.8, transform=ax.transAxes,
            s=f'$R^2$={np.round(r2_score(pred, truth),2)}, ' +\
                f'MAE={np.round(median_absolute_error(pred, truth),3)}, ' +\
                f'RMSE={np.round( np.sqrt(mean_squared_error(pred, truth)), 2)}',
            bbox=props)

    # Change axis formatting
    prettify_plot(ax)
    add_121_line(ax)
    ax.set_title(title)
    ax.set_xlabel('Truth')
    ax.set_ylabel('Prediction')


def prediction_within_ci(test_mean, pred_mean, pred_var, ci=0.95):
    from scipy import stats
    lower, upper = stats.norm.interval(ci, loc=pred_mean, scale=np.sqrt(pred_var))
    within = (upper > test_mean) & (lower < test_mean)
    return lower, upper, within


def validation_plot(test_mean, pred_mean, pred_var, figsize=(7, 7), minx=None, miny=None, maxx=None, maxy=None, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)

    # Deal with input arrays that might be masked
    if isinstance(test_mean, np.ma.MaskedArray):
        test_mean, pred_mean, pred_var = test_mean[~test_mean.mask], pred_mean[~test_mean.mask], pred_var[~test_mean.mask]

    lower, upper, within_95_ci = prediction_within_ci(test_mean, pred_mean, pred_var)
    valid_points = test_mean.shape[0]

    print("Proportion of 'Bad' estimates : {:.2f}%".format(((~within_95_ci).sum()/valid_points)*100.))

    col = ['r' if b else "k" for b in ~within_95_ci]

    # There's no way to set individual colors for errorbar points...
    #     Pull out the lines and set those, but do the points separately
    connector, caplines, (vertical_lines,) = ax.errorbar(test_mean, pred_mean, fmt='none',
                                                         yerr=np.asarray([pred_mean - lower, upper - pred_mean]))
    ax.scatter(test_mean, pred_mean, c=col)

    vertical_lines.set_color(col)

    ax.set_xlabel("Model")
    ax.set_ylabel("Emulator")
    add_121_line(ax)

    minx = minx if minx is not None else test_mean.min() - 0.05
    maxx = maxx if maxx is not None else test_mean.max() + 0.05
    miny = miny if miny is not None else lower.min() - 0.05
    maxy = maxy if maxy is not None else upper.max() + 0.05

    ax.set_xlim([minx, maxx])
    ax.set_ylim([miny, maxy])


def validation_plot_bastos(X_test, Y_test, m_test, v_test):
    """
    Validation plot following Bastos and O'Hagan (2009)

    Source:
        Bastos and O'Hagan (2009): Diagnostics for Gaussian Process Emulators, Technometrics,
        51, 425-438. https://doi.org/10.1198/TECH.2009.08019
        Code for the lower-right plot adapted from the ESEm package validation_plot() (see above)

    Author:
        Ulrike Proske (ulrike.proske@env.ethz.ch)


    Parameters
    ----------
    X_test : array-like of shape (n_samples, n_features)
            Input data 
    Y_test : array-like of shape (n_samples,)
            Simulated output
    m_test : array-like of shape (n_samples, n_features)
            Emulator output
    v_test : array-like of shape (n_samples,)
            Variance of emulator

    """

    import matplotlib.pyplot as plt
    from statsmodels.compat.python import lzip
    import statsmodels.api as sm
    from scipy import stats

    # Namelist
    c_black = 'black'
    c_blue = '#1f78b4'
    c_green = '#33a02c'
    c_orange = '#ff7f00'
    c_purple = '#6a3d9a'
    colors = [c_blue, c_green, c_orange, c_purple]
    alpha = 0.75

    # Start plotting
    _, axs = plt.subplots(nrows=2, ncols=2, figsize=(4.5,4.5),
                          gridspec_kw={'hspace': 0.35, 'wspace': 0.75})
    errors_std = (Y_test - m_test)/np.sqrt(v_test) # standardized errors
    axs[0, 0].scatter(m_test, errors_std, c=c_black, marker='.', alpha=alpha)
    axs[0, 0].set_xlabel(r'$Y_{\mathrm{emu}}$')
    axs[0, 0].set_ylabel(r'$({Y_{\mathrm{sim}} - Y_{\mathrm{emu}})}/{\sqrt{V}}$')
    # customize qq plot
    pp = sm.ProbPlot(errors_std.ravel(), stats.t, fit=True)
    qq_plot = pp.qqplot(marker='.', markerfacecolor='k',
                        markeredgecolor='k', alpha=alpha, ax=axs[1, 0])
    end_pts = lzip(axs[1, 0].get_xlim(), axs[1, 0].get_ylim())
    sm.qqline(qq_plot.axes[2], line='45', fmt='k--')
    axs[1, 0].set_xlim([end_pts[0][0], end_pts[1][0]])
    axs[1, 0].set_ylim([end_pts[0][1], end_pts[1][1]])
    axs[1, 0].set_ylabel('Standardized quantiles')

    for i in range(0, np.shape(X_test)[1]):
        # Slightly convoluted way to expand the parameters to match the shape of the outputs
        expanded_params = np.broadcast_to(np.expand_dims(X_test.to_numpy()[:, i], axis=[i for i in range(1, len(errors_std.shape))]),
                                          errors_std.shape)
        if isinstance(X_test, pd.DataFrame):
            axs[0, 1].scatter(expanded_params, errors_std, c=colors[i], label=X_test.columns[i], marker='.', alpha=alpha)
        else:
            axs[0, 1].scatter(expanded_params, errors_std, c=colors[i], label=str(i), marker='.', alpha=alpha)

    axs[0, 1].legend()
    axs[0, 1].set_xlabel(r'$\eta_i$')
    axs[0, 1].set_ylabel(r'$({Y_{\mathrm{sim}} - Y_{\mathrm{emu}})}/{\sqrt{V}}$')

    # add hlines
    axs[0, 1].axhline(y=-2, c=c_black, linestyle='--')
    axs[0, 1].axhline(y=2, c=c_black, linestyle='--')
    axs[0, 1].axhline(y=0, c=c_black, linestyle='--')
    axs[0, 0].axhline(y=-2, c=c_black, linestyle='--')
    axs[0, 0].axhline(y=2, c=c_black, linestyle='--')
    axs[0, 0].axhline(y=0, c=c_black, linestyle='--')

    validation_plot(Y_test, m_test, v_test, ax=axs[1, 1])


def plot_parameter_space(df, nbins=100, target_df=None, smooth=True,
                         xmins=None, xmaxs=None, fig_size=(8, 6)):
    from itertools import repeat
    import matplotlib.pyplot as plt

    def get_dist_bins(x, xmin, xmax, nbins, *args, **kwargs):
        vals, bins = np.histogram(x, *args, range=(xmin, xmax), bins=nbins, density=True, **kwargs)
        return vals, bins

    def get_dist_kde(x, xmin, xmax, nbins, *args, **kwargs):
        from scipy.stats import gaussian_kde
        bins = np.linspace(xmin, xmax, nbins)
        density = gaussian_kde(x[np.isfinite(x)], *args, **kwargs)
        return density(bins), bins

    get_dist = get_dist_kde if smooth else get_dist_bins

    fig, axes = plt.subplots(nrows=1, ncols=df.shape[1], figsize=fig_size)

    xmins = repeat(0.) if xmins is None else xmins
    xmaxs = repeat(1.) if xmaxs is None else xmaxs

    for param, ax, xmin, xmax in zip(df, axes, xmins, xmaxs):
        vals, bins = get_dist(df[param], xmin, xmax, nbins)

        X, Y = np.meshgrid(np.arange(2), bins)
        ax.pcolor(X, Y, vals[:, np.newaxis], vmin=0, vmax=1)
        if target_df is not None:
            ax.plot([0, 1], [target_df[param], target_df[param]], c='r')
        ax.set_xticks([])
        ax.set_xticklabels('')
        ax.set_xlabel(param, rotation=90)

    for ax in axes[1:]:
        ax.set_yticks([])
        ax.set_yticklabels('')


def get_uniform_params(n_params, n_samples=5):
    """
    Slightly convoluted method for getting a flat set of points evenly
     sampling a (unit) N-dimensional space

    Parameters
    ----------
    n_params: int
        The number of parameters (dimensions) to sample from
    n_samples: int
        The number of uniformly spaced samples (in each dimension)

    Returns
    -------
    ndarray
        n_samples**n_params parameters uniformly sampled
    """
    return np.stack([*np.meshgrid(*[np.linspace(0., 1., n_samples)]*n_params)]).reshape(-1, n_params)


def get_random_params(n_params, n_samples=5):
    """
         Get points randomly sampling a (unit) N-dimensional space

    Parameters
    ----------
    n_params: int
        The number of parameters (dimensions) to sample from
    n_samples: int
        The number of parameters to (radnomly) sample
    Returns
    -------

    """
    return np.random.uniform(size=n_params*n_samples).reshape(n_samples, n_params)


def ensemble_collocate(ensemble, observations, member_dimension='job'):
    """
     Efficiently collocate (interpolate) many ensemble members on to a set of (un-gridded) observations

    Note
    ----
    This function requires both Iris and CIS to be installed

    Parameters
    ----------
    ensemble: ~cis.data_io.gridded_data.GriddedData
        The ensemble of (model) samples to interpolate on to the observations
    observations: ~cis.data_io.ungridded_data.UngriddedData
        The observations on to which the observations will be sampled
    member_dimension: str
        The name of the dimension which represents the ensemble members in `ensemble`

    Returns
    -------
    col_ensemble: iris.cube.Cube
        The ensemble values interpolated on to the observation locations, with the ensemble members
        along the leading dimension.
    """
    from iris.cube import Cube, CubeList
    from iris.coords import DimCoord, AuxCoord
    from cis.collocation.col_implementations import GriddedUngriddedCollocator, DummyConstraint
    from cis.data_io.gridded_data import make_from_cube

    col = GriddedUngriddedCollocator(missing_data_for_missing_sample=False)
    col_members = CubeList()

    for member in ensemble.slices_over(member_dimension):
        # Use CIS to collocate each ensemble member on to the observations
        #  The interpolation weights are cached within col automatically
        collocated_job, = col.collocate(observations, make_from_cube(member), DummyConstraint(), 'lin')
        # Turn the interpolated data in to a flat cube for easy stacking
        new_c = Cube(collocated_job.data.reshape(1, -1), long_name=collocated_job.name(), units='1',
                     dim_coords_and_dims=[(DimCoord(np.arange(collocated_job.data.shape[0]), long_name="obs"), 1),
                                          (DimCoord(member.coord(member_dimension).points, long_name=member_dimension), 0)],
                     aux_coords_and_dims=[(AuxCoord(c.points, standard_name=c.standard_name), 1) for c in collocated_job.coords()])
        col_members.append(new_c)
    col_ensemble = col_members.concatenate_cube()
    return col_ensemble


def leave_one_out(Xdata, Ydata, model='RandomForest', **model_kwargs):
    """
    Function to perform LeaveOneOut cross-validation with different models. 
    
    Parameters
    ----------
    Xdata : array-like of shape (n_samples, n_features)
            Parameter values
    Ydata : array-like of shape (n_samples,)
            Target values.
    model: {'RandomForest', 'GaussianProcess', 'NeuralNet'}, default='RandomForest'
    model_kwargs: dict
            More arguments to pass to the model.
            
    Returns
    ----------
    output: list of n_samples (truth, prediction, variance) tuples 
            which can then be passed to esem.utils.validation_plot()
    """
    from esem import rf_model
    from esem import gp_model
    from esem import cnn_model
    
    models = {'RandomForest': rf_model,
              'GaussianProcess': gp_model,
              'NeuralNet': cnn_model}
    
    if model not in models.keys():
        raise Exception(f"Model needs to be one of {list(models.keys())}, found '{model}'.")

    # Ensure the x data is an array
    Xdata = np.asarray(Xdata)

    # Output list for test value and prediction outputs
    output = []

    indices = np.arange(Xdata.shape[0])
    for test_idx in indices:
        # Split into training - validation sets
        X_train = Xdata[indices != test_idx, :]
        X_test = Xdata[test_idx:test_idx+1, :]
        
        Y_train = Ydata[indices != test_idx]
        Y_test = Ydata[test_idx]

        # Construct and fit model
        model_ = models[model](training_params=X_train, 
                               training_data=Y_train, 
                               **model_kwargs)
        model_.train()

        # Evaluate model on test data
        predictions, v = model_.predict(X_test)

        # Save output for validation plot later on
        output.append((Y_test, predictions, v))

    return output


class tf_tqdm(object):
    """
    A progress bar suitable for reporting on progress iterating over a TF dataset
    """
    def __init__(self, unit='sample', batch_size=1, total=None):
        import io
        self.unit = unit
        self.batch_size = batch_size
        self.total = total
        self.bar = tqdm(file=io.StringIO(), unit=unit, total=int(total))

    def update(self):
        self.bar.update(self.batch_size)
        # Print the status update manually.
        print('\r', end='')
        print(repr(self.bar), end='')

    def __call__(self, ds, *args, **kwargs):
        def advance_tqdm(e):
            tf.py_function(self.update, [], [])
            return e

        return ds.map(advance_tqdm)


def get_param_mask(X, y, criterion='bic', **kwargs):
    """
    Determine the most relevant parameters in the input space using a regularised linear model and either the
    Aikake or Baysian Information Criterion.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Parameter values
    y : array-like of shape (n_samples,)
        target values.
    criterion: {'bic' , 'aic'}, default='bic'
        The information criteria to apply for parameter selection. Either Aikake or Baysian Information Criterion.
    kwargs: dict
        Further arguments for sklearn.feature_selection.SelectFromModel

    Returns
    -------
    mask : ndarray
        A boolean array of shape [# input features], in which an element is
        True iff its corresponding feature is selected for retention.
    """
    from sklearn.linear_model import LassoLarsIC
    from sklearn.feature_selection import SelectFromModel

    lsvc = LassoLarsIC(criterion=criterion).fit(X, y)
    model = SelectFromModel(lsvc, prefit=True, **kwargs)
    return model.get_support()
