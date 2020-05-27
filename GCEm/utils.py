import numpy as np
import tensorflow as tf
from tqdm import tqdm


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


def validation_plot(test_mean, pred_mean, pred_var, figsize=(7, 7), minx=None, miny=None, maxx=None, maxy=None):
    from scipy import stats
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, figsize=figsize)
    lower, upper = stats.norm.interval(0.95, loc=pred_mean, scale=np.sqrt(pred_var))
    bad = (upper < test_mean) | (lower > test_mean)
    print("Proportion of 'Bad' estimates : {:.2f}%".format((bad.sum()/(~bad).sum())*100.))
    col = ['r' if b else "k" for b in bad]

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
        ax.set_xticks([], [])
        ax.set_xticklabels('')
        ax.set_xlabel(param, rotation=90)

    for ax in axes[1:]:
        ax.set_yticks([], [])
        ax.set_yticklabels('')


def get_uniform_params(n_params, n_samples=5):
    """
    Slightly convoluted method for getting a flat set of points evenly
     sampling a (unit) N-dimensional space

    :param int n_params: The number of parameters (dimensions) to sample from
    :param int n_samples: The number of uniformly spaced samples (in each dimension)
    :return np.array: n_samples**n_params parameters uniformly sampled
    """
    return np.stack([*np.meshgrid(*[np.linspace(0., 1., n_samples)]*n_params)]).reshape(-1, n_params)


def get_random_params(n_params, n_samples=5):
    """
     Get points randomly sampling a (unit) N-dimensional space

    :param int n_params: The number of parameters (dimensions) to sample from
    :param int n_samples: The number of parameters to (radnomly) sample
    :return np.array:
    """
    return np.random.uniform(size=n_params*n_samples).reshape(n_samples, n_params)


def ensemble_collocate(ensemble, observations, member_dimension='job'):
    """
     Efficiently collocate many ensemble members on to a set of (un-gridded) observations

    :param GriddedData ensemble:
    :param UngriddedData observations:
    :param str member_dimension:
    :return:
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
