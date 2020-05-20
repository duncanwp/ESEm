import numpy as np


def get_white_cube(cube, ref_mean=None, ref_std=None):
    """
        Whiten a cube

    :param iris.cubes.Cube cube: The input data
    :param float ref_mean: The mean of the data (will calculate the mean directly if not provided)
    :param float ref_std: The standard deviation of the data (will calculated the standard deviation directly if not
    provided)
    :return iris.cubes.Cube: A copy of the cube with zero mean and unit standard deviation
    """
    ref_mean = ref_mean or cube.data.mean()
    ref_std = ref_std or cube.data.std()
    return cube.copy(data=(cube.data - ref_mean) / ref_std)


def get_un_white_cube(data, ref_mean, ref_std):
    """
        Un-whiten a cube

    :param iris.cubes.Cube cube: The input data
    :param float ref_mean: The mean of the original data
    :param float ref_std: The standard deviation of the original data
    :return iris.cubes.Cube: A copy of the cube with its original mean and standard deviation
    """
    # TODO should this return an actual Cube?
    return (data * ref_std) + ref_mean


def extract_cube(cubelist, *args, **kwargs):
    import iris.cube
    res = iris.cube.CubeList(cubelist).extract(*args, **kwargs)
    assert len(res) == 1, "Expected single cube but got {}".format(len(res))
    return res[0]


def ensure_bounds(cube):
    if not cube.coord("latitude").has_bounds():
        cube.coord("latitude").guess_bounds()
    if not cube.coord("longitude").has_bounds():
        cube.coord("longitude").guess_bounds()


def get_weights(cube):
    import iris.analysis
    ensure_bounds(cube)

    return iris.analysis.cartography.area_weights(cube[0, :, :, :], normalize=True)


def get_param_mask(X, y, **kwargs):
    from sklearn.linear_model import LassoLarsIC
    from sklearn.feature_selection import SelectFromModel

    lsvc = LassoLarsIC(criterion='bic').fit(X, y)
    model = SelectFromModel(lsvc, prefit=True, **kwargs)
    return np.where(model.get_support())[0]


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

