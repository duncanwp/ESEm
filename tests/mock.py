import numpy as np


def make_dummy_1d_cube(job_n=0):
    """
    Makes a dummy 1d cube filled with dummy data. It has a scalar job coordinate
    to make ensemble stacking convenient
    """
    from iris.cube import Cube
    from iris.coords import DimCoord

    x = np.arange(100)
    y = np.sin(x) * np.cos(x + 0.3)

    obs = DimCoord(x, var_name='obs')
    job = DimCoord(job_n, var_name='job')
    cube = Cube(y.reshape((1, 100)), dim_coords_and_dims=[(job, 0), (obs, 1)])

    return cube


def make_dummy_2d_cube(job_n=0):
    """
    Makes a dummy 2d cube filled with dummy data. It has a realistic lat and lon
    coordinates and a scalar job coordinate to make ensemble stacking convenient
    """
    from iris.cube import Cube
    from iris.coords import DimCoord

    y = np.linspace(-90., 90., 20)
    x = np.linspace(0., 360., 30, endpoint=False)
    xx, yy = np.meshgrid(x, y)

    data = np.sin(np.deg2rad(xx)) * np.cos(np.deg2rad(yy))

    latitude = DimCoord(y, standard_name='latitude', units='degrees')
    longitude = DimCoord(x, standard_name='longitude', units='degrees', circular=True)
    job = DimCoord(job_n, var_name='job')

    cube = Cube(data.reshape(1, 20, 30),
                dim_coords_and_dims=[(job, 0), (latitude, 1), (longitude, 2)])

    return cube


def simple_polynomial_fn_two_param(x, y, a=1., b=1., x0=0., y0=0.):
    return 1. + a*(x-x0)**2 + b*(y-y0)**3


def simple_polynomial_fn_three_param(x, y, z, a=1., b=1., c=2., x0=0., y0=0., z0=0.):
    return 1. + a*(x-x0)**2 + b*(y-y0)**3 + c*(z-z0)**3


def get_uniform_params(n_params, n_samples=5):
    # Slightly convoluted method for getting a flat set of points evenly
    # sampling a (unit) N-dimensional space
    return np.stack([*np.meshgrid(*[np.linspace(0., 1., n_samples)]*n_params)]).reshape(-1, n_params)


def get_random_params(n_params, n_samples=5):
    # Slightly convoluted method for getting a flat set of points evenly
    # sampling a (unit) N-dimensional space
    return np.random.uniform(size=n_params*n_samples).reshape(n_samples, n_params)


def pop_elements(params, idx1, idx2=None):
    """
    Select a specific parameter index to remove
    """
    idx2 = idx2 if idx2 is not None else idx1
    popped = params[idx1:idx2+1]
    new_params = np.vstack((params[:idx1], params[idx2+1:]))
    return new_params, popped


def get_1d_two_param_cube(params=None, n_samples=10):
    """
    Create an ensemble of 1d cubes perturbed over two idealised parameter
    spaces. One of params or n_samples must be provided
    :param np.array params: A list of params to sample the ensemble over
    :param int n_samples: The number of params to sample (between 0. and 1.)
    :return:
    """
    from iris.cube import CubeList

    if params is None:
        params = np.linspace(np.zeros((2,)), np.ones((2,)), n_samples)

    cubes = CubeList([])
    for j, p in enumerate(params):
        c = make_dummy_1d_cube(j)
        # Perturb base data to represent some change in a parameter
        c.data *= simple_polynomial_fn_two_param(*p)
        cubes.append(c)

    ensemble = cubes.concatenate_cube()
    return ensemble


def get_2d_three_param_cube(params=None, n_samples=10):
    """
    Create an ensemble of 2d cubes perturbed over three idealised parameter
    spaces. One of params or n_samples must be provided
    :param np.array params: A list of params to sample the ensemble over
    :param int n_samples: The number of params to sample (between 0. and 1.)
    :return:
    """
    from iris.cube import CubeList

    if params is None:
        params = np.linspace(np.zeros((3,)), np.ones((3,)), n_samples)

    cubes = CubeList([])
    for j, p in enumerate(params):
        c = make_dummy_2d_cube(j)
        # Perturb base data to represent some change in a parameter
        c.data *= simple_polynomial_fn_three_param(*p)
        cubes.append(c)

    ensemble = cubes.concatenate_cube()
    return ensemble


def eval_1d_cube(params, **kwargs):
    """
    Create a single 1D cube representing the 'true' value for a given parameter

    :param np.array params: A set of params to represent the 'truth'
    :return:
    """
    cube = make_dummy_1d_cube(**kwargs)
    # Scale the base data
    cube.data *= simple_polynomial_fn_two_param(*params)

    return cube


def eval_2d_cube(params, **kwargs):
    """
    Create a single 2D cube representing the 'true' value for a given parameter

    :param np.array params: A set of params to represent the 'truth'
    :return:
    """
    cube = make_dummy_2d_cube(**kwargs)
    # Scale the base data
    cube.data *= simple_polynomial_fn_three_param(*params)

    return cube
