# %%

import pandas as pd
import numpy as np
import iris
import iris.quickplot as qplt

from keras.layers import Dense, Input,  Reshape, Conv2DTranspose
from keras.models import Model
from keras.utils import Sequence
from keras.optimizers import Adam, RMSprop

from cartopy.util import add_cyclic_point

# TODO - do I really need all of sklearn just for this?
from sklearn import preprocessing
import os

import matplotlib.pyplot as plt


class IrisSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size=1):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        res = batch_x, add_cyclic_point(batch_y.data)
        return res


def validation_plots(model, n_jobs, month=None):
    month = month or slice(None)
    # Create a copy of the zeroth job cube, predict the first test cube and then pop off the job dimension again
    pred = cube[:n_jobs, ].copy(decoder.predict(X_TEST[:n_jobs, ...])[..., :-1]) * (
                forcing_max - forcing_min) + forcing_min

    for job in range(n_jobs):
        fig = plt.figure(figsize=(12, 5))
        plt.subplot(121)
        qplt.pcolormesh(pred[job, month, ...].collapsed('t', iris.analysis.MEAN), vmin=-10., vmax=10., cmap='RdBu_r')
        plt.gca().set_title(
            "Predicted ({:.2f})".format(pred[job, month, ...].collapsed(pred.coords(), iris.analysis.MEAN).data))
        plt.gca().coastlines()

        ax = plt.subplot(122)
        t = (test_cube * (forcing_max - forcing_min) + forcing_min)[job, month, ...]
        qplt.pcolormesh(t.collapsed('t', iris.analysis.MEAN), vmin=-10., vmax=10., cmap='RdBu_r')
        plt.gca().set_title("Truth ({:.2f})".format(t.collapsed(pred.coords(), iris.analysis.MEAN).data))
        plt.gca().coastlines()


def fit_and_validate(model, n_plots=0, month=None):
    month = month or slice(None)
    model.fit_generator(IrisSequence(X, train_cube[...], batch_size), epochs=epochs,
                        validation_data=IrisSequence(X_TEST, test_cube[...], 1))
    if n_plots > 0:
        validation_plots(model, n_plots, month)

    pred = rescale(model.predict(X_TEST)[:, month, :, :-1])
    mse = (pred - rescale(test_cube.data)) ** 2
    rmse_per_val = pd.DataFrame(np.sqrt(mse.mean(axis=(1, 2, 3))))
    print(rmse_per_val.describe())
    return rmse_per_val


# %%

project_path = "/home/ubuntu/A-CURE-project"
unit_path = project_path + "/PPE_Unit.csv"

N_PARAMS = 26
N_TEST = 183

if os.path.isfile(unit_path):
    all_data = pd.read_csv(unit_path, index_col=0)
    coordinate_cols = all_data.columns[:N_PARAMS]
else:
    all_data = pd.read_csv(project_path + "/data/PPE.csv", index_col=0)

    coordinate_cols = all_data.columns[:N_PARAMS]
    all_data[coordinate_cols] = preprocessing.scale(all_data[coordinate_cols])

    all_data.to_csv(project_path + "/data/PPE_Unit.csv")

# latent_cols = all_data.columns[26:]
latent_cols = all_data.columns[N_PARAMS:N_PARAMS + 1]

# train, test = train_test_split(all_data, test_size=0.2)
train = all_data[:N_TEST]
test = all_data[N_TEST + 1:]

X = train[coordinate_cols].values
Y = train[latent_cols].values

X_TEST = test[coordinate_cols].values
Y_TEST = test[latent_cols].values


# %%

def load_callback(cube, field, fname):
    cube.attributes = None
    #     cube.attributes
    cube.add_dim_coord(iris.coords.DimCoord(np.arange(235), var_name='job'), (0,))


# Load a single merged cube for psuedo_level=2
cube = iris.util.squeeze(
    iris.load(project_path + "/data/ACI_pm2008???_N48.nc", callback=load_callback).concatenate_cube())

# Whiten the forcing
forcing_std_dev = np.std(cube.data, axis=(1, 2, 3), keepdims=True)
forcing_mean = np.mean(cube.data, axis=(1, 2, 3), keepdims=True)
normalised_input = (cube.data - forcing_mean) / forcing_std_dev

# Normalise the forcing
# forcing_min = np.min(cube.data, axis=(1,2,3), keepdims=True)
# forcing_max = np.max(cube.data, axis=(1,2,3), keepdims=True)
forcing_min = np.min(cube.data, axis=0, keepdims=True)
forcing_max = np.max(cube.data, axis=0, keepdims=True)


def scale(arr):
    return (arr - forcing_min) / (forcing_max - forcing_min)


def rescale(arr):
    return arr * (forcing_max - forcing_min) + forcing_min


normal_cube = cube.copy(data=scale(cube.data))

print(normal_cube)

# train, test = train_test_split(all_data, test_size=0.2)
train_cube = normal_cube[:N_TEST]
test_cube = normal_cube[N_TEST + 1:]

# %%

# network parameters
input_shape = (12, 73, 97)
batch_size = 8
kernel_size = (2, 3)
filters = 12
latent_dim = N_PARAMS
epochs = 30

learning_rate = 1e-5
decay_rate = 0.01

# %%

# build decoder model
latent_inputs = Input(shape=(N_PARAMS,), name='params')

shape = (None, 3, (73 // 1), (97 // 1))

x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

x = Conv2DTranspose(filters=6, input_shape=(12, 73, 97),
                    kernel_size=kernel_size,
                    activation='relu',
                    strides=1,
                    padding='same', data_format='channels_first')(x)

outputs = Conv2DTranspose(filters=filters, input_shape=(12, 73, 97),
                          kernel_size=kernel_size,
                          activation='relu',
                          strides=1,
                          padding='same', data_format='channels_first')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')

# decoder.compile(optimizer=Adam(lr=learning_rate, decay=decay_rate), loss='binary_crossentropy')
# decoder.compile(optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.01), loss='binary_crossentropy')
decoder.compile(optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.01), loss='mean_absolute_error')
decoder.summary()

fit_and_validate(decoder, n_plots=10)

# %%

# build decoder model
latent_inputs = Input(shape=(N_PARAMS,), name='params')

shape = (None, 3, (73 // 1), (97 // 1))

x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

x = Conv2DTranspose(filters=6, input_shape=(12, 73, 97),
                    kernel_size=kernel_size,
                    activation='relu',
                    strides=1,
                    padding='same', data_format='channels_first')(x)

outputs = Conv2DTranspose(filters=filters, input_shape=(12, 73, 97),
                          kernel_size=kernel_size,
                          activation='relu',
                          strides=1,
                          padding='same', data_format='channels_first')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')

decoder.compile(optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.01), loss='binary_crossentropy')
decoder.summary()

_ = fit_and_validate(decoder, n_plots=10)

# %%

# build decoder model
latent_inputs = Input(shape=(N_PARAMS,), name='params')

shape = (None, 3, (73 // 1), (97 // 1))

x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

x = Conv2DTranspose(filters=6, input_shape=(12, 73, 97),
                    kernel_size=kernel_size,
                    activation='relu',
                    strides=1,
                    padding='same', data_format='channels_first')(x)

outputs = Conv2DTranspose(filters=filters, input_shape=(12, 73, 97),
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          strides=1,
                          padding='same', data_format='channels_first')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')

decoder.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy')
decoder.summary()

_ = fit_and_validate(decoder, n_plots=10)

# %% md

# July stats

# %%

# build decoder model
latent_inputs = Input(shape=(N_PARAMS,), name='params')

shape = (None, 3, (73 // 1), (97 // 1))

x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

x = Conv2DTranspose(filters=6, input_shape=(12, 73, 97),
                    kernel_size=kernel_size,
                    activation='relu',
                    strides=1,
                    padding='same', data_format='channels_first')(x)

outputs = Conv2DTranspose(filters=filters, input_shape=(12, 73, 97),
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          strides=1,
                          padding='same', data_format='channels_first')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')

decoder.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy')
decoder.summary()

_ = fit_and_validate(decoder, n_plots=10, month=slice(6, 7))

# %%


