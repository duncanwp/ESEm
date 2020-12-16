"""
===================================================================
July 2020 --- Andrew Williams
===================================================================
Utility functions for plotting and analysis
===================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    
    
def LeaveOneOut(Xdata, Ydata, model='RandomForest', rndseed=0, **rf_kwargs):
    """
    Function to perform LeaveOneOut cross-validation with different models. 
    """
    from GCEm.rf_model import RFModel
    from GCEm.gp_model import GPModel
    from GCEm.nn_model import NNModel
    from sklearn.linear_model import LinearRegression
    
    estimators = {'Linear': LinearRegression(),
                  'RandomForest': RFModel,
                  'GaussianProcess': GPModel,
                  'NeuralNet': NNModel}
    
    if model not in estimators.keys():
        raise Exception(f"Method needs to be one of {list(estimators.keys())}, found '{method}'.")
    
    # fix random seed for reproducibility 
    # then shuffle X,Y indices so that they're not ordered (in time, for example)
    np.random.seed(rndseed)
    rndperm = np.random.permutation(Xdata.shape[0])
    
    # How many indices?
    n_data = Xdata.shape[0]
    
    # Output array for test value and prediction
    output = np.vstack([np.empty(2) for _ in range(n_data)])
    
    for test_idx in range(n_data):
        """Split into training - validation sets"""
        X_train = Xdata.iloc[rndperm[np.arange(len(rndperm))!= test_idx], :]
        X_test  = Xdata.iloc[rndperm[test_idx], :].to_numpy().reshape(1,-1)
        
        Y_train = Ydata[rndperm[np.arange(len(rndperm))!= test_idx]]
        Y_test  = Ydata[rndperm[test_idx]]
       
        """Construct and fit model"""
        if model=='Linear':
            model_ = estimators[model]
            model_.fit(X=X_train, y=Y_train)
            """Evaluate model on test data"""
            predictions = model_.predict(X_test)

            """Save output for validation plot  later on"""
            output[test_idx] = (Y_test, predictions)
            
        else:
            if model=='RandomForest':
                model_ = estimators[model](training_params=X_train, 
                                           training_data=Y_train, 
                                           random_state=rndseed, 
                                           **rf_kwargs)
            else:
                model_ = estimators[model](training_params=X_train, 
                                           training_data=Y_train)
        
            model_.train()

            """Evaluate model on test data"""
            predictions,v = model_.predict(X_test)

            """Save output for validation plot later on"""
            output[test_idx] = (Y_test, predictions)

    return output

def get_crm_data(cache_path='.', preprocess=True):
    """
    Load the example cloud-resolving model data, download if not present.
    :param str cache_path: Path to load/store the data
    :param bool preprocess: Whether or not to clean and concatenate the data
    :return:
    """
    import pandas as pd
    import os
    import urllib.request

    N1_200_cache = os.path.join(cache_path, 'NARVAL1_1hr_200cdnc.csv')
    if not os.path.isfile(N1_200_cache):
        urllib.request.urlretrieve("https://zenodo.org/record/4323300/files/NARVAL1_1hr_200cdnc.csv?download=1", N1_200_cache)
        
        
    N1_20_cache = os.path.join(cache_path, 'NARVAL1_1hr_20cdnc.csv')
    if not os.path.isfile(N1_20_cache):
        urllib.request.urlretrieve("https://zenodo.org/record/4323300/files/NARVAL1_1hr_20cdnc.csv?download=1", N1_20_cache)
        
    N1_20_shal_cache = os.path.join(cache_path, 'NARVAL1_1hr_20cdnc_shal.csv')
    if not os.path.isfile(N1_20_shal_cache):
        urllib.request.urlretrieve("https://zenodo.org/record/4323300/files/NARVAL1_1hr_20cdnc_shal.csv?download=1", N1_20_shal_cache)
        
        
    N1_200_shal_cache = os.path.join(cache_path, 'NARVAL1_1hr_200cdnc_shal.csv')
    if not os.path.isfile(N1_200_shal_cache):
        urllib.request.urlretrieve("https://zenodo.org/record/4323300/files/NARVAL1_1hr_200cdnc_shal.csv?download=1", N1_200_shal_cache)

    
    if preprocess:
        df20 = pd.read_csv(N1_20_shal_cache).set_index('time').drop(columns='plev')
        df200 = pd.read_csv(N1_200_shal_cache).set_index('time').drop(columns='plev')

        new_df = pd.concat([df20, df200]).reset_index().drop(columns='time')

        return new_df
    
    else:
        df20 = pd.read_csv('NARVAL1_1hr_20cdnc_shal.csv')
        df200 = pd.read_csv('NARVAL1_1hr_200cdnc_shal.csv')
        return df20, df200