import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from swapp.engineered_features.gaussian_fits import fit_one_population


def concatenate_files(len_data, size_file, columns, description='', path='/home/ghisalberti/make_datasets/'):
    if description != '':
        description = '_' + description

    warnings.filterwarnings("ignore")
    gaussian_fit = pd.DataFrame([], columns=columns)

    i = 0
    for i in range(len_data // size_file):
        i0, i1 = size_file * i, size_file * (i + 1) - 1
        gaussian_fit = pd.concat([gaussian_fit, pd.read_pickle(path + f'{i0}_{i1}' + description + '.pkl')])
        print(i)
    i0, i1 = len_data // size_file * size_file, len_data - 1
    gaussian_fit = pd.concat([gaussian_fit, pd.read_pickle(path + f'{i0}_{i1}' + description + '.pkl')])
    print(i + 1)

    gaussian_fit.to_pickle(path + 'MMS1_gaussian_fit' + description + '.pkl')
    print('Files are concatenated.')


def fit_populations_by_energy_range(x, y, verbose=False):
    if verbose:
        fig, ax = plt.subplots(ncols=4, figsize=(20, 5))
    else:
        fig, ax = 0, [0, 0, 0, 0]
    max_cold, center_cold, std_cold = fit_one_population(x, y, 0, 150, verbose=verbose, fig=fig, ax=ax[0])
    # max_msh, center_msh   = fit_one_population(x, y, 80, 6000, verbose=verbose, fig=fig, ax=ax[1])
    max_msh, center_msh, std_msh = fit_one_population(x, y, 70, 800, verbose=verbose, fig=fig, ax=ax[1])
    # max_msp, center_msp   = fit_one_population(x, y, 6000, np.max(x), verbose=verbose, fig=fig, ax=ax[2])
    max_msp, center_msp, std_msp = fit_one_population(x, y, 800, np.max(x), verbose=verbose, fig=fig, ax=ax[2])
    if max_msp + max_msh == 0:
        max_msh, center_msh, std_msh = fit_one_population(x, y, 300, 3000, verbose=verbose, fig=fig, ax=ax[3])

    return np.array([max_cold, center_cold, std_cold, max_msh, center_msh, std_msh, max_msp, center_msp, std_msp])
