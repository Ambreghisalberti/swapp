import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

maxfev = 80000

def fit_cold_ions(df, energies, **kwargs):
    columns = ['max_coldions', 'center_coldions', 'std_coldions']
    for indice in range(len(df)):
        y = df.iloc[indice][[f'spectro_{i}' for i in range(32)]].values.astype('float')
        params = fit_cold_ions_one_time(energies, y, max_energy = kwargs.get('max_energy', 100))
        df.loc[df.iloc[indice].name, columns] = params
        if (indice % kwargs.get('freq_verbose', 1000) == 0) and kwargs.get('verbose', False):
            print(f'Cold ions fitted until index {indice} out of {len(df)}!')
    return df


def fit_cold_ions_one_time(x, y, **kwargs):
    max_energy = kwargs.get('max_energy', 100)
    max_cold, center_cold, std_cold, err = fit_populations(x, y, 0, max_energy, **kwargs)
    if (center_cold <= x[x < max_energy][0]) or (center_cold >= x[x < max_energy][-1]):
        max_cold, center_cold, std_cold = 0, -1, -1
    if std_cold > max_energy:
        max_cold, center_cold, std_cold = 0, -1, -1
    return max_cold, center_cold, std_cold


def fit_populations(x, y, min_energy, max_energy, **kwargs):
    verbose = kwargs.get('verbose', False)
    if verbose:
        if 'fig' in kwargs and 'ax' in kwargs:
            fig, ax = kwargs['fig'], kwargs['ax']
        else:
            fig, ax = plt.subplots()
        ax.plot(x, y, '+', label='Data')
        ax.set_xscale('log')
        ax.axvline(min_energy, linestyle='--', color='red', alpha=0.5)
        ax.axvline(max_energy, linestyle='--', color='red', alpha=0.5)
        ax.set_xlabel('Energies')
    else:
        fig, ax = 0, [0, 0, 0, 0]

    x_popu = x[(x >= min_energy) & (x <= max_energy)]
    y_popu = y[(x >= min_energy) & (x <= max_energy)]

    popt_popu = fit_one_gaussian(x_popu, y_popu)
    err = compute_relative_error(x_popu, y_popu, popt_popu, gaussian)

    maximum, center, std = retrieve_characteristics_from_parameters(*popt_popu, x_popu)
    if verbose:
        ax.plot(x_popu, gaussian(x_popu, *popt_popu), label='Fit')
        ax.legend()
        fig.show()

    return np.array([maximum, center, std, err])


def gaussian(x, a, b, c):
    return a ** 2 * np.sqrt(x) * np.exp(-(x + b * np.sqrt(x)) / c)

def fit(func, x_data, y_data, guess, **kwargs):
    count = 0
    err = 1
    while (count < 10) & (err > 0.05):  # The fit did not raise an error, but it didn't work well
        popt, _ = curve_fit(func, x_data, y_data, p0=guess, **kwargs)
        err = abs((y_data - func(x_data, *popt))).sum() / y_data.sum()
        count += 1
        # if err > 0.2:
        # print(f'Optimization done but not good enough: n°{count}')
    return popt

def fit_one_gaussian_guess(x, y, guess):
    try:
        popt = fit(gaussian, x, y, guess, maxfev=maxfev)
    except:
        popt = [0, 0, 1]
    return popt


def fit_one_gaussian(x, y):
    # guess = [80,-100,1000]
    guess = [15, -150, 1000]
    popt = fit_one_gaussian_guess(x, y, guess)
    err = compute_relative_error(x, y, popt, gaussian)
    if err < 0.2:
        return popt

    guess = [1000, -20, 1000]
    popt2 = fit_one_gaussian_guess(x, y, guess)
    err2 = compute_relative_error(x, y, popt2, gaussian)
    if err2 < 0.2:
        return popt2

    if err > err2:
        err, err2, popt, popt2 = err2, err, popt2, popt  # Err has the smallest error between popt and popt2

    guess = [150, -10, 10]
    popt3 = fit_one_gaussian_guess(x, y, guess)
    err3 = compute_relative_error(x, y, popt3, gaussian)
    if err3 < 0.2:
        return popt3

    if err < err3:
        return popt
    else:
        return popt3


def retrieve_characteristics_from_parameters(a, b, c, x):
    y = gaussian(x, a, b, c)
    maximum = np.max(y)
    # center = (x*y).sum()/y.sum()
    center = x[np.argmax(y)]
    sigma = np.sqrt((y * (x - center) ** 2).sum() / y.sum())

    return maximum, center, sigma


def compute_relative_error(x, y, popt, func):
    integral_y = ((y[1:] + y[:-1]) * (np.log(x[1:]) - np.log(x[:-1])) / 2).sum()
    diff = abs(func(x, *popt) - y)
    integral_diff = ((diff[1:] + diff[:-1]) * (np.log(x[1:]) - np.log(x[:-1])) / 2).sum()
    err = integral_diff / integral_y
    return err
