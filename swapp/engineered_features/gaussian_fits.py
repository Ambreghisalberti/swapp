import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

maxfev = 80000


# Basic functions
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


def compute_relative_error(x, y, popt, func):
    integral_y = ((y[1:] + y[:-1]) * (np.log(x[1:]) - np.log(x[:-1])) / 2).sum()
    diff = abs(func(x, *popt) - y)
    integral_diff = ((diff[1:] + diff[:-1]) * (np.log(x[1:]) - np.log(x[:-1])) / 2).sum()
    err = integral_diff / integral_y
    return err


# For one gaussian
def f(x, a, b, c):
    return a ** 2 * np.sqrt(x) * np.exp(-(x + b * np.sqrt(x)) / c)


def fit_one_gaussian_guess(x, y, guess):
    try:
        popt = fit(f, x, y, guess, maxfev=maxfev)
    except:
        popt = [0, 0, 1]
    return popt


def fit_one_gaussian(x, y):
    # guess = [80,-100,1000]
    guess = [15, -150, 1000]
    popt = fit_one_gaussian_guess(x, y, guess)
    err = compute_relative_error(x, y, popt, f)
    if err < 0.2:
        return popt

    guess = [1000, -20, 1000]
    popt2 = fit_one_gaussian_guess(x, y, guess)
    err2 = compute_relative_error(x, y, popt2, f)
    if err2 < 0.2:
        return popt2

    if err > err2:
        err, err2, popt, popt2 = err2, err, popt2, popt  # Err has the smallest error between popt and popt2

    guess = [150, -10, 10]
    popt3 = fit_one_gaussian_guess(x, y, guess)
    err3 = compute_relative_error(x, y, popt3, f)
    if err3 < 0.2:
        return popt3

    if err < err3:
        return popt
    else:
        return popt3


def retrieve_characteristics_from_parameters(a, b, c, x):
    y = f(x, a, b, c)
    maximum = np.max(y)
    # center = (x*y).sum()/y.sum()
    center = x[np.argmax(y)]
    sigma = np.sqrt((y * (x - center) ** 2).sum() / y.sum())

    return maximum, center, sigma


def plot_one_gaussian(x, y, popt, **kwargs):
    if 'fig' in kwargs and 'ax' in kwargs:
        fig, ax = kwargs['fig'], kwargs['ax']
    else:
        fig, ax = plt.subplots()
    ax.plot(x, y, '+', label='data')
    ax.plot(x, f(x, *popt), label='Fit one gaussian')
    ax.set_xlabel('Energies')
    ax.set_xscale('log')
    ax.legend()
    fig.show()


def fit_one_population(x, y, min_energy, max_energy, **kwargs):
    x_popu = x[(x >= min_energy) & (x <= max_energy)]
    y_popu = y[(x >= min_energy) & (x <= max_energy)]
    popt_popu = fit_one_gaussian(x_popu, y_popu)
    maximum, center, std = retrieve_characteristics_from_parameters(*popt_popu, x_popu)
    verbose = kwargs.get('verbose', False)
    if verbose:
        if 'fig' in kwargs and 'ax' in kwargs:
            fig, ax = kwargs['fig'], kwargs['ax']
        else:
            fig, ax = plt.subplots()
        ax.plot(x, y, '+', label='Data')
        ax.plot(x_popu, f(x_popu, *popt_popu), label='Fit')
        ax.set_xscale('log')
        ax.axvline(min_energy, linestyle='--', color='red', alpha=0.5)
        ax.axvline(max_energy, linestyle='--', color='red', alpha=0.5)
        ax.set_xlabel('Energies')
        ax.set_ylabel('Flux')
        ax.legend()
        fig.show()

    err = compute_relative_error(x_popu, y_popu, popt_popu, f)
    if (x_popu[0] != x[0]) & (err > 0.33):
        return 0, -1, -1

    if (x_popu[-1] == x[-1]) & (center <= x_popu[0]):
        return 0, -1, -1

    elif (x_popu[-1] != x[-1]) & ((center <= x_popu[0]) or (center >= x_popu[-1])):
        return 0, -1, -1

    return maximum, center, std


# For a sum of two gaussians
def g(x, a, b, c, a1, b1, c1):
    return f(x, a, b, c) + f(x, a1, b1, c1)


def fit_two_gaussians_guess(x, y, guess):
    try:
        popt = fit(g, x, y, guess, maxfev=maxfev)
    except:
        try:
            popt = fit(f, x, y, guess[:3], maxfev=maxfev)
        except:
            popt = [0, 0, 1]
        popt = list(popt) + [0, 0, 1]
    return popt


def fit_two_gaussians(x, y, verbose=False):
    guess1 = [2 * 10 ** 2, -25, 100, 7, -220, 2000]
    guess2 = [5 * 10 ** 2, -25, 100, 7, -220, 2000]
    guess3 = [10 ** 2, -25, 100, 7, -220, 2000]
    guess4 = [4 * 10 ** 2, -25, 100, 250, -100, 2000]

    popt1 = fit_two_gaussians_guess(x, y, guess1)
    err1 = compute_relative_error(x, y, popt1, g)
    if verbose:
        print(f'Err1 = {err1}')
        fig, ax = plt.subplots(ncols=4, figsize=(16, 4))
        plot_two_gaussians(x, y, popt1, fig=fig, ax=ax[0])
    else:
        fig, ax = 0, [0, 0, 0, 0]

    popt2 = fit_two_gaussians_guess(x, y, guess2)
    err2 = compute_relative_error(x, y, popt2, g)
    if verbose:
        print(f'Err2 = {err2}')
        plot_two_gaussians(x, y, popt2, fig=fig, ax=ax[1])

    popt3 = fit_two_gaussians_guess(x, y, guess3)
    err3 = compute_relative_error(x, y, popt3, g)
    if verbose:
        print(f'Err3 = {err3}')
        plot_two_gaussians(x, y, popt3, fig=fig, ax=ax[2])

    popt4 = fit_two_gaussians_guess(x, y, guess4)
    err4 = compute_relative_error(x, y, popt4, g)
    if verbose:
        print(f'Err4 = {err4}')
        plot_two_gaussians(x, y, popt4, fig=fig, ax=ax[3])

    if (err1 < err2) & (err1 < err3) & (err1 < err4):
        return popt1
    elif (err2 < err3) & (err2 < err4):
        return popt2
    elif err3 < err4:
        return popt3
    else:
        return popt4


# For a sum of three gaussians

def h(x, a, b, c, a1, b1, c1, a2, b2, c2):
    return f(x, a, b, c) + f(x, a1, b1, c1) + f(x, a2, b2, c2)


def fit_three_gaussians(x, y, verbose=False):
    guess = [3 * 10 ** 2, -10, 50, 160, -40, 1000, 17, -180, 2000]

    try:
        popt = fit(h, x, y, guess, maxfev=maxfev)
    except:
        try:
            popt = fit(g, x, y, guess[:6], maxfev=maxfev)
            if verbose:
                print('A sum of two gaussians was fitted')
        except:
            try:
                popt = fit(f, x, y, guess[:3], maxfev=maxfev)
                if verbose:
                    print('Only one gaussian was fitted')
            except:
                popt = [0, 0, 1]
                if verbose:
                    print('No gaussian was fitted')
            popt = list(popt) + [0, 0, 1]
        popt = list(popt) + [0, 0, 1]
    return popt


def order(maximum, center, sigma, max1, center1, sigma1, max2, center2, sigma2):
    maxes = np.array([maximum, max1, max2])
    centers = np.array([center, center1, center2])
    stds = np.array([sigma, sigma1, sigma2])

    i0 = np.argmin(centers)
    i2 = np.argmax(centers)
    i1 = np.nan
    for i in range(3):
        if i0 != i and i2 != i:
            i1 = i
    assert i1 == 0 or i1 == 1 or i1 == 2, "i1 has to be 0, 1 or 2."
    params = [maxes[i0], centers[i0], stds[i0], maxes[i1], centers[i1], stds[i1], maxes[i2], centers[i2], stds[i2]]

    return np.array(params)


# Plotting functions
def plot_two_gaussians(x, y, popt2, **kwargs):
    if 'fig' in kwargs and 'ax' in kwargs:
        fig, ax = kwargs['fig'], kwargs['ax']
    else:
        fig, ax = plt.subplots()
    ax.plot(x, y, '+', label='data')
    ax.plot(x, f(x, *popt2[:3]), ':', label='Gaussian 1')
    ax.plot(x, f(x, *popt2[3:]), ':', label='Gaussian 2')
    ax.plot(x, g(x, *popt2), label='Fit two gaussians')
    ax.set_xlabel('Energies')
    ax.set_xscale('log')
    ax.legend()
    fig.show()


def plot_three_gaussians(x, y, popt3, **kwargs):
    if 'fig' in kwargs and 'ax' in kwargs:
        fig, ax = kwargs['fig'], kwargs['ax']
    else:
        fig, ax = plt.subplots()
    ax.plot(x, y, '+', label='data')
    ax.plot(x, f(x, *popt3[:3]), ':', label='Gaussian 1')
    ax.plot(x, f(x, *popt3[3:6]), ':', label='Gaussian 2')
    ax.plot(x, f(x, *popt3[6:]), ':', label='Gaussian 3')
    ax.plot(x, h(x, *popt3), label='Fit three gaussians')
    ax.set_xlabel('Energies')
    ax.set_xscale('log')
    ax.legend()
    fig.show()


def find_info_gaussians_without_fit(x, y, popt, popt2, popt3, verbose=False):
    threshold_good_enough = 0.2
    cold_ions = False
    nb_gaussians = 0

    err = compute_relative_error(x, y, popt, f)
    err2 = compute_relative_error(x, y, popt2, g)
    err3 = compute_relative_error(x, y, popt3, h)

    if verbose:
        print(f'Err = {err}, err2 = {err2}, err3 = {err3}')
        # print 3 fits
        fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
        plot_one_gaussian(x, y, popt, fig=fig, ax=ax[0])
        plot_two_gaussians(x, y, popt2, fig=fig, ax=ax[1])
        plot_three_gaussians(x, y, popt3, fig=fig, ax=ax[2])
        plt.tight_layout()

    if (err > threshold_good_enough) & (err2 > threshold_good_enough) & (err3 > threshold_good_enough):
        if verbose:
            print("No fit is good")

        return 0, False

    else:

        if err <= threshold_good_enough:
            if verbose:
                print(f"Fit with one 'gaussian' is good enough. The relative error is of {err}.")
            nb_gaussians = 1
            # plot_one_gaussian(x,y,popt)
            maximum, center, std = retrieve_characteristics_from_parameters(*popt, x)
            if verbose:
                print(f'Lowest energies : max = {maximum}, center = {center}, std = {std}.')
            if (center < 150) & (maximum > 5 * 10 ** 5) & (std < 200):  # Choose thresholds
                if verbose:
                    print("There are cold ions.")
                cold_ions = True

        elif err2 <= threshold_good_enough:
            if verbose:
                print(f"Fit with two 'gaussians' is good enough. The relative error is of {err2}.")
            nb_gaussians = 2
            # plot_two_gaussians(x,y,popt2)
            maximum, center, std = retrieve_characteristics_from_parameters(*popt2[:3], x)
            max2, center2, std2 = retrieve_characteristics_from_parameters(*popt2[3:], x)
            if center > center2:
                # a, b, c = maximum, center, std
                maximum, center, std = max2, center2, std2
                # max2, center2, std2 = a, b, c
            if verbose:
                print(f'Lowest energies : max = {maximum}, center = {center}, std = {std}.')
            if (center < 150) & (maximum > 5 * 10 ** 5) & (std < 200):  # Choose thresholds
                if verbose:
                    print("There are cold ions.")
                cold_ions = True

        elif err3 <= threshold_good_enough:
            if verbose:
                print(f"Fit with three 'gaussians' is good enough. The relative error is of {err3}.")
            nb_gaussians = 3
            # plot_three_gaussians(x,y,popt3)
            maximum, center, std = retrieve_characteristics_from_parameters(*popt3[:3], x)
            max2, center2, std2 = retrieve_characteristics_from_parameters(*popt3[3:6], x)
            max3, center3, std3 = retrieve_characteristics_from_parameters(*popt3[6:], x)
            params = order(maximum, center, std, max2, center2, std2, max3, center3, std3)
            if verbose:
                print(f'Lowest energies : max = {params[0]}, center = {params[1]}, std = {params[2]}.')
            if (params[1] < 150) & (params[0] > 5 * 10 ** 5) & (params[2] < 200):  # Choose thresholds
                if verbose:
                    print("There are cold ions.")
                cold_ions = True

        return nb_gaussians, cold_ions


def find_info_gaussians(x, y, verbose=False):
    popt = fit_one_gaussian(x, y)
    popt2 = fit_two_gaussians(x, y)
    popt3 = fit_three_gaussians(x, y)

    nb_gaussians, cold_ions = find_info_gaussians_without_fit(x, y, popt, popt2, popt3, verbose=verbose)

    return [*popt, *popt2, *popt3, nb_gaussians, cold_ions]


def fit_populations(x, y, min_energy, max_energy, nb_gaussians, **kwargs):
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

    if nb_gaussians == 1:
        popt_popu = fit_one_gaussian(x_popu, y_popu)
        err = compute_relative_error(x_popu, y_popu, popt_popu, f)

        maximum, center, std = retrieve_characteristics_from_parameters(*popt_popu, x_popu)
        if verbose:
            ax.plot(x_popu, f(x_popu, *popt_popu), label='Fit')
            ax.legend()
            fig.show()
        '''
        if (center <= x_popu[0]) or (center >= x_popu[-1]):
            return np.array([0,-1,err])
        else:
            return np.array([maximum, center,err])
       '''
        return np.array([maximum, center, std, err])

    else:
        popt_popu = fit_two_gaussians(x_popu, y_popu)
        max1, center1, std1 = retrieve_characteristics_from_parameters(*popt_popu[:3], x_popu)
        max2, center2, std2 = retrieve_characteristics_from_parameters(*popt_popu[3:], x_popu)
        err = compute_relative_error(x_popu, y_popu, popt_popu, g)

        if verbose:
            print(f'Center1 = {center1}, center2 = {center2}, err = {err}')
            ax.plot(x_popu, f(x_popu, *popt_popu[:3]), ':', label='Fit gaussian 1')
            ax.plot(x_popu, f(x_popu, *popt_popu[3:]), ':', label='Fit gaussian 2')
            ax.plot(x_popu, g(x_popu, *popt_popu), ':', label='Total fit')
            ax.legend()
            fig.show()

        '''
        if (center1 <= x_popu[0]) :
            max1, center1 = 0,-1
        if (center2 <= x_popu[0]) :
            max2, center2 = 0,-1
        '''

        if max2 > max1:
            max1, center1, std1, max2, center2, std2 = max2, center2, std2, max1, center1, std1
        return np.array([max1, center1, std1, max2, center2, std2, err])


def fit_sum_populations_by_energy_range(x, y, verbose=False):
    if verbose:
        fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
    else:
        fig, ax = 0, [0, 0, 0]
    max_cold, center_cold, std_cold, err = fit_populations(x, y, 0, 100, 1, verbose=verbose, fig=fig, ax=ax[0])
    if (center_cold <= x[x < 100][0]) or (center_cold >= x[x < 100][-1]):
        max_cold, center_cold, std_cold = 0, -1, -1
    if std_cold > 100:
        max_cold, center_cold, std_cold = 0, -1, -1

    max_main, center_main, std_main, err1 = fit_populations(x, y, 80, np.max(x), 1, verbose=verbose, fig=fig, ax=ax[1])
    if verbose:
        print(f'Err1 = {err1}')
    if err1 < 0.15:  # Fit with one gaussian is good enough ?
        if verbose:
            print('One population because fits well enough.')
        if center_main <= x[x > 80][0]:
            max_main, center_main, std_main = 0, -1, -1
            if verbose:
                print('...but in the cold ions range.')
        max_secondary, center_secondary, std_secondary = 0, -1, -1
    else:
        max_main2, center_main2, std_main2, max_secondary, center_secondary, std_secondary, err2 = fit_populations(x, y,
                                            80, np.max(x), 2, verbose=verbose, fig=fig, ax=ax[2])

        if center_secondary <= x[x > 80][0]:
            max_secondary, center_secondary, std_secondary = 0, -1, -1
            if verbose:
                print('Secondary population is in the cold ions range.')
        if center_main2 <= x[x > 80][0]:
            max_main2, center_main2, std_main2 = max_secondary, center_secondary, std_secondary
            max_secondary, center_secondary, std_secondary = 0, -1, -1
            if verbose:
                print('Main population is in the cold ions range.')

        if verbose:
            print(f'Err2 = {err2}')
        if err2 > err1 * 0.6:  # Fit with 2 gaussians is not that much better than with one
            if verbose:
                print('One population, because fit with two gaussians does not lower the error that much.')
            max_secondary, center_secondary, std_secondary = 0, -1, -1
        elif max(center_main, center_secondary) / min(center_main, center_secondary) < 2:
            # The two populations are very close in mean energy
            if verbose:
                print(f'One population because center_main = {center_main} and center_secondary = {center_secondary}')
            max_secondary, center_secondary, std_secondary = 0, -1, -1
            max_main, center_main, std_main = max_main2, center_main2, std_main2
        else:
            if verbose:
                print('Two populations')
            max_main, center_main, std_main = max_main2, center_main2, std_main2
    return np.array([max_cold, center_cold, std_cold, max_main, center_main, std_main, max_secondary, center_secondary,
                     std_secondary])
