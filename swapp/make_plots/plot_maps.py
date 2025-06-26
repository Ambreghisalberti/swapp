from spok.models import planetary
import matplotlib
from spok.coordinates.coordinates import cartesian_to_spherical, spherical_to_cartesian
from scipy.stats import binned_statistic_2d, binned_statistic
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from swapp.make_plots.plot_functions import make_bins
from spok.models.planetary import mp_shue1997
from matplotlib.colors import LogNorm
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from spok.models.planetary import Magnetosheath
from spok.plot import planet_env
import matplotlib.patches as mpatches
import pandas as pd
import os
from datetime import timedelta
from spok.models.planetary import mp_shue1998_normal, mp_shue1998_tangents
from skimage import measure
from sklearn.utils import resample

diverging_cmaps = ['PiYG', 'seismic', 'coolwarm']


def add_spherical_coordinates(df):
    R, theta, phi = cartesian_to_spherical(df.X.values, df.Y.values, df.Z.values)
    df['R'] = R
    df['theta'] = theta
    df['phi'] = phi
    return df


def gaussian_filter_nan_datas(df, sigma):
    if sigma > 0:
        # Get the coordinates of the non-NaN values
        nan_mask = np.isnan(df)
        x, y = np.indices(df.shape)
        valid_points = ~nan_mask
        points = np.vstack((x[valid_points], y[valid_points])).T
        values = df[valid_points]
        # Interpolate the NaN values based on the surrounding values linearly
        interpolated_arr = df.copy()
        interpolated_arr[nan_mask] = griddata(points, values, (x[nan_mask], y[nan_mask]), method='linear')
        assert np.isnan(interpolated_arr).sum() <= nan_mask.sum(), "The first interpolation step adds NaNs."

        # Red-do a step to also fill the ones that are not surrounded by values (nearest inseatd of linear)
        nan_mask2 = np.isnan(interpolated_arr)
        x, y = np.indices(interpolated_arr.shape)
        valid_points = ~nan_mask2
        points = np.vstack((x[valid_points], y[valid_points])).T
        values = interpolated_arr[valid_points]
        interpolated_arr[nan_mask2] = griddata(points, values, (x[nan_mask2], y[nan_mask2]), method='nearest')
        assert np.isnan(
            interpolated_arr).sum() == 0, "The Nanfilling steps have not workdes : there still are Nans in the array."

        # Apply the Gaussian filter to the interpolated array, and add again the initial nans
        filtered_arr = gaussian_filter(interpolated_arr, sigma=sigma)
        filtered_arr[nan_mask] = np.nan

        assert np.isnan(filtered_arr).sum() == nan_mask.sum(), (
            "The gaussian filter should not add ""or remove any NaN value.")
        return filtered_arr
    else:
        return df


def plot_normalized_pannel(df, all_pos, featurex, featurey, fig, bins, sigma, cmap, ax):
    stat, xbins, ybins, _ = binned_statistic_2d(all_pos[featurex].values, all_pos[featurey].values,
                                                all_pos[featurex].values, statistic='count', bins=bins)
    stat2, xbins, ybins, _ = binned_statistic_2d(df[featurex].values, df[featurey].values, df[featurey].values,
                                                 bins=(xbins, ybins), statistic='count')
    stat2[np.isnan(stat2)] = 0

    ratio = stat2.T / stat.T
    ratio[ratio == np.float64('inf')] = np.nan
    ratio[ratio == -np.float64('inf')] = np.nan
    ratio = gaussian_filter_nan_datas(ratio, sigma)
    im = ax.pcolormesh(xbins, ybins, ratio, cmap=cmap)

    fig.colorbar(im, ax=ax)


def plot_relative_diff_pannel(df, all_pos, featurex, featurey, fig, bins, sigma, cmap, ax, **kwargs):
    stat, xbins, ybins, _ = binned_statistic_2d(all_pos[featurex].values, all_pos[featurey].values,
                                                all_pos[featurex].values, statistic='count', bins=bins)
    stat2, xbins, ybins, _ = binned_statistic_2d(df[featurex].values, df[featurey].values, df[featurey].values,
                                                 bins=(xbins, ybins), statistic='count')
    stat[np.isnan(stat)] = 0
    stat2[np.isnan(stat2)] = 0
    stat = stat.T/stat.sum()
    stat2 = stat2.T/stat2.sum()

    relative_diff = 2*(stat2.T - stat.T)/ (stat.T + stat2.T)
    relative_diff[relative_diff == np.float64('inf')] = np.nan
    relative_diff[relative_diff == -np.float64('inf')] = np.nan
    relative_diff = gaussian_filter_nan_datas(relative_diff, sigma)
    if cmap in diverging_cmaps:
        vmin = -np.nanmax(abs(relative_diff))
        vmax = np.nanmax(abs(relative_diff))
    else:
        vmin = np.nanmin(relative_diff)
        vmax = np.nanmax(relative_diff)
    vmin = kwargs.get('vmin', vmin)
    vmax = kwargs.get('vmax', vmax)

    im = ax.pcolormesh(xbins, ybins, relative_diff, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax)


def get_plot_panel_info(method, inputs):
    # Inputs for f except ax
    df, all_pos, featurex, featurey, fig, bins, sigma, cmap = inputs
    if method == 'normal':
        f = plot_panel
        inputs = (df, featurex, featurey, fig, bins, sigma, cmap)
    elif method == 'normalized':
        f = plot_normalized_pannel
    elif method == 'relative diff':
        f = plot_relative_diff_pannel
    else:
        raise Exception("The method given as input is not known.")
    return f, inputs


def plot_pos(df, **kwargs):
    method = kwargs.get('method', 'normal')
    if method == 'normal':
        # In this case, df2 will not be used but need to be declared to avoid an error
        df2 = df.copy()
    if method != 'normal':  # In this case, the argument must be a tuple of 2 dataframes to plot
        df, df2 = df

    ncols = 0
    for k in ['x_slice', 'y_slice', 'z_slice']:
        if k in kwargs:
            ncols += 1
    fig, ax = plt.subplots(ncols=ncols, figsize=(3 * ncols, 3))
    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])

    df = df.dropna()
    df2 = df2.dropna()

    bins = kwargs.pop('bins', 100)
    cmap = kwargs.pop('cmap', 'jet')

    i = 0
    if 'z_slice' in kwargs:
        inputs = (df, df2, 'X', 'Y', fig, bins, kwargs.get('sigma', 0), cmap)
        f, inputs = get_plot_panel_info(method, inputs)
        f(*inputs, ax[i])
        i += 1
    if 'y_slice' in kwargs:
        inputs = (df, df2, 'X', 'Z', fig, bins, kwargs.get('sigma', 0), cmap)
        f, inputs = get_plot_panel_info(method, inputs)
        f(*inputs, ax[i])
        i += 1
    if 'x_slice' in kwargs:
        inputs = (df, df2, 'Y', 'Z', fig, bins, kwargs.get('sigma', 0), cmap)
        f, inputs = get_plot_panel_info(method, inputs)
        f(*inputs, ax[i])
        i += 1

    msh = planetary.Magnetosheath(magnetopause='mp_shue1998', bow_shock='bs_jelinek2012')
    x_lim = kwargs.pop('x_lim', (-2, 25))
    fig, ax = planet_env.layout_earth_env(msh, figure=fig, axes=np.array([ax]), x_lim=x_lim, **kwargs)
    if 'title' in kwargs:
        fig.suptitle(kwargs['title'])
    for a in ax.ravel():
        a.set_aspect('equal')


def plot_panel(to_plot, featurex, featurey, fig, bins, sigma, cmap, ax, **kwargs):
    hist, xbins, ybins, _ = ax.hist2d(to_plot[featurex].values, to_plot[featurey].values, cmin=1, bins=bins,
                                      cmap=cmap, range=[[-40, 40], [-40, 40]])
    hist = gaussian_filter_nan_datas(hist, sigma)
    im = ax.pcolormesh(xbins, ybins, hist.T, cmap=cmap)
    fig.colorbar(im, ax=ax)


def plot_pos_hist(pos, fig, ax, **kwargs):  # Transform with the slices kwargs to make it more general
    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])
    assert isinstance(ax[0], matplotlib.axes._axes.Axes), "The ax given as input must an array of Axes."
    to_plot = pos.dropna()

    bins = kwargs.pop('bins', 100)
    cmap = kwargs.pop('cmap', 'jet')
    sigma = kwargs.pop('sigma', 0)

    i = 0
    if 'z_slice' in kwargs:
        plot_panel(to_plot, 'X', 'Y', fig, bins, sigma, cmap, ax[i], **kwargs)
        i += 1
    if 'y_slice' in kwargs:
        plot_panel(to_plot, 'X', 'Z', fig, bins, sigma, cmap, ax[i], **kwargs)
        i += 1
    if 'x_slice' in kwargs:
        plot_panel(to_plot, 'Y', 'Z', fig, bins, sigma, cmap, ax[i], **kwargs)
        i += 1
    plt.tight_layout()


"""
def plot_pos(df, **kwargs):
    ncols = 0
    for k in ['x_slice', 'y_slice', 'z_slice']:
        if k in kwargs:
            ncols += 1
    fig, ax = plt.subplots(ncols=ncols, figsize=(3 * ncols, 3))
    if 'title' in kwargs:
        fig.suptitle(kwargs.pop('title'))
    plot_pos_hist(df[['X', 'Y', 'Z']].dropna(), fig, ax, **kwargs)

    msh = planetary.Magnetosheath(magnetopause='mp_shue1998', bow_shock='bs_jelinek2012')
    fig, ax = planet_env.layout_earth_env(msh, figure=fig, axes=np.array([ax]), x_lim=(-2, 25), **kwargs)
"""


def median_curve_transition_param(temp, feature):
    stat, xbins, im = binned_statistic(temp.normalized_logNoverT.values, temp[feature].values, statistic='median',
                                       bins=1000)
    stat_count, xbins, im = binned_statistic(temp.normalized_logNoverT.values, temp[feature].values, statistic='count',
                                             bins=1000)

    fig, ax = plt.subplots(ncols=2, figsize=(8, 4))

    ax[0].plot(xbins[:-1], gaussian_filter(stat, 3))
    ax[0].set_xlabel('Transition parameter')
    ax[0].set_ylabel(r'$\frac{|\overrightarrow{V_{MSH}} - \overrightarrow{V}|}{V_{MSH}}$', rotation='horizontal')
    ax[0].yaxis.set_label_coords(-0.23, 0.5)
    plt.xticks([0, 1], ['MSP', 'MSH'])

    ax[1].plot(xbins[:-1], gaussian_filter(stat_count, 3))
    ax[1].set_xlabel('Transition parameter')
    ax[1].set_ylabel('Count')
    plt.xticks([0, 1], ['MSP', 'MSH'])

    fig.tight_layout()
    return fig, ax


def hist_transition_param(temp, feature, transition='normalized_logNoverT', scale='linear', **kwargs):
    bins = make_bins(scale, temp[[feature]].dropna(), feature, nb_bins=kwargs.get('nb_bins', 200))
    stat, xbins, ybins, im = binned_statistic_2d(temp.normalized_logNoverT.values, temp[feature].values,
                                                 temp.normalized_logNoverT.values, statistic='count',
                                                 bins=(np.linspace(0, 1, 100), bins))
    stat[stat == 0] = np.nan

    if kwargs.get('normalized', False):
        stat = stat / np.array([np.nansum(stat, axis=1) for _ in range(len(ybins) - 1)]).T
    stat = gaussian_filter_nan_datas(stat, 1)

    if 'fig' not in kwargs or 'ax' not in kwargs:
        fig, ax = plt.subplots()
    else:
        fig, ax = kwargs['fig'], kwargs['ax']

    im = ax.pcolormesh(xbins, ybins, stat.T, cmap='jet')
    plt.colorbar(im)
    ax.set_xlabel(f'Transition parameter {transition}')
    ax.set_xlabel(f'Transition parameter {transition}')
    ax.set_ylabel(feature)
    ax.yaxis.set_label_coords(-0.23, 0.5)
    plt.xticks([0, 1], ['MSP', 'MSH'])
    ax.set_title(f'Distribution of {feature}\nacross the Boundary layer depth')
    ax.set_yscale(scale)

    infos = []
    if kwargs.get('plot_max', False):
        interp = gaussian_filter_nan_datas(stat, 5)
        maxes = []
        for i in range(len(stat)):
            maxes += [ybins[np.nanargmax(interp[i])]]
        maxes = np.array(maxes) + (ybins[1] - ybins[0]) / 2
        ax.plot(xbins[:-1] + (xbins[1] - xbins[0]) / 2, maxes, label='max value')
        infos += [maxes]
    if kwargs.get('plot_mean', False):
        means = []
        for i in range(len(stat)):
            means += [np.nanmean(temp[np.logical_and(temp[transition].values >= xbins[i],
                                                     temp[transition].values < xbins[i + 1])][
                                     feature].values).item()]
        ax.plot(xbins[:-1] + (xbins[1] - xbins[0]) / 2, means, label='mean')
        infos += [means]
    if kwargs.get('plot_median', False):
        medians = []
        for i in range(len(stat)):
            medians += [np.nanmedian(temp[np.logical_and(temp[transition].values >= xbins[i],
                                                         temp[transition].values < xbins[i + 1])][
                                         feature].values).item()]
        ax.plot(xbins[:-1] + (xbins[1] - xbins[0]) / 2, medians, label='median')
        infos += [medians]
    if kwargs.get('plot_median', False) or kwargs.get('plot_mean', False) or kwargs.get('plot_max', False):
        ax.legend()

    fig.tight_layout()
    return fig, ax, xbins, ybins, stat, infos


def reposition(df, **kwargs):
    if 'Rmin' in kwargs:
        rmin = kwargs['Rmin']
    else:
        rmin = df.r_MP_Shue.values / 2
    rmax = df.r_MP_Shue.values
    df['normalized_R'] = rmin + (rmax - rmin) * df.normalized_logNoverT.values
    norm_x, norm_y, norm_z = spherical_to_cartesian(df.normalized_R.values, df.theta.values, df.phi.values)
    df['normalized_X'] = norm_x
    df['normalized_Y'] = norm_y
    df['normalized_Z'] = norm_z
    return df


def plot_repositionned_pannel(temp, featurex, featurey, f, fig, ax, cmap, bins, sigma, **kwargs):
    stat, xbins, ybins, _ = binned_statistic_2d(temp[featurex].values, temp[featurey].values,
                                                f(temp), statistic='median', bins=bins)
    stat = gaussian_filter_nan_datas(stat, sigma)
    if cmap in diverging_cmaps:
        vmax = np.nanmax(abs(stat))
        vmin = -vmax
    else:
        vmax, vmin = None, None
    vmin = kwargs.get('vmin', vmin)
    vmax = kwargs.get('vmax', vmax)
    im = ax.pcolormesh(xbins, ybins, stat.T, vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(im, ax=ax)


def plot_repositionned_stat_binned(df, feature, **kwargs):
    zscale = kwargs.get('zscale', 'linear')
    cmap = kwargs.get('cmap', 'jet')
    sigma = kwargs.get('sigma', 0)
    bins = kwargs.get('bins', 100)

    if kwargs.get('cmap') in diverging_cmaps and kwargs.get('zscale') == 'log':
        raise Exception(
            f'The {kwargs.get("cmap")} cmap was designed in this function to be centered around 0, '
            'so it will not work for a log scale.')

    if 'r_MP_Shue' not in df.columns:
        r, theta, phi = mp_shue1997(df.theta.values, df.phi.values, coord_sys='spherical')
        df['r_MP_Shue'] = r
    if 'normalized_R' not in df.columns:
        df = reposition(df, **kwargs)
    if zscale == 'log':
        kwargs['norm'] = LogNorm()
    f = kwargs.pop('function', lambda x: x[feature].values.flatten())

    ncols = 0
    for k in ['x_slice', 'y_slice', 'z_slice']:
        if k in kwargs:
            ncols += 1
    if ('fig' not in kwargs) or ('ax' not in kwargs):
        fig, ax = plt.subplots(ncols=ncols, figsize=(4 * ncols, 4))
    else:
        fig, ax = kwargs.pop('fig'), kwargs.pop('ax')

    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])

    # Drawing cuts instead of projection to see better : points at max 1Re of the plane
    kwargsplot = {}
    if 'vmin' in kwargs:
        kwargsplot['vmin'] = kwargs.get('vmin')
    if 'vmax' in kwargs:
        kwargsplot['vmax'] = kwargs.get('vmax')

    i = 0
    if 'z_slice' in kwargs:
        temp = df[abs(df.normalized_Z.values) < 1]
        plot_repositionned_pannel(temp, 'normalized_X', 'normalized_Y', f, fig, ax[i], cmap, bins,
                                  sigma, **kwargsplot)
        i += 1

    if 'y_slice' in kwargs:
        temp = df[abs(df.normalized_Y.values) < 1]
        plot_repositionned_pannel(temp, 'normalized_X', 'normalized_Z', f, fig, ax[i], cmap, bins,
                                  sigma, **kwargsplot)
        i += 1

    if 'x_slice' in kwargs:
        plot_repositionned_pannel(df, 'normalized_Y', 'normalized_Z', f, fig, ax[i], cmap, bins,
                                  sigma, **kwargsplot)

    msh = planetary.Magnetosheath(magnetopause='mp_shue1998', bow_shock='bs_jelinek2012')
    fig, ax = planet_env.layout_earth_env(msh, figure=fig, axes=np.array([ax]), x_lim=(-2, 25), **kwargs)
    if 'title' in kwargs:
        fig.suptitle(kwargs['title'])
    else:
        fig.suptitle(f'{feature} for data repositioned with transition parameter\nCuts around plane +/- 1Re are shown')
    fig.tight_layout()


def make_mp_grid(**kwargs):
    N_grid = kwargs.get('N_grid', 300)
    coord = kwargs.get('coord', 'spherical')
    if coord == 'spherical':
        th = np.linspace(0, 0.5 * np.pi, N_grid)
        ph = np.linspace(-np.pi, np.pi, 2 * N_grid)
        theta, phi = np.meshgrid(th, ph)
        msh = Magnetosheath()
        Xmp, Ymp, Zmp = msh.magnetopause(theta, phi)
    elif coord == 'cartesian':
        N_grid = 600
        th = np.linspace(0, 0.5 * np.pi, 3000)
        ph = np.linspace(-np.pi, np.pi, 3000)
        theta, phi = np.meshgrid(th, ph)
        msh = Magnetosheath()
        X, Y, Z = msh.magnetopause(theta, phi)
        X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()
        Ymp = np.linspace(-15, 15, N_grid)
        Zmp = np.linspace(-15, 15, N_grid)
        Ymp, Zmp = np.meshgrid(Ymp, Zmp)
        Ymp, Zmp = Ymp.flatten(), Zmp.flatten()

        model = KNeighborsRegressor(n_neighbors=1, weights='distance', n_jobs=1)
        model.fit(np.array([Y, Z]).T, X)
        Xmp = model.predict(np.array([Ymp, Zmp]).T)
        Xmp = Xmp.reshape((N_grid, N_grid))
        Ymp = Ymp.reshape((N_grid, N_grid))
        Zmp = Zmp.reshape((N_grid, N_grid))
    else:
        raise Exception(f'coord should be either spherical or cartesian but is {coord}.')

    return Xmp, Ymp, Zmp


def make_data_to_grid(df, **kwargs):
    features = kwargs.get('features', list(df.columns.values))
    data = df[np.unique(np.array(features + ['X', 'Y', 'Z']))].dropna()
    values = data[features]
    pos = data[['X', 'Y', 'Z']].values
    return pos, values


def train_knn(x, y, N_neighbours=10000, r=1, method='KNN', **kwargs):
    if method == 'KNN':
        model = KNeighborsRegressor(n_neighbors=N_neighbours, weights='distance', n_jobs=1)
    elif method == 'RNN':
        model = RadiusNeighborsRegressor(radius=r, weights='distance', n_jobs=1)
    else:
        raise Exception(f'The method should be either KNN or RNN, but is {method}')
    model.fit(x, y)
    return model


def compute_knn(knn_function, pos, values, p0, cpu=40, **kwargs):
    interp = train_knn(pos, values, **kwargs)
    with Pool(cpu) as pool:
        p0 = [(interp, p) for p in p0]
        qty = pool.map(knn_function, p0)
    pool.close()
    return np.asarray(qty)


def f_interp(inputs):
    interp, p0 = inputs
    b = interp.predict(p0)
    return b


def make_maps(df, **kwargs):
    Xmp, Ymp, Zmp = make_mp_grid(**kwargs)
    if kwargs.get('balance',False):
        df = equal_sample_data(df, kwargs.get('feature_to_balance','omni_CLA'), **kwargs)
    pos, values = make_data_to_grid(df, **kwargs)

    if kwargs.get('N_neighbours_method') == 'automatic':
        kwargs['N_neighbours'] = int(len(df) * kwargs.get('N_neighbours_proportion', 0.1))
    interpolated_values = compute_knn(f_interp, pos, values, np.asarray([Xmp, Ymp, Zmp]).T, **kwargs).T
    interpolated_values = {feat: values for feat, values in
                           zip(kwargs.get('features', list(df.columns.values)), interpolated_values)}
    return interpolated_values


def is_map_valid(df, **kwargs):
    Xmp, Ymp, Zmp = make_mp_grid(**kwargs)

    _ = kwargs.pop('features', None)
    pos, values = make_data_to_grid(df, features=['X'], **kwargs)
    interp = train_knn(pos, values, **kwargs)

    max_distance = kwargs.get('max_distance', 2)
    valid = True * np.ones(Xmp.shape)
    for i in range(Xmp.shape[0]):
        for j in range(Xmp.shape[1]):
            distances, indices = interp.kneighbors([[Xmp[i, j], Ymp[i, j], Zmp[i, j]]])
            if np.min(distances) > max_distance:
                valid[i, j] = False
    return valid


def compute_and_plot_map(df, **kwargs):
    results = make_maps(df, **kwargs)
    valid = is_map_valid(df, **kwargs)
    _, _ = plot_maps(df, results, valid=valid, **kwargs)
    return results, valid


def arc_patch(center, radius, theta1, theta2, ax=None, resolution=50, **kwargs):  # Adapted from stack overflow
    # make sure ax is not empty
    if ax is None:
        ax = plt.gca()
    # generate the points
    theta = np.linspace(theta1, theta2, resolution)
    points = np.vstack((list(radius*np.cos(theta) + center[0])+[center[0]],
                        list(radius*np.sin(theta) + center[1])+[center[1]]))
    # build the polygon and add it to the axes
    poly = mpatches.Polygon(points.T, closed=True, **kwargs)
    ax.add_patch(poly)
    return poly


def plot_CLA_sector(cx, cy, r, theta1, theta2, ax):
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(cx + r * np.cos(th), cy + r * np.sin(th), color='k', alpha=0.5)
    arc_patch((cx, cy), r, (np.pi/2-theta2), (np.pi/2-theta1), ax=ax, fill=True, color='red', alpha=0.3)


def hist_by_CLA(BL, feature, **kwargs):
    if 'sectors' in kwargs:
        sectors_CLA = kwargs['sectors']
        nb_sectors = len(sectors_CLA) - 1
    else:
        nb_sectors = kwargs.get('nb_sectors', 9)
        sectors_CLA = np.linspace(-np.pi, np.pi, nb_sectors + 1)

    ncols = kwargs.get('ncols', 3)
    cmap = kwargs.get('cmap', 'jet')

    msh = planetary.Magnetosheath(magnetopause='mp_shue1998', bow_shock='bs_jelinek2012')

    nrows = int(np.ceil(nb_sectors / ncols))
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(3 * ncols, 3 * nrows))
    if len(ax.shape) == 1:
        ax = np.array([ax])
    for i in range(nb_sectors):
        temp = make_CLA_slice(BL, sectors_CLA[i], sectors_CLA[i+1])

        stat, xbins, ybins, im = binned_statistic_2d(temp.Y.values, temp.Z.values, temp[feature].values,
                                                     statistic='mean', bins=kwargs.get('bins', 50))
        stat = gaussian_filter_nan_datas(stat, kwargs.get('sigma',0))

        if cmap in diverging_cmaps:
            vmin, vmax = -np.nanmax(abs(stat)), np.nanmax(abs(stat))
        else:
            vmin, vmax = np.nanmin(stat), np.nanmax(stat)
        vmin = kwargs.get('vmin', vmin)
        vmax = kwargs.get('vmax', vmax)

        im = ax[i // ncols, i % ncols].pcolormesh(xbins, ybins, stat.T, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax[i // ncols, i % ncols])
        ax[i // ncols, i % ncols].set_title(
            f'{feature}\nfor {round(sectors_CLA[i]*180/np.pi, 2)}° < CLA < '
            f'{round(sectors_CLA[i + 1]*180/np.pi, 2)}°\n{len(temp)} points')
        plot_CLA_sector(14, 12, 2.5, sectors_CLA[i], sectors_CLA[i + 1], ax[i // ncols, i % ncols])
        _, _ = planet_env.layout_earth_env(msh, figure=fig, axes=np.array([ax[i // ncols, i % ncols]]), y_lim=(-17, 17),
                                           z_lim=(-17, 17), x_slice=0)

    fig.tight_layout()
    return fig, ax


def make_CLA_slice(df, min_cla, max_cla):
    if (min_cla >= -np.pi) and (max_cla <= np.pi):
        temp = df[df.omni_CLA.values >= min_cla]
        temp = temp[temp.omni_CLA.values < max_cla]
    elif min_cla < -np.pi:
        temp1 = df[df.omni_CLA.values >= min_cla + 2 * np.pi]
        temp2 = df[df.omni_CLA.values < max_cla]
        temp = pd.concat([temp1, temp2])
    else:  # sectors_CLA[i+1] > np.pi
        temp1 = df[df.omni_CLA.values >= min_cla]
        temp2 = df[df.omni_CLA.values < max_cla - 2 * np.pi]
        temp = pd.concat([temp1, temp2])
    return temp


def make_slice(df, feature, min_val, max_val):
    if 'CLA' in feature or 'COA' in feature:
        return make_CLA_slice(df, min_val, max_val)
    else:
        return df[np.logical_and(min_val < df[feature].values, df[feature].values < max_val)]


def make_description_from_kwargs(N_neighbours, **kwargs):
    plot_kwargs = ['ncols', 'nrows', 'min_cla', 'max_cla', 'cmap', 'nb_sectors', 'sectors', 'min_sectors',
                   'max_sectors', 'sigma', 'fig', 'ax', 'valid', 'show_ylabel', 'show_colorbar', 'plot_arrows',
                   'vmax', 'vmin', 'step', 'head_width', 'factor', 'slice', 'arrows_coordinates', 'x_lim',
                   'y_lim', 'z_lim']
    # First order by alphabetical order, to avoid recomputing just because we gave kwargs in a different order
    # Second, don't use plot kwargs because they have no effect on data to plot
    keys = list(kwargs.keys())
    keys.sort()
    description = ''
    chosen_description = ''
    for k in keys:
        if k != "N_neighbours" and k != "overwrite" and k != 'chosen_description' and k not in plot_kwargs:
            description += f'{k}={kwargs[k]}_'
        if k == 'chosen_description':
            chosen_description = '_'+kwargs[k]
    description += f'Nneighbours={N_neighbours}{chosen_description}'
    return description


def make_sectors(df, feature, **kwargs):
    if 'min_sectors' in kwargs and 'max_sectors' in kwargs:
        min_sectors = kwargs['min_sectors']
        max_sectors = kwargs['max_sectors']
        nb_sectors = len(min_sectors)
    elif 'sectors' in kwargs:
        sectors = kwargs['sectors']
        nb_sectors = len(sectors) - 1
        min_sectors, max_sectors = sectors[:-1], sectors[1:]
    else:
        nb_sectors = kwargs.get('nb_sectors', 9)
        sectors = np.linspace(np.nanmin(df[feature].values), np.nanmax(df[feature].values), nb_sectors + 1)
        min_sectors, max_sectors = sectors[:-1], sectors[1:]
    return nb_sectors, min_sectors, max_sectors


def add_speed_arrows(ax, length_arrow=1, **kwargs):
    max_distance = kwargs.get('max_distance', 3)
    N_neighbours = kwargs.get('N_neighbours', 500)
    nb_sectors = kwargs.get('nb_sectors', 9)
    coord = kwargs.get('coord', 'spherical')

    Xmp, Ymp, Zmp = make_mp_grid(**kwargs)

    sectors_CLA = np.linspace(-np.pi, np.pi, nb_sectors + 1)

    nrows, ncols = ax.shape
    for i in range(nb_sectors):
        results_Vy = pd.read_pickle(f'/home/ghisalberti/Maps/gap_to_MSH_Vy_CLA_{sectors_CLA[i]}_{sectors_CLA[i + 1]}_'
                                    f'Nneighbours={N_neighbours}_coord={coord}.pkl')['gap_to_MSH_Vy']
        results_Vz = pd.read_pickle(f'/home/ghisalberti/Maps/gap_to_MSH_Vz_CLA_{sectors_CLA[i]}_{sectors_CLA[i + 1]}_'
                                    f'Nneighbours={N_neighbours}_coord={coord}.pkl')['gap_to_MSH_Vz']
        valid = pd.read_pickle(
            f'/home/ghisalberti/Maps/data/validity_CLA_{sectors_CLA[i]}_{sectors_CLA[i + 1]}_'
            f'Nneighbours={N_neighbours}_maxdistance={max_distance}_coord={coord}.pkl')
        results_Vy[valid == 0] = np.nan
        results_Vz[valid == 0] = np.nan

        nb_hop = kwargs.get('nb_hop', 20)

        sub_Vy = results_Vy[::nb_hop, ::nb_hop]
        sub_Vz = results_Vz[::nb_hop, ::nb_hop]
        norms = np.sqrt(sub_Vy ** 2 + sub_Vz ** 2)/length_arrow
        # sub_Vy = (sub_Vy/norms).flatten()
        # sub_Vz = (sub_Vz/norms).flatten()
        sub_Vy = sub_Vy.flatten() / np.nanmax(norms)
        sub_Vz = sub_Vz.flatten() / np.nanmax(norms)

        for y, z, vy, vz in zip(Ymp[::nb_hop, ::nb_hop].flatten(), Zmp[::nb_hop, ::nb_hop].flatten(), sub_Vy, sub_Vz):
            ax[i // ncols, i % ncols].arrow(y, z, vy, vz, color='red', head_width=0.2)


def associate_SW_Safrankova(X_sat, omni, BS_standoff, dtm=0, sampling_time='5S', vx_median=-406.2):
    if dtm != 0:
        vxmean = abs(omni.Vx.rolling(int((2*dtm+1)*timedelta(minutes=1)/(omni.index[-1]-omni.index[-2])),
                                     center=True, min_periods=1).mean())
    else:
        vxmean = abs(omni.Vx)
    BS_x0 = BS_standoff[BS_standoff.index.isin(X_sat.index)]
    BS_x0 = BS_x0.fillna(13.45)
    lag = np.array(np.round((BS_x0.values-X_sat.values)*6371/vx_median),dtype='timedelta64[s]')
    time = (X_sat.index-lag).round(sampling_time)
    vx = pd.Series(name='Vx',dtype=float)
    vx = pd.concat([vx, vxmean.loc[time]]).fillna(abs(vx_median)).values
    lag = np.array(np.round((BS_x0.values.flatten()-X_sat.values.flatten())*6371/vx), dtype='timedelta64[s]')
    time = (X_sat.index-lag).round(sampling_time)
    OMNI = pd.DataFrame(columns=omni.columns)
    OMNI = pd.concat([OMNI, omni.loc[time]])
    OMNI.index = X_sat.index
    return OMNI


def get_kwargsplot(to_plot, **kwargs):
    kwargsplot = {'cmap': kwargs.get('cmap')}
    if kwargs.get('cmap') in diverging_cmaps:
        kwargsplot['vmax'] = kwargs.get('vmax', abs(np.nanmax(to_plot)))
        kwargsplot['vmin'] = kwargs.get('vmin', -abs(np.nanmax(to_plot)))
    else:
        kwargsplot['vmax'] = kwargs.get('vmax', np.nanmax(to_plot))
        kwargsplot['vmin'] = kwargs.get('vmin', np.nanmin(to_plot))
    kwargsplot['vmin'] = kwargs.get('vmin', kwargsplot['vmin'])
    kwargsplot['vmax'] = kwargs.get('vmax', kwargsplot['vmax'])
    return kwargsplot


def plot_colorbar_next_to_axis(im, fig, ax):
    xmin = ax.get_position().x0
    xmax = ax.get_position().x1
    ymin = ax.get_position().y0
    ymax = ax.get_position().y1
    xwidth = xmax - xmin
    ywidth = ymax - ymin
    cax = fig.add_axes([xmax * 1.1, ymin + 0.1 * ywidth, xwidth * 0.05, ywidth * 0.75])
    cb = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize='small')


def plot_colorbar_separate_axis(im, fig, ax):
    ax.axis("off")
    cb = plt.colorbar(im, ax=ax, location='left')
    cb.ax.tick_params(labelsize='small')


def plot_colorbar(im, fig, ax, method='separate_axis'):
    if method == 'separate_axis':
        plot_colorbar_separate_axis(im, fig, ax)
    elif method == 'next_to_axis':
        plot_colorbar_next_to_axis(im, fig, ax)
    else:
        raise Exception(
            f"Only separate_axis and next_to_axis methods have been implemented for plotting the colorbar, but {method}"
            f" was asked for.")


def make_ax_2D(ax):
    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])
    if len(ax.shape) == 1:
        ax = np.array([ax])
    return ax


def get_fig_ax(features, **kwargs):
    if 'fig' not in kwargs or 'ax' not in kwargs:
        ncols = kwargs.get('ncols', 5)
        if len(features) < ncols:
            ncols = len(features)
        nrows = int(np.ceil(len(features) / ncols))
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(3 * ncols, 3 * nrows))
    else:
        fig, ax = kwargs['fig'], kwargs['ax']
    ax = make_ax_2D(ax)
    nrows, ncols = ax.shape
    return fig, ax, nrows, ncols


def plot_maps(df, results, **kwargs):
    features = kwargs.get('features', list(results.keys()))
    if 'Ymp' not in kwargs or 'Zmp' not in kwargs:
        _, Ymp, Zmp = make_mp_grid(**kwargs)
    else:
        Ymp, Zmp = kwargs['Ymp'], kwargs['Zmp']
    valid = kwargs.get('valid', np.ones(Ymp.shape))

    fig, ax, nrows, ncols = get_fig_ax(features, **kwargs)

    msh = Magnetosheath(magnetopause='mp_shue1998', bow_shock='bs_jelinek2012')
    for i, feature in enumerate(features):
        a = ax[i // ncols, i % ncols]

        if kwargs.get('method_map','KNN') == 'KNN':
            to_plot = results[feature].copy()
            to_plot[valid == 0] = np.nan
        elif kwargs.get('method_map','KNN') == 'binned_stat':
            to_plot = results[feature].copy()
        to_plot = gaussian_filter_nan_datas(to_plot, kwargs.get('sigma', 0))
        kwargsplot = get_kwargsplot(to_plot, **kwargs)
        im = a.pcolormesh(Ymp, Zmp, to_plot, **kwargsplot)
        y, z = find_stagnation_line(to_plot, **kwargs)
        a.scatter(y,z,color='green')

        a.tick_params(labelsize='small')
        a.set_title(feature)
        if kwargs.get('show_colorbar', True):
            if kwargs.get('method_show_colorbar', 'separate_axis') == 'separate_axis':
                plot_colorbar(im, fig, ax[i // ncols, i % ncols + 1])
            else:
                plot_colorbar(im, fig, a, method=kwargs.get('method_show_colorbar'))
        _, _ = planet_env.layout_earth_env(msh, figure=fig, axes=np.array([a]),
                                           y_lim=(-17, 17), z_lim=(-15, 15), x_slice=0)
        if not (kwargs.get('show_ylabel', False)):
            a.set_ylabel('')
        a.axis('equal')

    if 'arrows_coordinates' in kwargs:
        Ymp, Zmp, Vy, Vz = kwargs['arrows_coordinates'].values()
        plot_arrows(a, Ymp, Zmp, Vy, Vz)

    fig.tight_layout()
    return fig, ax


def update_vmin_vmax(results, vmin, vmax, nb_iter, kwargs):
    results = gaussian_filter_nan_datas(results, kwargs.get('sigma', 0))
    if nb_iter == 0 and (('vmin' not in kwargs) or ('vmax' not in 'kwargs')):
        vmin_temp, vmax_temp = np.nanmin(results), np.nanmax(results)
        if vmin_temp < vmin:
            vmin = vmin_temp
        if vmax_temp > vmax:
            vmax = vmax_temp
    return vmin, vmax


def manage_vmin_vmax(vmin, vmax, nb_iter, kwargs):
    if nb_iter == 1:
        if 'vmin' not in kwargs:
            kwargs['vmin'] = vmin
        if 'vmax' not in kwargs:
            kwargs['vmax'] = vmax
        if kwargs.get('cmap', 'jet') == 'seismic' or kwargs.get('cmap', 'jet') == 'coolwarm':
            vmin, vmax = kwargs['vmin'], kwargs['vmax']
            vmin, vmax = -max(abs(vmin), abs(vmax)), max(abs(vmin), abs(vmax))
            kwargs['vmin'], kwargs['vmax'] = vmin, vmax
    return kwargs


def get_valid(feature_to_slice, min_val, max_val, description, temp, N_neighbours, max_distance, kwargs):
    path = (f'/home/ghisalberti/Maps/data/validity_{feature_to_slice}_{min_val}_{max_val}_'
            + description + '.pkl')
    if not (kwargs.get('overwrite', False)) and os.path.isfile(path):
        valid = pd.read_pickle(path)
    else:
        valid = is_map_valid(temp, N_neighbours=N_neighbours, max_distance=max_distance, **kwargs)
        pd.to_pickle(valid, path)
    return valid


def get_map_from_path(path, feature_to_map, temp, N_neighbours, **kwargs):
    if not (kwargs.get('overwrite', False)) and os.path.isfile(path):
        results = pd.read_pickle(path)
    else:
        if len(temp) > N_neighbours:
            results = make_maps(temp, features=[feature_to_map], N_neighbours=N_neighbours, **kwargs)
        else:
            results = make_maps(temp, features=[feature_to_map], N_neighbours=int(N_neighbours // 4), **kwargs)
        pd.to_pickle(results, path)
    return results


def get_path_slice(feature_to_map, feature_to_slice, min_val, max_val, N_neighbours, **kwargs):
    description = make_description_from_kwargs(N_neighbours, **kwargs)
    path = (f'/home/ghisalberti/Maps/data/{feature_to_map}_{feature_to_slice}_{min_val}_{max_val}_'
            + description + '.pkl')
    return path


def get_path(feature_to_map, N_neighbours, **kwargs):
    description = make_description_from_kwargs(N_neighbours, **kwargs)
    path = (f'/home/ghisalberti/Maps/data/{feature_to_map}_'+ description + '.pkl')
    return path


def get_map_slice(feature_to_map, feature_to_slice, min_val, max_val, temp, N_neighbours, **kwargs):
    description = make_description_from_kwargs(N_neighbours, **kwargs)
    path = get_path_slice(feature_to_map, feature_to_slice, min_val, max_val, N_neighbours, **kwargs)
    results = get_map_from_path(path, feature_to_map, temp, N_neighbours, **kwargs)

    return results, description


def get_map_whole_dataset(feature_to_map, temp, N_neighbours, **kwargs):
    description = make_description_from_kwargs(N_neighbours, **kwargs)
    path = (f'/home/ghisalberti/Maps/data/{feature_to_map}_'
            + description + '.pkl')
    results = get_map_from_path(path, feature_to_map, temp, N_neighbours, **kwargs)
    return results, description


def get_map(*inputs, **kwargs):
    if kwargs.get('slice', False):
        results, description = get_map_slice(*inputs, **kwargs)
    else:
        results, description = get_map_whole_dataset(*inputs, **kwargs)
    return results, description


def compute_one_sector(df, feature_to_map, feature_to_slice, min_sectors, max_sectors, vmin, vmax, nb_iter,
                       N_neighbours,
                       max_distance, fig, ax, i, ncols, show_ylabel, show_colorbar, kwargs):
    temp = make_slice(df, feature_to_slice, min_sectors[i], max_sectors[i])
    if kwargs.get('method_map','KNN')=='KNN':
        results, description = get_map_slice(feature_to_map, feature_to_slice, min_sectors[i], max_sectors[i], temp,
                                         N_neighbours, **kwargs)
        vmin, vmax = update_vmin_vmax(results[feature_to_map], vmin, vmax, nb_iter, kwargs)
        valid = get_valid(feature_to_slice, min_sectors[i], max_sectors[i], description, temp, N_neighbours, max_distance,
                          kwargs)
    elif kwargs.get('method_map', 'KNN') == 'binned_stat':
        results, xbins, ybins, im = binned_statistic_2d(temp.Y.values, temp.Z.values, temp[feature_to_map].values,
                                                     statistic='mean', bins=kwargs.get('bins', 50))
        vmin, vmax = update_vmin_vmax(results, vmin, vmax, nb_iter, kwargs)

    if kwargs.get('plot_arrows', False):
        if kwargs.get('slice', False):
            inputs = (feature_to_slice, min_sectors[i], max_sectors[i], temp, N_neighbours)
        else:
            inputs = (temp, N_neighbours)
        Y, Z, valy, valz = get_arrows_coordinates(*inputs, **kwargs)
        kwargs['arrows_coordinates'] = {'Y': Y, 'Z': Z, 'valy': valy, 'valz': valz}

    if nb_iter == 1:
        a = ax[i // ncols, i % ncols]
        if kwargs.get('method_map','KNN')=='KNN':
            _, _ = plot_maps(temp, results, fig=fig, ax=ax[i // ncols, i % ncols:i % ncols + 2], valid=valid,
                         show_ylabel=show_ylabel, show_colorbar=show_colorbar, **kwargs)
        elif kwargs.get('method_map', 'KNN') == 'binned_stat':
            _, _ = plot_maps(temp, {feature_to_map:results.T}, Ymp=xbins, Zmp=ybins, fig=fig, ax=ax[i // ncols, i % ncols:i % ncols + 2],
                             show_ylabel=show_ylabel, show_colorbar=show_colorbar, **kwargs)

        if feature_to_slice in ['omni_CLA','omni_COA','CLA','COA', 'tilt']:
            title = (f'{feature_to_map}\nfor {round(min_sectors[i]*180/np.pi, 2)}° < {feature_to_slice} < '
                     f'{round(max_sectors[i]*180/np.pi, 2)}°\n{len(temp)} points')
        else:
            title = (f'{feature_to_map}\nfor {round(min_sectors[i], 2)} < {feature_to_slice} < '
                     f'{round(max_sectors[i], 2)}\n{len(temp)} points')
        a.set_title(title, fontsize='medium')  # , fontsize=7)
        if feature_to_slice == 'omni_CLA':
            plot_CLA_sector(12, 14, 2.5, min_sectors[i], max_sectors[i], a)
    return vmin, vmax


def maps_by_sectors(df, feature_to_map, feature_to_slice, **kwargs):
    show_colorbar = kwargs.pop('show_colorbar', True)
    show_ylabel = kwargs.pop('show_ylabel', True)
    nb_sectors, min_sectors, max_sectors = make_sectors(df, feature_to_slice, **kwargs)

    max_distance = kwargs.pop('max_distance', 3)
    N_neighbours = kwargs.pop('N_neighbours', 500)
    ncols = kwargs.get('ncols', 3)

    if 'fig' in kwargs and 'ax' in kwargs:
        fig, ax = kwargs.pop('fig'), kwargs.pop('ax')
        ax = make_ax_2D(ax)
        nrows = len(ax)
        assert len(ax.ravel()) >= nb_sectors + nrows, (f"There are not enough subplots to plot all the slices of "
                                                       f"{feature_to_slice}, plus the colorbars.")
    else:
        nrows = int(np.ceil(nb_sectors / ncols))
        fig, ax = plt.subplots(ncols=ncols + 1, nrows=nrows, figsize=(3 * ncols, 3 * nrows),
                               sharey=True)  # +1 for the colorbars

    ax = make_ax_2D(ax)
    ncols = len(ax[0]) - 1  # to leave the last column for colorbars

    vmin = kwargs.get('vmin', float('inf'))
    vmax = kwargs.get('vmax', -float('inf'))

    for nb_iter in range(2):
        kwargs = manage_vmin_vmax(vmin, vmax, nb_iter, kwargs)

        for i in range(nb_sectors):
            vmin, vmax = compute_one_sector(df, feature_to_map, feature_to_slice, min_sectors, max_sectors, vmin, vmax,
                                            nb_iter, N_neighbours,
                                            max_distance, fig, ax, i, ncols, show_ylabel and ((i % ncols) == 0),
                                            show_colorbar and (((i % ncols) == (ncols - 1)) or i == (nb_sectors - 1)),
                                            kwargs)

    for a in ax[-1, i % ncols + 1:]:
        a.axis('off')

    for a in ax.ravel():
        a.set_aspect('equal')

    ax[0, 0].set_ylim(*kwargs.get('y_lim',(-17, 17)))
    fig.suptitle(kwargs.get('chosen_description', ''))
    fig.tight_layout()
    return fig, ax


def maps_by_sectors_and_ref_MSP_MSH(df, feature_to_map, feature_to_slice, **kwargs):
    show_colorbar = kwargs.pop('show_colorbar', True)
    show_ylabel = kwargs.pop('show_ylabel', True)
    nb_sectors, min_sectors, max_sectors = make_sectors(df, feature_to_slice, **kwargs)

    max_distance = kwargs.pop('max_distance', 3)
    N_neighbours = kwargs.pop('N_neighbours', 500)
    ncols = kwargs.get('ncols', 3)

    if 'fig' in kwargs and 'ax' in kwargs:
        fig, ax = kwargs.pop('fig'), kwargs.pop('ax')
        nrows = len(ax)
        assert len(ax.ravel()) >= nb_sectors + nrows + 2, (f"There are not enough subplots to plot all the slices of "
                                                           f"{feature_to_slice}, plus the colorbars, "
                                                           f"plus reference MSP and MSH.")
    else:
        nrows = int(np.ceil((nb_sectors + 2) / ncols))  # +2 for reference MSP and MSH
        fig, ax = plt.subplots(ncols=ncols + 1, nrows=nrows, figsize=(3 * ncols, 3 * nrows),
                               sharey=True)  # +1 for the colorbars

    ax = make_ax_2D(ax)
    ncols = len(ax[0]) - 1  # to leave the last column for colorbars

    vmin = kwargs.get('vmin', float('inf'))
    vmax = kwargs.get('vmax', -float('inf'))

    min_sectors = np.array([np.nan] + list(min_sectors))
    max_sectors = np.array([np.nan] + list(max_sectors))
    for nb_iter in range(2):
        kwargs = manage_vmin_vmax(vmin, vmax, nb_iter, kwargs)

        for i in range(nb_sectors):
            vmin, vmax = compute_one_sector(df, feature_to_map, feature_to_slice, min_sectors, max_sectors, vmin, vmax,
                                            nb_iter, N_neighbours,
                                            max_distance, fig, ax, i + 1, ncols, show_ylabel and ((i % ncols) == 0),
                                            show_colorbar and ((((i + 1) % ncols) == (ncols - 1)) or i == nb_sectors),
                                            kwargs)

        results, description = get_map(feature_to_map + '_MSP', df, N_neighbours, **kwargs)
        vmin, vmax = update_vmin_vmax(results[feature_to_map + '_MSP'], vmin, vmax, nb_iter, kwargs)
        valid = get_valid('None', 0, 0, description, df, N_neighbours, max_distance, kwargs)

        if nb_iter == 1:
            _, _ = plot_maps(pd.DataFrame(df[['Vtan1_MP_MSP', 'Vtan2_MP_MSP', 'Vn_MP_MSP']].values,
                                          index=df.index.values, columns=['Vtan1_MP', 'Vtan2_MP', 'Vn']),
                             results, valid=valid, fig=fig, ax=ax[0, 0], show_colorbar=False, show_ylabel=True,
                             **kwargs)
            ax[0, 0].set_title(f'{feature_to_map} for KNN\nin neighbouring MSP\n{len(df)} points',
                               fontsize='medium')
            plot_CLA_sector(12, 14, 2.5, kwargs.get('min_cla',0), kwargs.get('max_cla',0), ax[0, 0])

        # MSH
        results, _ = get_map(feature_to_map + '_MSH', df, N_neighbours, **kwargs)
        vmin, vmax = update_vmin_vmax(results[feature_to_map + '_MSH'], vmin, vmax, nb_iter, kwargs)
        if nb_iter == 1:
            _, _ = plot_maps(pd.DataFrame(df[['Vtan1_MP_MSH', 'Vtan2_MP_MSH', 'Vn_MP_MSH']].values,
                                          index=df.index.values, columns=['Vtan1_MP', 'Vtan2_MP', 'Vn']),
                             results, valid=valid, fig=fig, ax=ax.ravel()[nb_sectors + nrows:nb_sectors + nrows + 2],
                             show_ylabel=False, **kwargs)
            ax.ravel()[nb_sectors + nrows].set_title(
                f'{feature_to_map} for KNN\nin neighbouring MSH\n{len(df)} points', fontsize='medium')
            plot_CLA_sector(12, 14, 2.5, kwargs.get('min_cla', 0), kwargs.get('max_cla', 0),
                            ax.ravel()[nb_sectors + nrows])
            # ax.ravel()[nb_sectors+nrows].set_xlim(-17,17)

    for a in ax[-1, (i + 2) % ncols + 1:]:
        a.axis('off')
    ax[0, 0].set_ylim(-17, 17)

    fig.suptitle(kwargs.get('chosen_description', ''))
    fig.tight_layout()
    return fig, ax


def get_arrows_coordinates_from_maps(map_Vtan1, map_Vtan2, **kwargs):
    Xmp, Ymp, Zmp = make_mp_grid(**kwargs)
    _, theta, phi = cartesian_to_spherical(Xmp, Ymp, Zmp)

    theta = theta.ravel()
    phi = phi.ravel()
    vtan1 = map_Vtan1['Vtan1_MP'].ravel()
    vtan2 = map_Vtan2['Vtan2_MP'].ravel()
    vn = np.zeros_like(theta)

    vx, vy, vz = get_cartesian_from_tangential(theta, phi, vtan1, vtan2, vn, **kwargs)
    vy = vy.reshape(map_Vtan1['Vtan1_MP'].shape)
    vz = vz.reshape(map_Vtan1['Vtan1_MP'].shape)
    v = np.sqrt(vy ** 2 + vz ** 2)
    normalized_vy = vy / np.max(v)
    normalized_vz = vz / np.max(v)

    return Ymp, Zmp, normalized_vy, normalized_vz


def get_cartesian_from_tangential(theta, phi, vtan1, vtan2, vn, mp='shue1998',
                                  **kwargs):  # vx, vy, vz are the values of the vector to transform into cartesian coordinates
    if mp == 'shue1998':
        x_normal, y_normal, z_normal = mp_shue1998_normal(theta, phi, **kwargs)
        [x_tan1, y_tan1, z_tan1], [x_tan2, y_tan2, z_tan2] = mp_shue1998_tangents(theta, phi, **kwargs)
    else:
        raise Exception(f"The MP model {mp} has not been implemented yet in get_cartesian_from_tangential.")

    all_vx, all_vy, all_vz = [], [], []
    pb = 0
    for i in range(len(x_tan1)):
        matrix = np.array([[x_tan1[i], y_tan1[i], z_tan1[i]], [x_tan2[i], y_tan2[i], z_tan2[i]],
                           [x_normal[i], y_normal[i], z_normal[i]]])
        try:
            [vx, vy, vz] = np.dot(np.linalg.inv(matrix), np.array([vtan1[i], vtan2[i], vn[i]]))
        except:
            vx, vy, vz = vn[i], vtan2[i], vtan1[i]
        all_vx.append(vx.item())
        all_vy.append(vy.item())
        all_vz.append(vz.item())
    return np.array(all_vx), np.array(all_vy), np.array(all_vz)


def get_arrows_coordinates(*inputs, **kwargs):
    map_Vtan1, _ = get_map('Vtan1_MP', *inputs, **kwargs)
    map_Vtan2, _ = get_map('Vtan2_MP', *inputs, **kwargs)

    # Projection on Vy and Vz (this step has been checked, normalement)
    Ymp, Zmp, Vy, Vz = get_arrows_coordinates_from_maps(map_Vtan1, map_Vtan2, **kwargs)

    return Ymp, Zmp, Vy, Vz


def plot_arrows(ax, Ymp, Zmp, Vy, Vz, step=40, factor=1.5, head_width=0.7):
    for i in range(0, len(Ymp), step):
        for j in range(0, len(Ymp[0]), step):
            ax.arrow(Ymp[i, j], Zmp[i, j], Vy[i, j]*factor, Vz[i, j]*factor, head_width=head_width)


def find_stagnation_line(Vz, **kwargs):
    if 'Y_mp' and 'Z_mp' in kwargs:
        Y_grid, Z_grid = kwargs['Y_mp'], kwargs['Z_mp']
    else:
        _, Y_grid, Z_grid = make_mp_grid(**kwargs)
    assert Y_grid.shape == Vz.shape, f"The coordinate grids must have the same shape as the flow map, but Y_mp is {Y_grid.shape}, Z_mp is {Z_grid.shape} and the map is {Vz.shape}."

    # --- Step 1: Smooth the data to reduce noise ---
    Vz_smooth = gaussian_filter_nan_datas(Vz, kwargs.get('sigma', 5))  # Utiliser ma fonction

    # --- Step 2: Extract the zero contour (stagnation line) ---
    contours = measure.find_contours(Vz_smooth, level=0)

    # Select the longest contour as the main stagnation line
    if contours:
        # stagnation_line = max(contours, key=len)
        stagnation_line = np.concatenate(contours)
    else:
        stagnation_line = None

    if stagnation_line is not None:
        z_idx, y_idx = stagnation_line[:, 0], stagnation_line[:, 1]
        y_line = [Y_grid[int(i), int(j)] for i, j in zip(z_idx, y_idx)]
        z_line = [Z_grid[int(i), int(j)] for i, j in zip(z_idx, y_idx)]
    else:
        y_line, z_line = [], []

    return y_line, z_line


def equal_sample_data(df, feature_to_balance, **kwargs):
    equal_sampled_data = []

    # Hist
    stat, bins = plt.hist(df, feature_to_balance, bins=50)
    # Choose nb of samples per bin (as only that 10% of bins have less than that (except for bins = 0, don't count)
    nb_sample = np.quantile(stat[stat > 0], 0.2)

    # For each bin:
    min_sectors = bins[:-1]
    max_sectors = bins[1:]
    for i, (mini, maxi) in enumerate(zip(min_sectors, max_sectors)):
        if stat[i] > 0:
            temp = make_slice(df, feature_to_balance, mini, maxi)
            # Sample
            if len(temp) > nb_sample:
                bin_sample = resample(temp, replace=False, n_samples=nb_sample, random_state=0)
            elif len(temp) > 0:
                bin_sample = resample(temp, replace=True, n_samples=nb_sample, random_state=0)
            equal_sampled_data.append(bin_sample)

    equal_sampled_data = np.concatenate(equal_sampled_data)
    equal_sampled_data = pd.DataFrame(equal_sampled_data, columns=df.columns.values)

    return equal_sampled_data
