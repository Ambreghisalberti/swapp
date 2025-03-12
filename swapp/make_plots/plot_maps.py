from spok.models import planetary
import matplotlib
from spok.coordinates.coordinates import cartesian_to_spherical, spherical_to_cartesian
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from swapp.make_plots.plot_functions import make_bins
from spok.models.planetary import mp_shue1997
from matplotlib.colors import LogNorm
from sklearn.neighbors import KNeighborsRegressor
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from spok.models.planetary import Magnetosheath
from spok.plot import planet_env
import matplotlib.patches as mpatches
import pandas as pd
import os
from datetime import timedelta


def add_spherical_coordinates(df):
    R, theta, phi = cartesian_to_spherical(df.X.values, df.Y.values, df.Z.values)
    df['R'] = R
    df['theta'] = theta
    df['phi'] = phi
    return df


def gaussian_filter_nan_datas(df, sigma):
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


def plot_relative_diff_pannel(df, all_pos, featurex, featurey, fig, bins, sigma, cmap, ax):
    stat, xbins, ybins, _ = binned_statistic_2d(all_pos[featurex].values, all_pos[featurey].values,
                                                all_pos[featurex].values, statistic='count', bins=bins)
    stat2, xbins, ybins, _ = binned_statistic_2d(df[featurex].values, df[featurey].values, df[featurey].values,
                                                 bins=(xbins, ybins), statistic='count')
    stat[np.isnan(stat)] = 0
    stat2[np.isnan(stat2)] = 0

    relative_diff = 2*(stat2.T - stat.T)/ (stat.T + stat2.T)
    relative_diff[relative_diff == np.float64('inf')] = np.nan
    relative_diff[relative_diff == -np.float64('inf')] = np.nan
    relative_diff = gaussian_filter_nan_datas(relative_diff, sigma)
    im = ax.pcolormesh(xbins, ybins, relative_diff, cmap=cmap)
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

    inputs = (df, df2, 'X', 'Y', fig, bins, kwargs.get('sigma', 0), cmap)
    f, inputs = get_plot_panel_info(method, inputs)

    i = 0
    if 'z_slice' in kwargs:
        f(inputs, ax[i])
        i += 1
    if 'y_slice' in kwargs:
        f(inputs, ax[i])
        i += 1
    if 'x_slice' in kwargs:
        f(inputs, ax[i])
        i += 1

    msh = planetary.Magnetosheath(magnetopause='mp_shue1998', bow_shock='bs_jelinek2012')
    fig, ax = planet_env.layout_earth_env(msh, figure=fig, axes=np.array([ax]), x_lim=(-2, 25), **kwargs)
    if 'title' in kwargs:
        fig.suptitle(kwargs['title'])


def plot_panel(to_plot, featurex, featurey, fig, bins, cmap, sigma, ax, **kwargs):
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
        plot_panel(to_plot, 'X', 'Y', fig, bins, cmap, sigma, ax[i], **kwargs)
        i += 1
    if 'y_slice' in kwargs:
        plot_panel(to_plot, 'X', 'Z', fig, bins, cmap, sigma, ax[i], **kwargs)
        i += 1
    if 'x_slice' in kwargs:
        plot_panel(to_plot, 'Y', 'Z', fig, bins, cmap, sigma, ax[i], **kwargs)
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


def show_evolution_for_BL_depth(BL, feature, scale='linear'):
    plt.figure()
    bins = make_bins(scale, BL[[feature]].dropna(), feature, nb_bins=200)
    _ = plt.hist2d(BL.dropna().normalized_logNoverT.values, BL.dropna()[feature].values.flatten(),
                   bins=(np.linspace(-0.2, 1.2, 200), bins), cmin=1)
    plt.xlim((-0.2, 1.2))
    plt.yscale(scale)
    plt.xlabel('Transition parameter')
    plt.ylabel(feature)
    plt.axvline(0, color='r', linestyle='--', linewidth=5)
    plt.axvline(1, color='r', linestyle='--', linewidth=5)
    plt.colorbar()
    plt.xticks(ticks=[0, 1], labels=['MSP','MSH'])
    plt.tight_layout()


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


def plot_repositionned_pannel(temp, featurex, featurey, f, fig, ax, cmap, bins, sigma):
    stat, xbins, ybins, _ = binned_statistic_2d(temp[featurex].values, temp[featurey].values,
                                                f(temp), statistic='median', bins=bins)
    stat = gaussian_filter_nan_datas(stat, sigma)
    if cmap == 'seismic':
        vmax = np.nanmax(abs(stat))
        vmin = -vmax
    else:
        vmax, vmin = None, None
    im = ax.pcolormesh(xbins, ybins, stat.T, vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(im, ax=ax)


def plot_repositionned_stat_binned(df, feature, **kwargs):
    zscale = kwargs.get('zscale', 'linear')
    cmap = kwargs.get('cmap', 'jet')
    sigma = kwargs.get('sigma', 0)
    bins = kwargs.get('bins', 100)

    if kwargs.get('cmap') == 'seismic' and kwargs.get('zscale') == 'log':
        raise Exception(
            'The seismic cmap was designed in this function to be centered around 0, '
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
    i = 0
    if 'z_slice' in kwargs:
        temp = df[abs(df.normalized_Z.values) < 1]
        plot_repositionned_pannel(temp, 'normalized_X', 'normalized_Y', f, fig, ax[i], cmap, bins, sigma)
        i += 1

    if 'y_slice' in kwargs:
        temp = df[abs(df.normalized_Y.values) < 1]
        plot_repositionned_pannel(temp, 'normalized_X', 'normalized_Z', f, fig, ax[i], cmap, bins, sigma)
        i += 1

    if 'x_slice' in kwargs:
        plot_repositionned_pannel(df, 'normalized_Y', 'normalized_Z', f, fig, ax[i], cmap, bins, sigma)

    msh = planetary.Magnetosheath(magnetopause='mp_shue1998', bow_shock='bs_jelinek2012')
    fig, ax = planet_env.layout_earth_env(msh, figure=fig, axes=np.array([ax]), x_lim=(-2, 25), **kwargs)
    if 'title' in kwargs:
        fig.suptitle(kwargs['title'])
    else:
        fig.suptitle(f'{feature} for data repositioned with transition parameter\nCuts around plane +/- 1Re are shown')
    fig.tight_layout()


def make_mp_grid(**kwargs):
    N_grid = kwargs.get('N_grid', 300)
    coord = kwargs.get('coord','spherical')
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
        model.fit(np.array([Y,Z]).T, X)
        Xmp = model.predict(np.array([Ymp, Zmp]).T)
        Xmp = Xmp.reshape((N_grid,N_grid))
        Ymp = Ymp.reshape((N_grid,N_grid))
        Zmp = Zmp.reshape((N_grid,N_grid))

    return Xmp, Ymp, Zmp


def make_data_to_grid(df, **kwargs):
    features = kwargs.get('features', list(df.columns.values))
    data = df[np.unique(np.array(features + ['X', 'Y', 'Z']))].dropna()
    values = data[features]
    pos = data[['X', 'Y', 'Z']].values
    return pos, values


def train_knn(x, y, N=10000):
    model = KNeighborsRegressor(n_neighbors=N, weights='distance', n_jobs=1)
    model.fit(x, y)
    return model


def compute_knn(knn_function, pos, values, p0, cpu=40, **kwargs):
    interp = train_knn(pos, values, N=kwargs.get('N_neighbours', 10000))
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

    pos, values = make_data_to_grid(df, **kwargs)

    interpolated_values = compute_knn(f_interp, pos, values, np.asarray([Xmp, Ymp, Zmp]).T, **kwargs).T
    interpolated_values = {feat: values for feat, values in
                           zip(kwargs.get('features', list(df.columns.values)), interpolated_values)}
    return interpolated_values


def is_map_valid(df, **kwargs):
    Xmp, Ymp, Zmp = make_mp_grid(**kwargs)

    _ = kwargs.pop('features', None)
    pos, values = make_data_to_grid(df, features=['X'], **kwargs)
    interp = train_knn(pos, values, N=kwargs.get('N_neighbours', 10000))

    max_distance = kwargs.get('max_distance', 5)
    valid = True * np.ones(Xmp.shape)
    for i in range(Xmp.shape[0]):
        for j in range(Xmp.shape[1]):
            distances, indices = interp.kneighbors([[Xmp[i, j], Ymp[i, j], Zmp[i, j]]])
            if np.max(distances) > max_distance:
                valid[i, j] = False
    return valid


def plot_maps(interpolated_features, **kwargs):
    features = kwargs.get('features', list(interpolated_features.keys()))
    Xmp, Ymp, Zmp = make_mp_grid(**kwargs)
    valid = kwargs.get('valid', np.ones(Xmp.shape))

    if 'fig' not in kwargs or 'ax' not in kwargs:
        ncols = kwargs.get('ncols', 5)
        if len(features) < ncols:
            ncols = len(features)
        nrows = int(np.ceil(len(features) / ncols))
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(3 * ncols, 3 * nrows))
    else:
        fig, ax = kwargs['fig'], kwargs['ax']
    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])
    if len(ax.shape) == 1:
        ax = np.array([ax])
    nrows, ncols = ax.shape

    msh = Magnetosheath(magnetopause='mp_shue1998', bow_shock='bs_jelinek2012')
    for i, feature in enumerate(features):
        to_plot = interpolated_features[feature].copy()
        to_plot[valid == 0] = np.nan
        ax[i // ncols, i % ncols].set_title(feature)
        kwargsplot = {'cmap': kwargs.get('cmap')}
        if kwargs.get('cmap') == 'seismic':
            kwargsplot['vmax'] = abs(np.nanmax(to_plot))
            kwargsplot['vmin'] = -abs(np.nanmax(to_plot))

        im = ax[i // ncols, i % ncols].pcolormesh(Ymp, Zmp, to_plot, **kwargsplot)
        plt.colorbar(im, ax=ax[i // ncols, i % ncols])

        fig, ax[i // ncols, i % ncols] = planet_env.layout_earth_env(msh, figure=fig,
                                                                     axes=np.array([ax[i // ncols, i % ncols]]),
                                                                     y_lim=(-17, 17), z_lim=(-15, 15), x_slice=0)
    fig.tight_layout()
    return fig, ax


def compute_and_plot_map(df, **kwargs):
    results = make_maps(df, **kwargs)
    valid = is_map_valid(df, **kwargs)
    _,_ = plot_maps(results, valid=valid, **kwargs)
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
    arc_patch((cx, cy), r, theta1, theta2, ax=ax, fill=True, color='red', alpha=0.3)


def maps_by_CLA_sector(df, feature, **kwargs):
    max_distance = kwargs.get('max_distance', 3)
    N_neighbours = kwargs.get('N_neighbours', 500)
    nb_sectors = kwargs.get('nb_sectors', 9)
    ncols = kwargs.get('ncols', 3)
    coord = kwargs.get('coord', 'spherical')

    sectors_CLA = np.linspace(-np.pi, np.pi, nb_sectors + 1)

    nrows = int(np.ceil(nb_sectors / ncols))
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(3 * ncols, 3 * nrows))
    for i in range(nb_sectors):
        temp = df[df.omni_CLA.values >= sectors_CLA[i]]
        temp = temp[temp.omni_CLA.values < sectors_CLA[i + 1]]

        path = (f'/home/ghisalberti/Maps/{feature}_CLA_{sectors_CLA[i]}_{sectors_CLA[i + 1]}_'
                f'Nneighbours={N_neighbours}_coord={coord}.pkl')
        if os.path.isfile(path):
            results = pd.read_pickle(path)
        else:
            results = make_maps(temp, features=[feature], N_neighbours=N_neighbours)
            pd.to_pickle(results, path)

        # Validity
        path = (f'/home/ghisalberti/Maps/validity_CLA_{sectors_CLA[i]}_{sectors_CLA[i + 1]}_'
                f'Nneighbours={N_neighbours}_maxdistance={max_distance}_coord={coord}.pkl')
        if os.path.isfile(path):
            valid = pd.read_pickle(path)
        else:
            valid = is_map_valid(temp, N_neighbours=N_neighbours, max_distance=max_distance)
            pd.to_pickle(valid, path)
        fig, ax[i // ncols, i % ncols] = plot_maps(results, fig=fig, ax=ax[i // ncols, i % ncols], valid=valid, **kwargs)
        ax[i // ncols, i % ncols].set_title(
            f'{feature}\nfor {round(sectors_CLA[i], 2)} < CLA < {round(sectors_CLA[i + 1], 2)}\n{len(temp)} points')
        plot_CLA_sector(14, 12, 2.5, sectors_CLA[i], sectors_CLA[i + 1], ax[i // ncols, i % ncols])

    fig.tight_layout()
    return fig, ax


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
            f'/home/ghisalberti/Maps/validity_CLA_{sectors_CLA[i]}_{sectors_CLA[i + 1]}_'
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
        # vxmean = abs(omni.Vx.rolling(dt,min_periods=1).mean())
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
    lag = np.array(np.round((BS_x0.values.flatten()-X_sat.values.flatten())*6371/vx),dtype='timedelta64[s]')
    time = (X_sat.index-lag).round(sampling_time)
    OMNI = pd.DataFrame(columns=omni.columns)
    OMNI = pd.concat([OMNI, omni.loc[time]])
    OMNI.index = X_sat.index
    return OMNI
