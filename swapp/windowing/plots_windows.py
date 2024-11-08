import matplotlib
import matplotlib.pyplot as plt
from .make_windows.utils import select_windows
import numpy as np
from spok.models import planetary
from spok.plot import planet_env
from scipy.stats import binned_statistic_2d


def diagnostic_windows(df, pos, omni, conditions, **kwargs):
    if isinstance(conditions, str):
        conditions = [conditions]

    nbr_slices = max(np.sum([1 for arg in ['x_slice', 'y_slice', 'z_slice'] if arg in kwargs]), 1)
    fig, axes = _make_figure_for_diagnostic(omni, conditions, **kwargs)

    slice_kwargs = {arg: kwargs.pop(arg) for arg in ['x_slice', 'y_slice', 'z_slice'] if arg in kwargs}
    for i, condition in enumerate(conditions):
        subaxes = axes[:len(omni.columns)]
        fig, subaxes = plot_characteristics_windows_with_condition(omni, df, condition, figure=fig, axes=subaxes,
                                                                   **kwargs)
        subaxes = get_pos_condition_subaxes(axes, len(conditions)-1-i, nbr_slices)
        fig, subaxes = plot_pos_windows_with_condition(fig, subaxes, pos, df, condition, **slice_kwargs, **kwargs)

    plt.tight_layout()
    return fig, axes


def plot_hist_2D(df, x, y, ax, **kwargs):
    # Give colors because automatic ones are ugly.

    assert isinstance(ax, matplotlib.axes._axes.Axes), "The ax given as input must be of matplotlib type Axe."

    bins = kwargs.pop('bins', 100)
    cmap = kwargs.pop('cmap', 'jet')
    hist = ax.hist2d(df[x].values, df[y].values, cmin=1, bins=bins, cmap=cmap, range=[[-40, 40], [-40, 40]],
                     **kwargs)
    plt.colorbar(hist[3], ax=ax)
    return ax


def plot_hist_1D(df, X, ax, **kwargs):
    assert isinstance(ax, matplotlib.axes._axes.Axes), "The ax given as input must be of matplotlib type Axe."

    bins = kwargs.pop('bins', 50)
    alpha = kwargs.pop('alpha', 0.4)
    if len(df.dropna()) > 0:
        v_min = np.quantile(df.dropna()[X].values, 0.01)
        v_max = np.quantile(df.dropna()[X].values, 0.99)
        density = True
    else:
        v_min, v_max = -1, 1
        density = False
    ax.hist(df[X].values, density=density, bins=bins, alpha=alpha, range=[v_min, v_max], **kwargs)
    ax.set_xlabel(X)
    ax.set_ylabel("Normalized count")

    return ax


def plot_pos_hist(pos, fig, ax, **kwargs):           # Transform with the slices kwargs to make it more general
    """
    Precondition : ax must have a length of three.
    """
    assert isinstance(ax, np.ndarray), "The ax given as input must be an array."
    assert isinstance(ax[0], matplotlib.axes._axes.Axes), "The ax given as input must an array of Axes."
    assert len(ax) == 3, "The array of Axes given as input must have a length of three."

    to_plot = pos.dropna()
    ax[0] = plot_hist_2D(to_plot, 'X', 'Y', ax[0], **kwargs)
    ax[1] = plot_hist_2D(to_plot, 'X', 'Z', ax[1], **kwargs)
    ax[2] = plot_hist_2D(to_plot, 'Y', 'Z', ax[2], **kwargs)
    plt.tight_layout()


def plot_pos_windows_with_condition(fig, ax, pos, df, condition, **kwargs):
    """
    Precondition : ax must have a length of three.
    """
    windows = select_windows(df, condition)
    subpos = pos.loc[windows.index.values]
    plot_pos_hist(subpos, fig, ax.flatten(), label=condition)

    msh = planetary.Magnetosheath(magnetopause='mp_shue1998', bow_shock='bs_jelinek2012')
    fig, ax = planet_env.layout_earth_env(msh, figure=fig, axes=ax, x_lim=(-2, 25), **kwargs)

    for a in ax.flatten():
        if isinstance(condition, str):
            title = condition
        else:
            title = ''
            for cond in condition[:-1]:
                title += cond+' & \n'
            title += condition[-1]
        a.set_title(title)

    plt.tight_layout()
    return fig, ax


def _make_axes_array(axes):
    if isinstance(axes, np.ndarray):
        return axes.ravel()
    elif isinstance(axes, dict):
        return np.array(list(axes.values()))
    else:
        raise Exception("Axes must be a dictionary or an array.")


def plot_characteristics_windows_with_condition(df_characteristics, df, condition, **kwargs):
    windows = select_windows(df, condition)
    to_plot = df_characteristics.loc[windows.index.values]

    nbr_features = len(to_plot.columns)

    ########################################################################
    # I don't know why I can't extract this part in a function! Ask Nico
    if 'axes' in kwargs and 'figure' in kwargs:    
        fig = kwargs.pop('figure')
        axes = kwargs.pop('axes')
        axes = _make_axes_array(axes)
        assert axes.size >= nbr_features, "The given axes' size must match the number of features to plot."
    else:
        ncols = kwargs.get('ncols', 5)
        nrows = int(np.ceil(nbr_features / ncols))
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5 * nrows, 5 * ncols))

    assert isinstance(axes, np.ndarray), "The created axes must be an array."
    assert axes.size >= nbr_features, ("The array of Axes given as input must have the same length as the "
                                       "number of features.")
    axes = axes.flatten()[:nbr_features]

    for i, (ax, feature) in enumerate(zip(axes, to_plot.columns)):
        ax = plot_hist_1D(to_plot, feature, ax, label=condition, **kwargs)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncols=2)

    return fig, axes


def get_pos_condition_subaxes(axes, i, nbr_slices):
    axes = axes.flatten()
    if i == 0:
        subaxes = axes[-nbr_slices:]
    else:
        subaxes = axes[-(i + 1) * nbr_slices:-i * nbr_slices]
    return subaxes


def _make_figure_for_diagnostic(omni, conditions, **kwargs):
    if 'axes' in kwargs and 'figure' in kwargs:
        axes = kwargs.pop('axes')
        fig = kwargs.pop('figure')
    else:
        ncols = kwargs.get('ncols', 5)
        nrows = np.ceil(len(omni.columns) / ncols) + len(conditions)
        # The *2 is because the position plots are twice as high
        fig, axes = plt.subplot_mosaic(mosaic_structure(omni, len(conditions), **kwargs),
                                       figsize=(5 * ncols, 5 * nrows))
        axes = np.array(list(axes.values()))

    for i in range(1, len(conditions)):
        nbr_slices = max(np.sum([1 for arg in ['x_slice', 'y_slice', 'z_slice'] if arg in kwargs]), 1)
        for j in range(nbr_slices):
            index = - i*nbr_slices - j - 1
            axes[index].sharex(axes[-1-j])
            axes[index].sharey(axes[-1-j])

    return fig, axes


def mosaic_structure(df, nbr_conditions, **kwargs):
    # alphabet = list(string.ascii_uppercase)
    nbr_slices = max(np.sum([1 for arg in ['x_slice', 'y_slice', 'z_slice'] if arg in kwargs]), 1)
    # mosaic = ""
    mosaic = []
    nbr_features = len(df.columns)
    ncols = kwargs.get('ncols', 5)
    nrows = int(np.ceil(nbr_features/ncols))
    count = 0
    for i in range(nrows):
        temp = []
        for j in range(ncols):
            # mosaic += alphabet[count]*nbr_slices
            temp += [str(count)]*nbr_slices
            count += 1
        # mosaic += '\n'
        mosaic += [temp]
    for i in range(nbr_conditions):
        # add = ''
        add = []
        for j in range(nbr_slices):
            # add += alphabet[count]*ncols
            add += [str(count)]*ncols
            count += 1
        # mosaic += add+'\n'+add+'\n'   #This line is added twice to increase the heigth of the plots
        # mosaic += add+'\n'
        mosaic += [add]
    return mosaic


def binned_stat(valuesx, valuesy, valuesz, ax, **kwargs):
    stat, binsx, binsy, _ = binned_statistic_2d(valuesx, valuesy, valuesz, bins=kwargs.pop('bins', 100),
                                                statistic=kwargs.pop('statistic', 'median'))
    if 'vmin' in kwargs:
        v_min = kwargs['vmin']
    elif kwargs.pop('cmap', 'seismic') == 'seismic':
        v_min = -np.nanmax(abs(stat))
    else:
        v_min = np.nanmin(stat)

    if 'vmax' in kwargs:
        v_max = kwargs['vmax']
    elif kwargs.pop('cmap', 'seismic') == 'seismic':
        v_max = np.nanmax(abs(stat))
    else:
        v_max = np.nanmax(stat)

    im = ax.pcolormesh(binsx, binsy, stat.T, cmap=kwargs.pop('cmap', 'seismic'), vmin=v_min, vmax=v_max)
    plt.colorbar(im)


def min_max_stat_bin(valuesx, valuesy, valuesz, vmin_total, vmax_total, **kwargs):
    stat, binsx, binsy, _ = binned_statistic_2d(valuesx, valuesy, valuesz, bins=kwargs.get('bins', 100),
                                                statistic=kwargs.get('statistic', 'median'))
    vmin_total = min(vmin_total, np.nanmin(stat))
    vmax_total = max(vmax_total, np.nanmax(stat))
    return vmin_total, vmax_total


def find_common_range(values, pos, **kwargs):
    if 'vmax' in kwargs and 'vmin' in kwargs:
        return kwargs['vmin'], kwargs['vmax']

    vmin_total, vmax_total = np.nanmax(values), np.nanmin(values)
    if 'z_slice' in kwargs:
        vmin_total, vmax_total = min_max_stat_bin(pos.X.values, pos.Y.values, values, vmin_total, vmax_total, **kwargs)

    if 'y_slice' in kwargs:
        vmin_total, vmax_total = min_max_stat_bin(pos.X.values, pos.Z.values, values, vmin_total, vmax_total, **kwargs)

    if 'x_slice' in kwargs:
        vmin_total, vmax_total = min_max_stat_bin(pos.Y.values, pos.Z.values, values, vmin_total, vmax_total, **kwargs)

    return vmin_total, vmax_total


def planet_env_stat_binned(feature, df, pos, **kwargs):
    if (df.index.values != pos.index.values).sum() > 0:
        df = df[df.index.isin(pos.index.values)]
        pos = pos[pos.index.isin(df.index.values)]
    values = df[feature].values
    norm = kwargs.pop('norm', 'linear')
    if norm == 'log':
        values = np.log10(values)

    vmin, vmax = find_common_range(values, pos, **kwargs)
    print(f'vmin={vmin}, vmax={vmax}')
    vmin = kwargs.pop('vmin', vmin)
    vmax = kwargs.pop('vmax', vmax)

    ncols = 0
    for str_slice in ['x_slice', 'y_slice', 'z_slice']:
        if str_slice in kwargs:
            ncols += 1
    fig, ax = plt.subplots(ncols=ncols, figsize=(5 * ncols, 5))
    if isinstance(ax, np.ndarray) is False:
        ax = np.array([ax])

    col = 0
    if 'z_slice' in kwargs:
        binned_stat(pos.X.values, pos.Y.values, values, ax[col], vmin=vmin, vmax=vmax, **kwargs)
        col += 1
    if 'y_slice' in kwargs:
        binned_stat(pos.X.values, pos.Z.values, values, ax[col], vmin=vmin, vmax=vmax, **kwargs)
        col += 1
    if 'x_slice' in kwargs:
        binned_stat(pos.Y.values, pos.Z.values, values, ax[col], vmin=vmin, vmax=vmax, **kwargs)

    msh = planetary.Magnetosheath(magnetopause='mp_shue1998', bow_shock='bs_jelinek2012')
    _, _ = planet_env.layout_earth_env(msh, figure=fig, axes=ax, x_lim=(-2, 25), **kwargs)

    title = feature + ' ' + kwargs.get('title', '')
    if norm == 'log':
        title += ', in log norm'
    fig.suptitle(title)

    fig.tight_layout()
