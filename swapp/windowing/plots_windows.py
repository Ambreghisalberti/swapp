import matplotlib
import matplotlib.pyplot as plt
from .make_windows.utils import select_windows
import numpy as np
import string
from space.models import planetary
from space.plot import planet_env


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


def plot_hist_2D(df, X, Y, ax, **kwargs):
    # Give colors because automatic ones are ugly.

    assert isinstance(ax, matplotlib.axes._axes.Axes), "The ax given as input must be of matplotlib type Axe."

    bins = kwargs.pop('bins', 100)
    cmap = kwargs.pop('cmap', 'jet')
    hist = ax.hist2d(df[X].values, df[Y].values, cmin=1, bins=bins, cmap=cmap, range=[[-40,40],[-40,40]],
                     **kwargs)
    plt.colorbar(hist[3], ax=ax)
    return ax


def plot_hist_1D(df, X, ax, **kwargs):
    assert isinstance(ax, matplotlib.axes._axes.Axes), "The ax given as input must be of matplotlib type Axe."

    bins = kwargs.pop('bins', 50)
    alpha = kwargs.pop('alpha', 0.4)
    vmin = np.quantile(df.dropna()[X].values,0.01)
    vmax = np.quantile(df.dropna()[X].values,0.99)
    ax.hist(df[X].values, density=True, bins=bins, alpha=alpha, range=[vmin,vmax], **kwargs)
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
    #return fig, ax
    plt.tight_layout()


def plot_pos_windows_with_condition(fig, ax, pos, df, condition, **kwargs):
    """
    Precondition : ax must have a length of three.
    """
    windows = select_windows(df, condition)
    subpos = pos.loc[windows.index.values]
    #fig, ax = plot_pos_hist(subpos, fig, ax.flatten(), label=condition)
    plot_pos_hist(subpos, fig, ax.flatten(), label=condition)

    msh = planetary.Magnetosheath(magnetopause='mp_shue1998', bow_shock='bs_jelinek2012')
    fig, ax = planet_env.layout_earth_env(msh, figure=fig, axes=ax, x_lim=(-2,25), **kwargs)

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


'''
def _make_figure_for_features(nbr_features, **kwargs):
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
    assert axes.size == nbr_features, ("The array of Axes given as input must have the same length as the "
                                       "number of features.")
    axes = axes.flatten()[:nbr_features]
    ##########################################################################
    return fig, axes
'''

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
    assert axes.size == nbr_features, ("The array of Axes given as input must have the same length as the "
                                       "number of features.")
    axes = axes.flatten()[:nbr_features]
    ##########################################################################

    #fig, axes = _make_figure_for_features(nbr_features, **kwargs)

    for i, (ax, feature) in enumerate(zip(axes, to_plot.columns)):
        ax = plot_hist_1D(to_plot, feature, ax, label=condition, **kwargs)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1))

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
        #nrows = np.ceil(len(omni.columns) / ncols) + len(conditions)*2
        # The *2 is because the position plots are twice as high
        fig, axes = plt.subplot_mosaic(mosaic_structure(omni, len(conditions), **kwargs),
                                       figsize=(3 * ncols, 3 * nrows))
        axes = np.array(list(axes.values()))

    for i in range(1,len(conditions)):
        nbr_slices = max(np.sum([1 for arg in ['x_slice', 'y_slice', 'z_slice'] if arg in kwargs]), 1)
        for j in range(nbr_slices):
            index = - i*nbr_slices - j - 1
            axes[index].sharex(axes[-1-j])
            axes[index].sharey(axes[-1-j])

    return fig, axes


def mosaic_structure(df, nbr_conditions, **kwargs):
    #alphabet = list(string.ascii_uppercase)
    nbr_slices = max(np.sum([1 for arg in ['x_slice', 'y_slice', 'z_slice'] if arg in kwargs]), 1)
    #mosaic = ""
    mosaic = []
    nbr_features = len(df.columns)
    ncols = kwargs.get('ncols', 5)
    nrows = int(np.ceil(nbr_features/ncols))
    count = 0
    for i in range(nrows):
        temp = []
        for j in range(ncols):
            #mosaic += alphabet[count]*nbr_slices
            temp += [str(count)]*nbr_slices
            count += 1
        #mosaic += '\n'
        mosaic += [temp]
    for i in range(nbr_conditions):
        #add = ''
        add = []
        for j in range(nbr_slices):
            #add += alphabet[count]*ncols
            add += [str(count)]*ncols
            count += 1
        #mosaic += add+'\n'+add+'\n'   #This line is added twice to increase the heigth of the plots
        #mosaic += add+'\n'
        mosaic += [add]
    return mosaic
