from swapp.engineered_features.gaussian_fits import *
from scipy.stats import binned_statistic_2d
from matplotlib.colors import LogNorm, Normalize


def plot_original_and_normalized_features(df, feature1, feature2, ax, s, alpha, label):
    ax[0].scatter(df[feature1].values, df[feature2].values, label=label, s=s, alpha=alpha)
    # Now normalized feature1 VS feature2 for not BL
    ax[1].scatter(abs(df[feature1].values - df[f"ref_MSH_{feature1}"].values) / df[f"ref_MSH_{feature1}"].values,
                  abs(df[feature2].values - df[f"ref_MSH_{feature2}"].values) / df[f"ref_MSH_{feature2}"].values,
                  label=label, s=s, alpha=alpha)


def plot_effect_feature_MSH_normalization(df, feature1, feature2, **kwargs):
    s = kwargs.get('s', 3)
    alpha = kwargs.get('alpha', 0.01)
    start = kwargs.get('start', df.index.values[0])
    stop = kwargs.get('stop', df.index.values[-1])

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    for classe, label in zip([0, 1], ['Not BL', 'BL']):
        temp = df[df.label_BL.values == classe][start:stop]
        plot_original_and_normalized_features(temp, feature1, feature2, ax, s, alpha, label)

    # Visualization
    ax[0].set_xlabel(feature1)
    ax[0].set_ylabel(feature2)
    ax[1].set_xlabel(f'Normalized {feature1}')
    ax[1].set_ylabel(f'Normalized {feature2}')
    ax[0].legend()
    ax[1].legend()

    # Scales
    xscale = kwargs.get('xscale', 'linear')
    yscale = kwargs.get('yscale', 'linear')
    for i in [0, 1]:
        ax[i].set_xscale(xscale)
        ax[i].set_yscale(yscale)


def make_bins(scale, df, feature, **kwargs):
    if scale == 'log':
        min_value = np.nanmin(df[df[feature].values > 0][feature].values)
        bins = np.logspace(np.log10(min_value), np.log10(np.max(df[feature].values)),
                           kwargs.get('nb_bins', 100))
    else:
        bins = np.linspace(np.min(df[feature].values), np.max(df[feature].values), kwargs.get('nb_bins', 100))
    return bins


def stat_binned_class_separation_plot(data, feature1, feature2, **kwargs):

    df = data.dropna()

    xscale = kwargs.get('xscale', 'linear')
    yscale = kwargs.get('yscale', 'linear')
    binsx = make_bins(xscale, df, feature1, **kwargs)
    binsy = make_bins(yscale, df, feature2, **kwargs)

    if ('fig' not in kwargs) or ('ax' not in kwargs):
        fig, ax = plt.subplots(ncols=3, figsize=(12, 4), sharex=True, sharey=True)
    else:
        fig, ax = kwargs['fig'], kwargs['ax']

    for i, (method, norm) in enumerate(
            zip(['mean', 'median', 'count'], [LogNorm(), Normalize(), LogNorm(vmin=1)])):
        stat, _, _, _ = binned_statistic_2d(df[feature1].values, df[feature2].values, df.label_BL.values,
                                            statistic=method, bins=[binsx, binsy], range=None,
                                            expand_binnumbers=False)

        # Plots
        im = ax[i].pcolormesh(binsx, binsy, stat.T, norm=norm)
        fig.colorbar(im, ax=ax[i])
        ax[i].set_xscale(xscale)
        ax[i].set_yscale(yscale)
        ax[i].set_title(method)
        ax[i].set_xlabel(feature1)
        ax[i].set_ylabel(feature2)


def relative_MSH_normalized_class_separation_plot(df, feature1, feature2, **kwargs):
    for feature in [feature1, feature2]:
        df['relative_gap_with_MSH_' + feature] = abs(df[feature].values - df['ref_MSH_' + feature].values) / df[
            'ref_MSH_' + feature].values
    stat_binned_class_separation_plot(df, 'relative_gap_with_MSH_' + feature1,
                                      'relative_gap_with_MSH_' + feature2, **kwargs)


def gap_to_MSH_class_separation_plot(df, feature1, feature2, **kwargs):
    for feature in [feature1, feature2]:
        df['gap_to_MSH_' + feature] = df[feature].values - df['ref_MSH_' + feature].values
    stat_binned_class_separation_plot(df, 'gap_to_MSH_' + feature1, 'gap_to_MSH_' + feature2, **kwargs)


def median_binned_class(data, feature1, feature2, fig, ax, **kwargs):
    df = data.dropna()
    xscale = kwargs.get('xscale', 'linear')
    yscale = kwargs.get('yscale', 'linear')
    binsx = make_bins(xscale, df, feature1, **kwargs)
    binsy = make_bins(yscale, df, feature2, **kwargs)

    stat, _, _, _ = binned_statistic_2d(df[feature1].values, df[feature2].values, df.label_BL.values,
                                        statistic='median', bins=[binsx, binsy], range=None, expand_binnumbers=False)

    precision, recall = class_2d_separation_scores(data, feature1, feature2)

    # Plots
    im = ax.pcolormesh(binsx, binsy, stat.T, norm=kwargs.get('norm', Normalize()))
    fig.colorbar(im, ax = ax)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_title(f'prec={precision}\nrec={recall}')
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)

    return precision, recall


def class_2d_separation_scores(data, feature1, feature2, **kwargs):
    df = data.dropna().astype('float')

    xscale = kwargs.get('xscale', 'linear')
    yscale = kwargs.get('yscale', 'linear')
    binsx = make_bins(xscale, df, feature1, **kwargs)
    binsy = make_bins(yscale, df, feature2, **kwargs)

    stat_decision, _, _, _ = binned_statistic_2d(df[feature1].values, df[feature2].values, df.label_BL.values,
                                                 statistic='median', bins=[binsx, binsy], range=None,
                                                 expand_binnumbers=False)
    stat_bl, _, _, _ = binned_statistic_2d(df[feature1].values, df[feature2].values, df.label_BL.values,
                                           statistic='sum', bins=[binsx, binsy], range=None, expand_binnumbers=False)
    stat_count, _, _, _ = binned_statistic_2d(df[feature1].values, df[feature2].values, df.label_BL.values,
                                              statistic='count', bins=[binsx, binsy], range=None,
                                              expand_binnumbers=False)
    stat_non_bl = stat_count - stat_bl

    stat_decision = stat_decision.flatten()
    stat_bl = stat_bl.flatten()[~np.isnan(stat_decision)]
    stat_non_bl = stat_non_bl.flatten()[~np.isnan(stat_decision)]
    stat_decision = stat_decision.flatten()[~np.isnan(stat_decision)]

    # Scores
    tp = (stat_bl * stat_decision).sum()
    fp = (stat_non_bl * stat_decision).sum()
    fn = (stat_bl * (1 - stat_decision)).sum()

    precision = round(tp / (tp + fp), 3)
    recall = round(tp / (tp + fn), 3)

    return precision, recall


def scatterplot2d_BL_VS_nonBL(df, col1, col2, **kwargs):
    if 'ax' in kwargs:
        ax = kwargs['ax']
    else:
        _, ax = plt.subplots()

    temp = df[df.isLabelled.values].dropna()
    temp1 = temp[temp.label_BL.values == 0]
    ax.scatter(temp1[col1].values, temp1[col2].values, s=20, color='blue', label='not BL')
    temp2 = temp[temp.label_BL.values == 1]
    ax.scatter(temp2[col1].values, temp2[col2].values, s=10, color='red', label='BL')
    ax.legend()
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)


def hist1d_BL_VS_nonBL(df, feature, **kwargs):
    df = df.copy()[[feature, 'label_BL']].dropna()
    if 'bins' in kwargs:
        if isinstance(kwargs['bins'], int):
            kwargs['nb_bins'] = kwargs.pop('bins')
            bins = make_bins(kwargs.get('xscale','linear'), df, feature, **kwargs)
        elif isinstance(kwargs['bins'], np.ndarray):
            bins = kwargs['bins']
        else:
            raise Exception(f"If given as kwargs, bins should be either an int or an array, but is {kwargs['bins']}.")
    else:
        bins = make_bins(kwargs.get('xscale', 'linear'), df, feature, **kwargs)

    if 'ax' in kwargs:
        ax = kwargs['ax']
    else:
        _, ax = plt.subplots()

    temp1 = df[df.label_BL.values == 0]
    _ = ax.hist(temp1[feature].values, bins=bins, color='blue', label='not BL', alpha=0.5, density=True)
    temp2 = df[df.label_BL.values == 1]
    _ = ax.hist(temp2[feature].values, bins=bins, color='red', label='BL', alpha=0.5, density=True)
    ax.legend()
    ax.set_xlabel(feature)
    ax.set_ylabel('Points count (normalized)')
    ax.set_xscale(kwargs.get('xscale', 'linear'))
    ax.set_yscale(kwargs.get('yscale', 'linear'))
