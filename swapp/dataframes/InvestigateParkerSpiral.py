import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
import numpy as np
import pandas as pd


def get_label(feature):
    if feature == 'CLA':
        return r'$\theta_{cl}$ (°)'
    if feature == 'COA':
        return r'$\theta_{co}$ (°)'
    if feature == 'Ma':
        return r'$\M_a$'
    if feature == 'Beta':
        return r'$\Beta$'
    else:
        return feature


def get_title(feature):
    if feature == 'B/Bimf':
        return r'$\frac{B}{B_{IMF}}$'
    if feature == 'N/Nimf':
        return r'$\frac{N}{N_{IMF}}$'
    else:
        return feature


def plot_stat_binned_reconnection_evidence(featurex, featurey, featurez, df, nbins=15):
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

    xlabel = get_label(featurex)
    ylabel = get_label(featurey)
    for i in range(2):
        ax[i].set_xlabel(xlabel)
        ax[i].set_ylabel(ylabel)

    # Feature repartition
    stat, xbins, ybins, _ = binned_statistic_2d(df[featurex].values.flatten(), df[featurey].values.flatten(),
                                                df[featurez].values.flatten(), statistic='mean', bins=nbins)
    im = ax[0].pcolormesh(xbins, ybins, stat.T)
    fig.colorbar(im, ax=ax[0])
    title = get_title(featurez)
    ax[0].set_title(title)

    # Also count nb points in bins!
    stat, xbins, ybins, _ = binned_statistic_2d(df[featurex].values.flatten(), df[featurey].values.flatten(),
                                                df[featurez].values.flatten(), statistic='count', bins=nbins)
    im = ax[1].pcolormesh(xbins, ybins, stat.T)
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title('Point count')

    fig.tight_layout()


def get_posCLA_negCOA(df):
    temp = df[df.CLA.values > 0]
    temp = temp[temp.COA.values < 0]
    return temp


def get_negCLA_posCOA(df):
    temp = df[df.CLA.values < 0]
    temp = temp[temp.COA.values > 0]
    return temp


def compute_diff_parker_spiral_orientations(df):
    posCLA_negCOA = get_posCLA_negCOA(df)
    negCLA_posCOA = get_negCLA_posCOA(df)
    diff = len(posCLA_negCOA) - len(negCLA_posCOA)
    try:
        ratio = len(posCLA_negCOA) / len(negCLA_posCOA)
    except:
        ratio = np.nan
    nb_points = len(posCLA_negCOA) + len(negCLA_posCOA)
    try:
        relative_diff = round(diff / nb_points, 2)
    except:
        relative_diff = np.nan
    try:
        proportion = round(len(posCLA_negCOA) / nb_points, 2)
    except:
        proportion = np.nan
    return diff, ratio, nb_points, relative_diff, proportion


def compute_temporal_diff_parker_spiral_orientations(df, starts, stops, verbose=False, **kwargs):
    all_diff, all_ratio, all_nb_points, all_relative_diff, all_proportion = [], [], [], [], []
    if verbose:
        title = kwargs.get('title')
        print(f'For {title}, the proportion of posCLA & negCOA is of:')
    for start, stop in zip(starts, stops):
        diff, ratio, nb_points, relative_diff, proportion = compute_diff_parker_spiral_orientations(df[start:stop])
        all_diff += [diff]
        all_ratio += [ratio]
        all_nb_points += [nb_points]
        all_relative_diff += [relative_diff]
        all_proportion += [proportion]
        if verbose:
            print(f'- {proportion} between {start} and {stop}, for a total of {nb_points} points.')
    if verbose:
        print('\n')
    return all_diff, all_ratio, all_nb_points, all_relative_diff, all_proportion


def create_empty_comparative_parker_array(starts, stops):
    return pd.DataFrame([], index=['OMNI', 'MMS windows', 'subsolar', 'detected BL'],
                        columns=[f'{start} to {stop}' for start, stop in zip(starts, stops)])


def create_empty_comparative_parker_scores_arrays(starts, stops, nb_metrics=5):
    return (*[create_empty_comparative_parker_array(starts, stops) for _ in range(nb_metrics)],)


def fill_comparative_scores_parker(arrays, scores, entry):
    all_diff, all_ratio, all_nb_points, all_relative_diff, all_proportion = scores
    diffs, ratios, nb_points, relative_diffs, proportions = arrays

    diffs.loc[entry, :] = all_diff
    ratios.loc[entry, :] = all_ratio
    nb_points.loc[entry, :] = all_nb_points
    relative_diffs.loc[entry, :] = all_relative_diff
    proportions.loc[entry, :] = all_proportion

    return diffs, ratios, nb_points, relative_diffs, proportions


def compare_temporal_bias_parker_spiral(df, omni, starts, stops, verbose=False):
    """
    Takes as input the list of stops and starts on which the dataframe should be sliced, to compare
    on each temporal period.
    Takes as input a dataframe that contains only the windows of interest of MMS, but all of them
    (no additional conditions).
    This dataframe needs to contain the following columns : pred_BL, Y, Z.
    The OMNI dataframe must contain the following columns : COA, CLA.
    Returns dataframes containing different comparison scores for the different conditions
    (all OMNI, MMS windows of interest, subsolar data, and predicted BL).
    """

    arrays = create_empty_comparative_parker_scores_arrays(starts, stops)

    # All omni
    scores = compute_temporal_diff_parker_spiral_orientations(omni, starts, stops, verbose=verbose,
                                                              title='OMNI')
    arrays = fill_comparative_scores_parker(arrays, scores, 'OMNI')

    # MMS windows
    scores = compute_temporal_diff_parker_spiral_orientations(df, starts, stops, verbose=verbose,
                                                              title='MMS windows of interest')
    arrays = fill_comparative_scores_parker(arrays, scores, 'MMS windows')

    # Subsolar
    temp = df[np.sqrt(df.Y.values ** 2 + df.Z.values ** 2) < 5]
    scores = compute_temporal_diff_parker_spiral_orientations(temp, starts, stops, verbose=verbose,
                                                              title='subsolar points of the MMS windows')
    arrays = fill_comparative_scores_parker(arrays, scores, 'subsolar')

    # predicted BL
    temp = temp[temp.pred_BL.values > 0.5]
    scores = compute_temporal_diff_parker_spiral_orientations(temp, starts, stops, verbose=verbose,
                                                              title='detected BL points among the subsolar '
                                                                    'MMS windows')
    arrays = fill_comparative_scores_parker(arrays, scores, 'detected BL')

    return arrays
