import pandas as pd
import numpy as np


def intersect(df1: pd.DataFrame, df2: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    df1 = df1[df1.index.isin(df2.index)]
    df2 = df2[df2.index.isin(df1.index)]
    return df1, df2


def spatial_slice(df, conditions):
    """
    Conditions need to be a tuple containing the name of the feature and the min and the max values.
    """
    if isinstance(conditions, tuple):
        conditions = [conditions]

    feature, vmin, vmax = conditions[0]
    isin = np.logical_and(vmin <= df[feature].values, df[feature].values <= vmax)
    subdf = df[isin]
    if len(conditions) == 1:
        return subdf
    else:
        subdf2 = spatial_slice(df, conditions[1:])
        subdf, _ = intersect(subdf, subdf2)
        return subdf


def get_dates_for_position(df, y, dy, z, dz):
    subdf = spatial_slice(df, [('Y', y, y + dy), ('Z', z, z + dz)])
    days = [str(date)[:10] for date in subdf.index.values]
    days = np.unique(np.array(days))
    return days


def pick_windows_grid(df, dy, dz, **kwargs):
    """
    kwargs can be a number of windows in each cell of the grid, or a percentage of total windows that cross the cell.
    """
    Y = np.arange(-25, 25, dy)
    Z = np.arange(-25, 25, dz)

    to_label = []
    for y in Y:
        for z in Z:
            days = get_dates_for_position(df, y, dy, z, dz)
            if len(days) > 0:
                if 'windows_per_cell' in kwargs:
                    size = kwargs['windows_per_cell']
                else:
                    proportion = kwargs.get('proportion', 0.1)
                    size = int(np.ceil(proportion * len(days)))

                if size >= len(days):
                    to_label += list(days)
                else:
                    to_label += list(np.random.choice(np.array(days), size=size))
    return np.unique(np.array(to_label))

# Could add a feature for if some windows are already labelled, not label even more in those regions!