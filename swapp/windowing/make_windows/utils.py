import numpy as np
from ...catalogues import create_catalogue


def nbr_windows(df, win_length):
    return len(df) // win_length


def durationToNbrPts(time, resolution):
    """
    returns the number of points in a given time interval
    """
    return int(time / resolution)


def add_dummy_first_line(df):
    df.loc[df.index.values[0] - time_resolution(df)] = np.nan
    df.sort_index(inplace=True)


def remove_last_incomplete_window(df, win_length):
    additional_points = (len(df) - 1) % win_length
    # Number of points that don't fit in a window (leaving the first dummy index out)
    if additional_points > 0:
        df.drop(labels=df.iloc[-additional_points:].index.values, inplace=True)


def resize_preprocess(pos, omni, all_data, win_length):
    for df in (pos, omni, all_data):
        add_dummy_first_line(df)
        remove_last_incomplete_window(df, win_length)


def all(window):
    return window.sum() == len(window)


def none(window):
    return window.sum() == 0


def any(window):
    size = window.sum()
    return 0 < size < len(window)


def flag(df, win_length, flagger, type=bool):
    # step=winLength is the modulo applied to the indexes before taking the
    # window on which apply is called.
    # therefore indexes of make_windows to be evaluated are : 0, winLength, 2winLength etc.
    # window associated with index 0 is NOT COMPLETE
    # window with index winLength is the first to be complete (== have all its points defined in the series)
    tmp = df[flagger["features"]]
    tmp = tmp.rolling(win_length, step=win_length, min_periods=0).apply(flagger["fun"]).astype(type).values
    if len(flagger['features']) > 1:
        tmp = flagger['merger'](tmp)
    df[flagger["name"]] = type(False)

    for i in range(1, win_length+1):
        df.iloc[i::win_length, -1] = tmp[1:]

    assert df.iloc[:,-1].sum() % win_length == 0, "The flag values sum is not a multiple of the size of the windows."
    # If flagger["feature"] contains several features, I will need to add a merger here, such as tmp.any(axis=1)
    # so tmp.flagger["merger"] with flagger['merger'] = np.any(axis=1)


def get_window(df, t_end, win_duration):
    resolution = time_resolution(df)
    return df.loc[t_end - win_duration + resolution:t_end]


def get_window_features(df, t_end, win_duration, features):
    resolution = time_resolution(df)
    return df.loc[t_end - win_duration + resolution:t_end, features]


def time_resolution(df):
    """
    return the time resolution
    precondition: assumes the df has uniform resolution
    """
    return df.index[1] - df.index[0]


def select_windows(df, condition):
    """ Needs to have one for all the points of the window, not only for the last one!
    Refactor flag function, and the counts of swapp."""
    if isinstance(condition, str):
        if condition == ('all'):
            return df
        else:
            return df[df[condition].values == True]
    elif isinstance(condition, list):
        if condition != []:
            subdf = select_windows(df, condition[0])
            return select_windows(subdf, condition[1:])
        else:
            return df
    else:
        raise Exception("Condition should be a string or a list of strings.")


def windows_to_catalogue(df, win_duration, name):
    resolution = time_resolution(df)
    win_length = durationToNbrPts(win_duration, resolution)
    stops = df.index.values[win_length - 1::win_length]
    starts = stops - win_duration + resolution
    return create_catalogue(starts, stops, name)


def cut_nightside(pos, omni, all_data):
    for df in (pos, omni, all_data):
        df[pos.X.values < 0] = np.nan
