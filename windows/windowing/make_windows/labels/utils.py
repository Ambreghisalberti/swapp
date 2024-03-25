from ..utils import flag, none
import numpy as np
import pandas as pd
from datetime import timedelta
from ....catalogues import read_catalogue_events
from ..utils import time_resolution


def days_to_dates(days, df):
    indices = []
    for day in days:
        next_day = pd.Timestamp(str(day)) + np.timedelta64(1, 'D')
        new_indices = pd.date_range(start=str(day), end=next_day, freq=time_resolution(df), inclusive='left')
        indices += list(new_indices)
    indices = np.array(indices)

    if ((indices[1:] - indices[:-1]) <= timedelta(0)).sum() != 0:
        raise Exception('CAREFUL! The indices are duplicated or not ordered.')

    return indices


def is_labelled(df, labelled_days):
    indices = days_to_dates(labelled_days, df)
    df['labelled_data'] = 0

    labelled_indices = df.index.isin(indices).index.values
    if len(labelled_indices) < len(indices):
        print('WARNING! Some labelled dates were not found in the data, pos, omni datasets!')
    df.loc[labelled_indices, 'labelled_data'] = np.ones(len(indices))


def flag_labelled(df, win_length):
    flag(df, win_length, {"fun": all, "name": "isLabelled", "features": ["labelled_data"]})


def catalogue_to_label(catalogue, df):
    for ev in catalogue:
        df.loc[ev['start']:ev['stop'], 'label'] = 1


def labels(df, paths):
    df['label'] = 0
    for path in paths:
        catalogue = read_catalogue_events(path)
        catalogue_to_label(catalogue, df)
    df['label'] = df.label.values * df.labelled_data.values


def flag_contains_labelled_bl(df, win_length):
    flag(df, win_length, {"name": "containsLabelledBL", "fun": lambda x: not (none(x)), "features": ["label"]})


def flag_count_BL_points(df, win_length):
    flag(df, win_length, {"name": "nbrLabelledBL", "fun": lambda x: x.sum(), "features": ["label"]}, type=int)
