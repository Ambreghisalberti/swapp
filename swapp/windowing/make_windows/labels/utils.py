from ..utils import flag, flag_select, none
import numpy as np
import pandas as pd
from datetime import timedelta
from ....catalogues import read_catalogue_events
from ..utils import time_resolution

'''
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

    labelled_indices = df[df.index.isin(indices)].index.values
    if len(labelled_indices) < len(indices):
        print('WARNING! Some labelled dates were not found in the data, pos, omni datasets!')
    df.loc[labelled_indices, 'labelled_data'] = np.ones(len(indices))
'''

'''
#Old version of is_labelled for 24h

def is_labelled(df, days):
    df['labelled_data'] = 0
    for day in days:
        day = pd.Timestamp(str(day))
        next_day = day + np.timedelta64(1, 'D')
        df.loc[day:next_day, 'labelled_data'] = 1
'''

def is_labelled(df, intervals):
    df['labelled_data'] = 0
    for start,stop in intervals.values:
        start = pd.Timestamp(str(start))
        stop = pd.Timestamp(str(stop))
        df.loc[start:stop, 'labelled_data'] = 1


def flag_labelled(df, win_length, **kwargs):
    flagger = {"fun": all, "name": "isLabelled", "features": ["labelled_data"]}
    flag(df, win_length, flagger, **kwargs)
    flag_select(df, win_length, flagger)


def catalogue_to_label(catalogue, df, name):
    for ev in catalogue:
        df.loc[ev['start']:ev['stop'], 'label_'+name] = 1


def labels(df, paths, name):
    df['label_'+name] = 0
    for path in paths:
        catalogue = read_catalogue_events(path)
        catalogue_to_label(catalogue, df, name)
        print(f'Path {path} done.')
    df['label_'+name] = df['label_'+name].values * df.labelled_data.values


def flag_contains_labelled_category(df, win_length, category, **kwargs):
    flagger = {"name": "containsLabelled"+category, "fun": lambda x: not (none(x)), "features": ["label_"+category]}
    flag(df, win_length, flagger, **kwargs)
    flag_select(df, win_length, flagger)


def flag_contains_labelled_bl(df, win_length, **kwargs):
    flag_contains_labelled_category(df, win_length, 'BL', **kwargs)


def flag_count_category_points(df, win_length, category, **kwargs):
    flag(df, win_length, {"name": "nbrLabelled"+category, "fun": lambda x: x.sum(),
                          "features": ["label_"+category]}, type=int, **kwargs)


def flag_count_BL_points(df, win_length, **kwargs):
    flag_count_category_points(df, win_length, 'BL', **kwargs)

