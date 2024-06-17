from .labels import prepare_df_labels
from .nodata import prepare_df_missing_data
from .MP import intersect_MP
from .utils import *
from .dayside import prepare_dayside


def prepare_df_original(data, pos, omni, win_duration, paths, labelled_days, **kwargs):
    win_length = durationToNbrPts(win_duration, time_resolution(data))

    # Making a copy of the datasets is useful to keep the original dataframes,
    # but it uses twice as much memory

    #data = all_data.copy()
    #pos = position.copy()
    #omni = omni_data.copy()

    #cut_nightside(pos, omni, data)
    pos, omni, data = resize_preprocess(pos, omni, data, win_length)

    '''
    Adds to the all_data dataframe columns of interest to characterize points (predicted regions) 
    or make_windows (full, with missing data, overlapping MSP and MSH, etc).
    Precondition : all_data has the following features : ['Bx', 'By', 'Bz', 'Np', 'Vx', 'Vy', 'Vz', 'Tp']
    '''
    prepare_dayside(data, pos, omni, win_length, **kwargs)
    intersect_MP(data, pos, omni, win_length, 'encountersMSPandMSH', **kwargs)
    print('Overlap MSP/MSH done!')
    prepare_df_missing_data(data, win_length, **kwargs)
    print('Missing data done!')
    intersect_MP(data, pos, omni, win_length, 'Shue', **kwargs)
    print('Close to MP done!')
    prepare_df_labels(data, win_length, paths, labelled_days, **kwargs)
    print('Labels done!')

    for df in (data, pos, omni):
        df.drop(df.index.values[0], inplace=True)

    return data, pos, omni


def prepare_df(data, pos, omni, win_duration, paths, labelled_days, **kwargs):
    win_length = durationToNbrPts(win_duration, time_resolution(data))

    #pos, omni, data = resize_preprocess(pos, omni, data, win_length)

    '''
    Adds to the all_data dataframe columns of interest to characterize points (predicted regions) 
    or make_windows (full, with missing data, overlapping MSP and MSH, etc).
    Precondition : all_data has the following features : ['Bx', 'By', 'Bz', 'Np', 'Vx', 'Vy', 'Vz', 'Tp']
    '''
    prepare_dayside(data, pos, omni, win_length, **kwargs)
    intersect_MP(data, pos, omni, win_length, 'encountersMSPandMSH', **kwargs)
    print('Overlap MSP/MSH done!')
    prepare_df_missing_data(data, win_length, **kwargs)
    print('Missing data done!')
    intersect_MP(data, pos, omni, win_length, 'Shue', **kwargs)
    print('Close to MP done!')
    prepare_df_labels(data, win_length, paths, labelled_days, **kwargs)
    print('Labels done!')

    return data, pos, omni