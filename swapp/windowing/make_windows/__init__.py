from .labels import prepare_df_labels, prepare_df_labels_pre_windowing, prepare_df_labels_windowing
from .nodata import prepare_df_missing_data, prepare_df_missing_data_pre_windowing, prepare_df_missing_data_windowing
from .MP import (intersect_MP, prepare_df_MSP_MSH_overlap_pre_windowing, prepare_df_MSP_MSH_overlap_windowing,
                 prepare_df_close_to_MP_pre_windowing, prepare_df_close_to_MP_windowing)
from .utils import *
from .dayside import prepare_dayside, prepare_dayside_pre_windowing, prepare_dayside_windowing
import time


def prepare_df_original(data, pos, omni, win_duration, paths, labelled_days, **kwargs):
    win_length = durationToNbrPts(win_duration, time_resolution(data))

    # Making a copy of the datasets is useful to keep the original dataframes,
    # but it uses twice as much memory

    #data = all_data.copy()
    #pos = position.copy()
    #omni = omni_data.copy()

    #cut_nightside(pos, omni, data)
    pos, omni, data = df_with_shared_index(pos, omni, data)
    pos, omni, data = resize_preprocess(pos, omni, data, win_length)

    '''
    Adds to the all_data dataframe columns of interest to characterize points (predicted regions) 
    or make_windows (full, with missing data, overlapping MSP and MSH, etc).
    Precondition : all_data has the following features : ['Bx', 'By', 'Bz', 'Np', 'Vx', 'Vy', 'Vz', 'Tp']
    '''
    data = prepare_dayside(data, pos, omni, win_length, **kwargs)
    intersect_MP(data, pos, omni, win_length, 'encountersMSPandMSH', **kwargs)
    print('Overlap MSP/MSH done!')
    prepare_df_missing_data(data, win_length, **kwargs)
    print('Missing data done!')
    intersect_MP(data, pos, omni, win_length, 'Shue', **kwargs)
    print('Close to MP done!')
    data = prepare_df_labels(data, win_length, paths, labelled_days, **kwargs)
    print('Labels done!')

    for df in (data, pos, omni):
        df.drop(df.index.values[0], inplace=True)

    return data, pos, omni


def prepare_df(data, pos, omni, win_duration, label_catalogues_dict, intervals, **kwargs):
    data, pos, omni = prepare_df_pre_windowing(data, pos, omni, label_catalogues_dict, intervals, **kwargs)
    data, pos, omni = prepare_df_windowing(data, pos, omni, win_duration, **kwargs)
    return data, pos, omni


def prepare_df_pre_windowing_without_labels(data, pos, omni, **kwargs):
    pos, omni, data = df_with_shared_index(pos, omni, data)

    '''
    Adds to the all_data dataframe columns of interest to characterize points (predicted regions) 
    or make_windows (full, with missing data, overlapping MSP and MSH, etc).
    Precondition : all_data has the following features : ['Bx', 'By', 'Bz', 'Np', 'Vx', 'Vy', 'Vz', 'Tp']
    '''
    t1 = time.time()
    prepare_dayside_pre_windowing(pos)
    t2 = time.time()
    print(f'Dayside prepared in {t2-t1} seconds.')

    t1 = time.time()
    data = prepare_df_MSP_MSH_overlap_pre_windowing(data)
    t2 = time.time()
    print(f'Regions prepared in {t2 - t1} seconds.')

    t1 = time.time()
    pos = prepare_df_close_to_MP_pre_windowing(pos, omni, **kwargs)
    t2 = time.time()
    print(f'Close to Shue prepared in {t2 - t1} seconds.')

    t1 = time.time()
    data = prepare_df_missing_data_pre_windowing(data)
    t2 = time.time()
    print(f'Missing data prepared in {t2 - t1} seconds.')

    return data, pos, omni


def prepare_df_pre_windowing(data, pos, omni, label_catalogues_dict, intervals, **kwargs):
    data, pos, omni = prepare_df_pre_windowing_without_labels(data, pos, omni, **kwargs)

    t1 = time.time()
    data = prepare_df_labels_pre_windowing(data, label_catalogues_dict, intervals)
    t2 = time.time()
    print(f'Labels prepared in {t2 - t1} seconds.')

    return data, pos, omni


def prepare_df_windowing(data, pos, omni, win_duration, **kwargs):
    pos, omni, data = df_with_shared_index(pos, omni, data)
    win_length = durationToNbrPts(win_duration, time_resolution(data))

    '''
    Adds to the all_data dataframe columns of interest to characterize points (predicted regions) 
    or make_windows (full, with missing data, overlapping MSP and MSH, etc).
    Precondition : all_data has the following features : ['Bx', 'By', 'Bz', 'Np', 'Vx', 'Vy', 'Vz', 'Tp']
    '''
    t1 = time.time()
    data = prepare_dayside_windowing(data, pos, win_length, **kwargs)
    t2 = time.time()
    print(f'Dayside windows prepared in {t2 - t1} seconds.')

    t1 = time.time()
    data = prepare_df_MSP_MSH_overlap_windowing(data, win_length, **kwargs)
    t2 = time.time()
    print(f'MSP/MSH windows prepared in {t2 - t1} seconds.')

    t1 = time.time()
    pos = prepare_df_close_to_MP_windowing(pos, omni, win_length, **kwargs)
    t2 = time.time()
    print(f'Windows close to Shue prepared in {t2 - t1} seconds.')

    t1 = time.time()
    data = prepare_df_missing_data_windowing(data, win_length, **kwargs)
    t2 = time.time()
    print(f'Missing data windows prepared in {t2 - t1} seconds.')

    t1 = time.time()
    data = prepare_df_labels_windowing(data, win_length, **kwargs)
    t2 = time.time()
    print(f'Labelled windows prepared in {t2 - t1} seconds.')

    return data, pos, omni


def prepare_df_new_labels(data, pos, omni, win_duration, label_catalogues_dict, intervals, **kwargs):
    ''' Precondition : takes as input data, pos and omni dataframes which already went through the
    prepare_df_pre_windowing_without_labels function, but this step should yield the same results everytime.'''
    t1 = time.time()
    data = prepare_df_labels_pre_windowing(data, label_catalogues_dict, intervals)
    t2 = time.time()
    print(f'Labels prepared in {t2 - t1} seconds.')
    data, pos, omni = prepare_df_windowing(data, pos, omni, win_duration, **kwargs)
    return data, pos, omni