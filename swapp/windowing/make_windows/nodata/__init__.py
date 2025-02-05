from .utils import *


def prepare_df_missing_data(df, win_duration, **kwargs):
    df = prepare_df_missing_data_pre_windowing(df)
    df = prepare_df_missing_data_windowing(df, win_duration, **kwargs)
    return df


def prepare_df_missing_data_pre_windowing(df):
    is_missing(df)
    return df


def prepare_df_missing_data_windowing(df, win_duration, **kwargs):
    flag_empty(df, win_duration, **kwargs)
    flag_partial(df, win_duration, **kwargs)
    flag_full(df, win_duration, **kwargs)
    return df