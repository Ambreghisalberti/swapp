from .utils import *


def prepare_df_missing_data(df, win_length, **kwargs):
    df = prepare_df_missing_data_pre_windowing(df)
    df = prepare_df_missing_data_windowing(df, win_length, **kwargs)
    return df

def prepare_df_missing_data_pre_windowing(df):
    is_missing(df)
    return df

def prepare_df_missing_data_windowing(df, win_length, **kwargs):
    flag_empty(df, win_length, **kwargs)
    flag_partial(df, win_length, **kwargs)
    flag_full(df, win_length, **kwargs)
    return df