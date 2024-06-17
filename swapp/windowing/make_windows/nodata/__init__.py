from .utils import *


def prepare_df_missing_data(df, win_length, **kwargs):
    is_missing(df)
    flag_empty(df, win_length, **kwargs)
    flag_partial(df, win_length, **kwargs)
    flag_full(df, win_length, **kwargs)

