from .utils import *

def prepare_df_labels(df, win_length, paths, intervals, **kwargs):
    intervals.sort_values(by='start')
    #intervals = np.array(intervals)
    print(len(intervals), 'days are labelled.')

    is_labelled(df, intervals)
    flag_labelled(df, win_length, **kwargs)
    labels(df, paths)
    flag_contains_labelled_bl(df, win_length, **kwargs)
    flag_count_BL_points(df, win_length, **kwargs)
