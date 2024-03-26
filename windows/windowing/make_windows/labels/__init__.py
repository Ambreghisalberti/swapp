from .utils import *

def prepare_df_labels(df, win_length, paths, labelled_days):
    labelled_days.sort()
    labelled_days = np.array(labelled_days)
    print(len(labelled_days), 'days are labelled.')

    is_labelled(df, labelled_days)
    flag_labelled(df, win_length)
    labels(df, paths)
    flag_contains_labelled_bl(df, win_length)
    flag_count_BL_points(df, win_length)
