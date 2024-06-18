from .utils import *

def prepare_df_labels(df, win_length, label_catalogues_dict, intervals, **kwargs):
    intervals.sort_values(by='start')
    #intervals = np.array(intervals)
    print(len(intervals), 'days are labelled.')

    is_labelled(df, intervals)
    flag_labelled(df, win_length, **kwargs)

    for category in label_catalogues_dict.keys():
        paths = label_catalogues_dict[category]
        labels(df, paths, category)

    flag_contains_labelled_bl(df, win_length, **kwargs)
    flag_count_BL_points(df, win_length, **kwargs)
