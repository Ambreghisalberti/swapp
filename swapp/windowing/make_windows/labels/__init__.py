from .utils import *


def prepare_df_labels(df, win_length, label_catalogues_dict, intervals, **kwargs):
    df = prepare_df_labels_pre_windowing(df, label_catalogues_dict, intervals)
    df = prepare_df_labels_windowing(df, win_length, **kwargs)
    return df

def prepare_df_labels_pre_windowing(df, label_catalogues_dict, intervals):
    intervals.sort_values(by='start')
    #intervals = np.array(intervals)
    print(len(intervals), 'intervals are labelled.')

    is_labelled(df, intervals)

    for category in label_catalogues_dict.keys():
        paths = label_catalogues_dict[category]
        labels(df, paths, category)
        print(f"Labels of {category} done.")
    return df

def prepare_df_labels_windowing(df, win_length, **kwargs):
    flag_labelled(df, win_length, **kwargs)
    flag_contains_labelled_bl(df, win_length, **kwargs)
    flag_count_BL_points(df, win_length, **kwargs)
    return df