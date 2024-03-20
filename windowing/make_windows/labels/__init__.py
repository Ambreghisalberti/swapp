from .utils import *

# Put into a file
labelled_days = [f"2020-01-0{i}" for i in range(1, 10)]+[f"2020-01-{i}" for i in range(10, 32)]
labelled_days += [f"2020-02-0{i}" for i in range(1, 10)]+[f"2020-02-{i}" for i in range(10, 30)]
labelled_days += [f"2021-03-0{i}" for i in range(1, 10)]+[f"2021-03-{i}" for i in range(10, 32)]
labelled_days += ['2015-11-22', '2015-12-17', '2016-01-02', '2016-01-22', '2016-02-01', '2016-02-02', '2016-02-11',
                  '2021-05-18', '2020-10-31']
labelled_days.sort()
labelled_days = np.array(labelled_days)
print(len(labelled_days), 'days are labelled.')


def prepare_df_labels(df, win_length, paths):
    is_labelled(df, labelled_days)
    flag_labelled(df, win_length)
    labels(df, paths)
    flag_contains_labelled_bl(df, win_length)
    flag_count_BL_points(df, win_length)
