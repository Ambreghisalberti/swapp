import pandas as pd
import numpy as np


def intersect(df1: pd.DataFrame, df2: pd.DataFrame) -> (pd.DataFrame,pd.DataFrame):
    df1 = df1[df1.index.isin(df2.index)]
    df2 = df2[df2.index.isin(df1.index)]
    return df1, df2


def fill_small_gaps(df, n):
    """
    interpolate values in data holes of length < N
    precondition: holes are identified as consecutive NaNs
    """
    df_interpolated = df.interpolate()

    for c in df.columns:
        mask = df[c].isna()
        x = mask.groupby((mask != mask.shift()).cumsum()).transform(lambda x: len(x) > n) & mask
        df_interpolated[c] = df_interpolated.loc[~x, c]

    return df_interpolated


def get_consecutive_interval(times, dt=np.timedelta64(5, 's')):
    split = str(dt).split(' ')
    if split[1] == 'minutes':
        dt = str(dt).split(' ')[0] + 'min'
    else:
        dt = str(dt).split(' ')[0] + (str(dt).split(' ')[1][0])
    times = np.sort(times)
    all_times = pd.date_range(times[0], times[-1], freq=dt)
    temp = pd.DataFrame(np.zeros(len(all_times)), index=all_times, columns=['is_present'])
    temp.loc[times, 'is_present'] = 1
    temp['change'] = [0] + list((temp.is_present.values[1:] != temp.is_present.values[:-1]).astype('int'))
    temp['group'] = temp.change.values.cumsum()
    temp = temp.drop(temp[temp.is_present.values == 0].index.values)

    starts = temp.groupby('group').apply(lambda x: x.index.values[0]).values
    stops = temp.groupby('group').apply(lambda x: x.index.values[-1]).values
    assert (starts > stops).sum() == 0
    return starts, stops
