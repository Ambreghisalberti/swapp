from ..utils import flag, flag_select


def prepare_dayside(data, pos, omni, win_length, **kwargs):
    prepare_dayside_pre_windowing(pos)
    data = prepare_dayside_windowing(data, pos, win_length, **kwargs)
    return data

def prepare_dayside_pre_windowing(df):
    df.loc[:, 'is_dayside'] = df.X.values >= 0


def prepare_dayside_windowing(data, pos, win_length, **kwargs):
    flag_dayside(pos, win_length, **kwargs)
    for col in ['is_dayside', 'isDayside', 'isDayside_select']:
        data[col] = pos[col].values
    return data


def flag_dayside(df, win_length,  **kwargs):
    flag(df, win_length, {"fun": all, "name": "isDayside", "features": ["is_dayside"]},  **kwargs)
    flag_select(df, win_length, {"fun": all, "name": "isDayside", "features": ["is_dayside"]})