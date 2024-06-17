from ..utils import flag

def prepare_dayside(data, pos, omni, win_length, **kwargs):
    is_dayside(pos)
    flag_dayside(pos, win_length, **kwargs)
    for col in ['is_dayside','isDayside']:
        data[col] = pos[col].values

def is_dayside(df):
    df.loc[:, 'is_dayside'] = df.X.values >= 0


def flag_dayside(df, win_length,  **kwargs):
    flag(df, win_length, {"fun": all, "name": "isDayside", "features": ["is_dayside"]},  **kwargs)
