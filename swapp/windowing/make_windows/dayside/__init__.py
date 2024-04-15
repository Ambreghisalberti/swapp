from ..utils import flag

def prepare_dayside(data, pos, omni, win_length):
    is_dayside(pos)
    flag_dayside(pos, win_length)


def is_dayside(df):
    df.loc[:, 'is_dayside'] = df.X.values >= 0


def flag_dayside(df, win_length):
    flag(df, win_length, {"fun": all, "name": "isDayside", "features": ["is_dayside"]})
