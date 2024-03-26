from ..utils import flag, none, all, any


def is_missing(df):
    df.loc[:, 'missing_data'] = df.isna().any(axis=1).values


def flag_empty(df, win_length):
    flag(df, win_length, {"name": "isEmpty", "fun": all, "features": ["missing_data"]})


def flag_full(df, win_length):
    flag(df, win_length, {"name": "isFull", "fun": none, "features": ["missing_data"]})


def flag_partial(df, win_length):
    flag(df, win_length, {"name": "isPartial", "fun": any, "features": ["missing_data"]})
