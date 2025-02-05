from ..utils import flag, flag_select, none, all, any


def is_missing(df):
    df.loc[:, 'missing_data'] = df.isna().any(axis=1).values


def flag_empty(df, win_duration, **kwargs):
    flagger = {"name": "isEmpty", "fun": all, "features": ["missing_data"]}
    flag(df, win_duration, flagger, **kwargs)
    flag_select(df, win_duration, flagger)


def flag_full(df, win_duration, **kwargs):
    flagger = {"name": "isFull", "fun": none, "features": ["missing_data"]}
    flag(df, win_duration, flagger, **kwargs)
    flag_select(df, win_duration, flagger)


def flag_partial(df, win_duration, **kwargs):
    flagger = {"name": "isPartial", "fun": any, "features": ["missing_data"]}
    flag(df, win_duration, flagger, **kwargs)
    flag_select(df, win_duration, flagger)
