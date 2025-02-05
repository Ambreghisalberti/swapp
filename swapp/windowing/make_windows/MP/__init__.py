import pandas as pd
from ...gradient_boosting import pred_boosting
from .utils import *


def intersect_MP(df, pos, omni, win_duration, method, **kwargs):
    if method == 'encountersMSPandMSH':
        prepare_df_MSP_MSH_overlap(df, win_duration, **kwargs)
    elif method == 'Shue':
        prepare_df_close_to_MP(pos, omni, win_duration, **kwargs)
    elif method == 'both':
        intersect_MP(df, pos, omni, win_duration, 'encountersMSPandMSH', **kwargs)
        intersect_MP(df, pos, omni, win_duration, 'encountersMSPandMSH', **kwargs)
    else:
        raise Exception("To determine which make_windows are close to the magnetopause, you need to specify the criteria :"
                        "it has to be either 'encountersMSPandMSH', 'Shue' or 'both'.")


def prepare_df_MSP_MSH_overlap(df, win_duration, **kwargs):
    """
    These are the features of the gradient boosting. This allows to run this function even after columns were added.
    """
    df = prepare_df_MSP_MSH_overlap_pre_windowing(df)
    df = prepare_df_MSP_MSH_overlap_windowing(df, win_duration, **kwargs)
    return df


def prepare_df_MSP_MSH_overlap_pre_windowing(df, **kwargs):
    data = df[['Bx', 'By', 'Bz', 'Np', 'Vx', 'Vy', 'Vz', 'Tp']]
    pred_boosting(data, **kwargs)
    regions(data)
    '''
      We add the results to the initial dataframe to keep the initial columns.
      '''
    for feature in ['regions_pred', 'isMSP', 'isMSH', 'isSW']:
        df[feature] = data[feature].values
    return df


def prepare_df_MSP_MSH_overlap_windowing(df, win_duration, **kwargs):
    df = flag_msp_and_msh(df, win_duration, **kwargs)
    return df


def prepare_df_close_to_MP(pos, omni, win_duration, **kwargs):
    pos = prepare_df_close_to_MP_pre_windowing(pos, omni, **kwargs)
    pos = prepare_df_close_to_MP_windowing(pos, omni, win_duration, **kwargs)
    return pos


def prepare_df_close_to_MP_pre_windowing(pos, omni, **kwargs):
    dl_inf = kwargs.get('dl_inf', 3)
    dl_sup = kwargs.get('dl_sup', 3)

    pos_omni = pd.concat([pos, omni], axis=1)
    shue_mp(pos_omni)
    beyond_sup_limit(pos_omni, dl_sup)
    below_inf_limit(pos_omni, dl_inf)
    is_around_mp(pos_omni)

    for feature in ['r_mp', 'beyond_sup_limit', 'below_inf_limit', 'is_around_mp']:
        pos[feature] = pos_omni[feature].values
    return pos


def prepare_df_close_to_MP_windowing(pos, omni, win_duration, **kwargs):
    pos_omni = pd.concat([pos, omni], axis=1)
    flag_around_mp(pos_omni, win_duration, **kwargs)

    for feature in ['isCloseToMP', 'isCloseToMP_select']:
        pos[feature] = pos_omni[feature].values
    return pos