import pandas as pd
from ...gradient_boosting import pred_boosting
from .utils import *


def intersect_MP(df, pos, omni, win_length, method, **kwargs):
    if method == 'encountersMSPandMSH':
        prepare_df_MSP_MSH_overlap(df, win_length, **kwargs)
    elif method == 'Shue':
        prepare_df_close_to_MP(pos, omni, win_length, **kwargs)
    elif method == 'both':
        intersect_MP(df, pos, omni, win_length, 'encountersMSPandMSH', **kwargs)
        intersect_MP(df, pos, omni, win_length, 'encountersMSPandMSH', **kwargs)
    else:
        raise Exception("To determine which make_windows are close to the magnetopause, you need to specify the criteria :"
                        "it has to be either 'encountersMSPandMSH', 'Shue' or 'both'.")


def prepare_df_MSP_MSH_overlap(df, win_length, **kwargs):
    """
    These are the features of the gradient boosting. This allows to run this function even after columns were added.
    """
    data = df[['Bx', 'By', 'Bz', 'Np', 'Vx', 'Vy', 'Vz', 'Tp']]
    pred_boosting(data)
    regions(data)
    flag_msp_and_msh(data, win_length, **kwargs)
    for feature in ['regions_pred', 'isMSP', 'isMSH', 'isSW', 'encountersMSPandMSH']:
        df[feature] = data[feature].values
    '''
    We add the results to the initial dataframe to keep the initial columns.
    '''


def prepare_df_close_to_MP(pos, omni, win_length, **kwargs):
    dl_inf = kwargs.get('dl_inf', 3)
    dl_sup = kwargs.get('dl_sup', 3)

    pos_omni = pd.concat([pos, omni], axis=1)
    shue_mp(pos_omni)
    beyond_sup_limit(pos_omni, dl_sup)
    below_inf_limit(pos_omni, dl_inf)
    is_around_mp(pos_omni)
    flag_around_mp(pos_omni, win_length, **kwargs)

    for feature in ['r_mp', 'isCloseToMP']:
        pos[feature] = pos_omni[feature].values
