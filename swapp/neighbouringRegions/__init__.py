# Code to look at properties of reference MSP and MSH for a given sub BL selection
import pandas as pd
import numpy as np
from swapp.dataframes import get_consecutive_interval


def get_interval_centers(df, BL, sat):
    # df is the selection of BL we're interested in
    # BL is the whole BL found, used for calculation of ref MSP and MSH
    temp_df = df[df.sat.values == sat]
    temp_BL = BL[BL.sat.values == sat]
    starts_intervals, stops_intervals = get_consecutive_interval(temp_BL.index.values)
    centers_BL_intervals = []
    for t in temp_df.index.values:
        location = np.arange(len(starts_intervals))[np.logical_and(t >= starts_intervals, t <= stops_intervals)]
        if len(location) != 1:
            print(f"There should be one interval containing this BL point, but there are {len(location)} instead.")
        else:
            centers_BL_intervals += [starts_intervals[location] + (stops_intervals[location]-starts_intervals[location])/2]
    return np.array(centers_BL_intervals)


def get_ref_region_dates(centers_BL_intervals, closest_region_dates):
    # Region is 'MSP' or 'MSH'
    ref_region_dates = []
    for t in centers_BL_intervals:
        ref_region_dates += list(closest_region_dates[t])
    ref_region_dates = np.unique(np.array(ref_region_dates))
    return ref_region_dates


# Thif function gives an error
def get_ref_region_values(all_df, region, datas, closest_MSP_dates_list, closest_MSH_dates_list):
    ref_region_values = pd.DataFrame()
    for sat in ['MMS1','THA','THB','THC']:
        df = all_df[all_df.sat.values == sat]
        centers_BL_intervals = df['date_center_BL_interval'].values
        if region=='MSP':
           closest_region_dates = closest_MSP_dates_list[sat]
        if region=='MSH':
           closest_region_dates = closest_MSH_dates_list[sat]
        ref_region_dates_df = get_ref_region_dates(centers_BL_intervals, closest_region_dates)
        data = datas[sat]
        if (~pd.Index(ref_region_dates_df).isin(data.index.values)).sum() > 0:
            print(f'{(~pd.Index(ref_region_dates_df).isin(data.index.values)).sum()} reference data times are not in the given dataframe for {sat}.')
            ref_region_dates_df = ref_region_dates_df[pd.Index(ref_region_dates_df).isin(data.index.values)]
        ref_region_values = pd.concat([ref_region_values,data.loc[ref_region_dates_df]])
    return ref_region_values
