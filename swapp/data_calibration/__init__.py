import pandas as pd
import numpy as np
from swapp.windowing.make_windows.utils import time_resolution
from datetime import datetime
from swapp.dataframes import intersect, get_consecutive_interval
import speasy as spz
import matplotlib.pyplot as plt


def compute_distance_between_satellites(df_pos_list, satellites):
    dt = time_resolution(df_pos_list[0])/2
    distances = pd.DataFrame([], columns=['distance', 'sat1', 'sat2'])

    for i, df1 in enumerate(df_pos_list[:-1]):
        for df2, sat in zip(df_pos_list[i + 1:], satellites[i + 1:]):
            start = max(df1.index.values[0], df2.index.values[0])
            stop = min(df1.index.values[-1], df2.index.values[-1])
            d = np.sqrt((df1[start:stop + dt].X.values - df2[start:stop + dt].X.values) ** 2 + (
                        df1[start:stop + dt].Y.values - df2[start:stop + dt].Y.values) ** 2 + (
                                    df1[start:stop + dt].Z.values - df2[start:stop + dt].Z.values) ** 2)
            d = pd.DataFrame(d, index=df1[start:stop].index.values, columns=['distance'])
            d['sat1'] = satellites[i]
            d['sat2'] = sat
            distances = pd.concat([distances, d])
            print(f'{satellites[i]} and {sat} done!')

    distances.to_pickle(
        f'/home/ghisalberti/make_datasets/datasets/distance_between_satellites_{str(datetime.now())[:10]}.pkl')
    return distances


def find_interval_close_2_satellites(threshold_distance, distances_df, sat, sat2, **kwargs):
    distances = distances_df.copy()

    if (sat == 'MMS1') or (sat2 == 'MMS1'):
        if 'MMS1_data' not in kwargs:
            raise Exception("If one of the two satellites is MMS1, MMS1_data dataframe should be given in kwargs, "
                            "in order to discard out of ROIs conjunctions.")
        mms1_data = kwargs['MMS1_data']
        rois = mms1_data.dropna()
        rois, distances = intersect(rois, distances)

    temp = distances[np.logical_and(distances.sat1.values == sat, distances.sat2.values == sat2)]
    if len(temp) == 0:
        temp = distances[np.logical_and(distances.sat1.values == sat2, distances.sat2.values == sat)]
    temp = temp[temp.distance.values < threshold_distance].sort_index()
    if len(temp) > 0:
        starts, stops = get_consecutive_interval(temp.index.values,
                                                 dt=kwargs.get('dt', np.timedelta64(1, 'm')))
    else:
        starts, stops = [], []

    return starts, stops


def count_intervals_close_satellites(threshold_distance, distances_df, sat, **kwargs):
    satellites = ['THA', 'THB', 'THC', 'THD', 'THE', 'MMS1']
    satellites.remove(sat)
    for sat2 in satellites:
        starts, stops = find_interval_close_2_satellites(threshold_distance, distances_df, sat, sat2, **kwargs)

        if len(starts) > 0:
            print(
                f"For {sat} and {sat2}, there are {len(starts)} intervals of length from "
                f"{int(np.min(np.array(stops - starts) / np.timedelta64(1, 'm')))} minutes to "
                f"{int(np.max(np.array(stops - starts) / np.timedelta64(1, 'm')))} minutes "
                f"where the satellites are closer than {threshold_distance} Earth radii from each other.")
        else:
            print(
                f"For {sat} and {sat2}, there are no intervals of length where the satellites are closer than "
                f"{threshold_distance} Earth radii from each other.")


def load_density_themis(satellite, start, stop):
    n = spz.get_data(f'cda/TH{satellite.upper()}_L2_ESA/th{satellite}_peir_density', start, stop)
    if n is not None:
        n = pd.DataFrame(data=n.values, index=n.time, columns=['Np'])[
            start: stop + np.timedelta64(1, 's')].resample('5S').mean()
    else:
        n = pd.DataFrame([], columns=['Np'])
    return n

def load_temperature_themis(satellite, start, stop):
    # Tp, Tpara, Tperp
    t = spz.get_data(f'cda/TH{satellite.upper()}_L2_ESA/th{satellite}_peir_magt3', start, stop)
    if t is not None:
        t = pd.DataFrame(t.values, index=t.time, columns=['Tperp1', 'Tperp2', 'Tpara']).loc[
            start: stop + np.timedelta64(1, 's')].resample('5S').mean()
        t['Tperp'] = (t.Tperp1.values + t.Tperp2.values) / 2
        t['Tp'] = (t.Tpara.values + 2 * t.Tperp.values) / 3 * 11600  # conversion from eV to K
    else:
        t = pd.DataFrame([], columns=['Tp', 'Tperp', 'Tpara'])
    return t


def load_V_themis(satellite, start, stop):
    # Vx, Vy, Vz, V
    v = spz.get_data(f'cda/TH{satellite.upper()}_L2_ESA/th{satellite}_peir_velocity_gsm', start, stop)
    if v is not None:
        v = pd.DataFrame(v.values, index=v.time, columns=['Vx', 'Vy', 'Vz'])[
            start: stop + np.timedelta64(1, 's')].resample('5S').mean()
        v['V'] = np.sqrt(v.Vx.values ** 2 + v.Vy.values ** 2 + v.Vz.values ** 2)
    else:
        v = pd.DataFrame([], columns=['V', 'Vx', 'Vy', 'Vz'])
    return v


def load_B_themis(satellite, start, stop):
    # Bx, By, Bz, B
    b = spz.get_data(f'cda/TH{satellite.upper()}_L2_FGM/th{satellite}_fgl_gsm', start, stop)
    if b is not None:
        b = pd.DataFrame(b.values, index=b.time, columns=['Bx', 'By', 'Bz'])[
            start: stop + np.timedelta64(1, 's')].resample('5S').mean()
        b['B'] = np.sqrt(b.Bx.values ** 2 + b.By.values ** 2 + b.Bz.values ** 2)
    else:
        b = pd.DataFrame([], columns=['B', 'Bx', 'By', 'Bz'])
    return b


def load_themis_data(sat, start, stop):
    start, stop = pd.to_datetime(start), pd.to_datetime(stop)
    satellite = sat[-1].lower()

    n = load_density_themis(satellite, start, stop)
    t = load_temperature_themis(satellite, start, stop)
    v = load_V_themis(satellite, start, stop)
    b = load_B_themis(satellite, start, stop)

    # pos : X,Y,Z
    pos = pd.read_pickle(
        f'/DATA/ghisalberti/Datasets/THEMIS/TH{satellite.upper()}/TH{satellite.upper()}_pos_5S_GSM.pkl')[
          start: stop + np.timedelta64(1, 's')]

    return n, t, v, b, pos


def merge_features_dataframes(n, t, v, b, pos, start, stop):
    try:
        temp = pd.DataFrame([], index=pd.date_range(start, stop, freq='5S'))
        for col in ['Np', 'Tp', 'Tpara', 'Tperp', 'Vx', 'Vy', 'Vz', 'V', 'Bx', 'By', 'Bz', 'B', 'X', 'Y', 'Z']:
            temp[col] = np.nan
        temp.loc[n.index.values, 'Np'] = n.values
        temp.loc[t.index.values, ['Tp', 'Tperp', 'Tpara']] = t[['Tp', 'Tperp', 'Tpara']].values
        temp.loc[v.index.values, ['Vx', 'Vy', 'Vz', 'V']] = v[['Vx', 'Vy', 'Vz', 'V']].values
        temp.loc[b.index.values, ['Bx', 'By', 'Bz', 'B']] = b[['Bx', 'By', 'Bz', 'B']].values
        temp.loc[pos.index.values, ['X', 'Y', 'Z', 'R']] = pos[['X', 'Y', 'Z', 'R']].values

    except:
        print("Error in merging")
        temp = {'Np': n, 'T': t, 'B': b, 'V': v, 'pos': pos}

    return temp


def get_themis_data(sat, start, stop):
    n, t, v, b, pos = load_themis_data(sat, start, stop)
    temp = merge_features_dataframes(n, t, v, b, pos, start, stop)
    return temp


def get_mms1_data(start, stop, **kwargs):
    if 'MMS1_data' not in kwargs:
        raise Exception("If one of the two satellites is MMS1, MMS1_data dataframe should be given in kwargs, "
                        "in order to discard out of ROIs conjunctions.")
    mms1_data = kwargs['MMS1_data']

    temp = mms1_data.loc[start:stop + np.timedelta64(5, 's'),
           ['Np', 'Tp', 'Tpara', 'Tperp', 'Vx', 'Vy', 'Vz', 'V', 'Bx', 'By', 'Bz', 'B', 'X', 'Y', 'Z']]
    temp['R'] = np.sqrt(temp.X.values ** 2 + temp.Y.values ** 2 + temp.Z.values ** 2)

    return temp


def get_data(start, stop, sat, **kwargs):
    if sat == 'MMS1':
        temp = get_mms1_data(start, stop, **kwargs)

    else:
        temp = get_themis_data(sat, start, stop)

    return temp


def hist_without_outliers(df, col, ax, **kwargs):
    values = df[[col]].dropna().values
    if len(values) > 0:
        low_outliers = np.nanquantile(values, 0.0001, method='higher')
        high_outliers = np.nanquantile(values, 0.9999, method='lower')
        values = values[np.logical_and(values > low_outliers, values < high_outliers)]
    _ = ax.hist(values, **kwargs)


def compare_satellites_distributions(df1, df2, sat1, sat2,
                                     features=['X', 'Y', 'Z', 'R', 'Bx', 'By', 'Bz', 'B',
                                               'Vx', 'Vy', 'Vz', 'V', 'Np', 'Tp', 'Tpara', 'Tperp']):
    nrows = int(np.ceil(len(features)/4))
    fig, ax = plt.subplots(nrows=nrows, ncols=4, figsize=(20, 3*nrows))
    for i, col in enumerate(features):
        hist_without_outliers(df1, col, ax[i//4, i % 4], bins=100, label=sat1, alpha=0.5, density=True)
        hist_without_outliers(df2, col, ax[i//4, i % 4], bins=100, label=sat2, alpha=0.5, density=True)

        ax[i//4, i % 4].set_xlabel(col)
        ax[i//4, i % 4].set_ylabel('Count (normalized)')
        ax[i//4, i % 4].legend()
        if col in ['Bx', 'By', 'Bz', 'B', 'Vx', 'Vy', 'Vz', 'V', 'Np', 'Tp', 'Tpara', 'Tperp']:
            ax[i//4, i % 4].set_yscale('log')
    fig.tight_layout()
    fig.show()


def compare_conjunctions(sat, sat2, threshold_distance, distances, verbose=False):
    starts, stops = find_interval_close_2_satellites(threshold_distance, distances, sat, sat2)
    print(f'{len(starts)} intervals were found.')
    datas = {}

    for i, (start, stop) in enumerate(zip(starts, stops)):
        print(f'From {start} to {stop}:')
        data_sat1 = get_data(start, stop, sat)
        data_sat2 = get_data(start, stop, sat2)
        if verbose:
            compare_satellites_distributions(data_sat1, data_sat2, sat, sat2)
            plt.show()
        datas[i] = {'start': start, 'stop': stop, 'sat1': sat, 'sat2': sat2, 'data1': data_sat1, 'data2': data_sat2}
    return datas
