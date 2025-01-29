from swapp.engineered_features.gaussian_fits import *
from scipy.signal import find_peaks
import pandas as pd
import warnings


def find_apogee_times(df):
    df['R'] = np.sqrt(df.X.values.astype('float') ** 2 +
                      df.Y.values.astype('float') ** 2 +
                      df.Z.values.astype('float') ** 2)
    df['R'] = df[['R']].interpolate().values
    times = df.index.values
    apogee_times, _ = find_peaks(df.R.values, prominence=(2, None))  # Check if prominence needs to be changed
    return times[apogee_times]


def find_perigee_times(df):
    df['R'] = np.sqrt(df.X.values.astype('float') ** 2 +
                      df.Y.values.astype('float') ** 2 +
                      df.Z.values.astype('float') ** 2)
    df['R'] = df[['R']].interpolate().values
    times = df.index.values
    perigee_times, _ = find_peaks(-df.R.values, prominence=(2, None))  # Check if prominence needs to be changed
    return times[perigee_times]


def get_closest(time, times):
    dt = times - time
    return times[np.argmin(abs(dt))]


def preprocess_perigees_apogees(apogee_times, perigee_times):
    if apogee_times[0] < perigee_times[0]:
        apogee_times = apogee_times[1:]
    if apogee_times[-1] > perigee_times[-1]:
        apogee_times = apogee_times[:-1]
    assert len(apogee_times) == len(perigee_times)-1, "Code error in find_closest_apogees_perigees"
    return apogee_times, perigee_times


def find_closest_apogees_perigees(apogee_times, perigee_times, df):
    # find dates of closest apogee and perigee for every t (could be made more quickly)
    df['apogee_time'] = np.nan
    df['perigee_time'] = np.nan

    apogee_times, perigee_times = preprocess_perigees_apogees(apogee_times, perigee_times)

    for i, apogee in enumerate(apogee_times):
        assert perigee_times[i] < apogee, (f"Perigee {perigee_times[i]} (n°{i}) should be before apogee {apogee} "
                                           f"(n°{i})")
        df.loc[df[perigee_times[i]:apogee].index.values, 'apogee_time'] = apogee
        df.loc[df[perigee_times[i]:apogee].index.values, 'perigee_time'] = perigee_times[i]

        assert perigee_times[
                   i + 1] > apogee, (f"Perigee {perigee_times[i + 1]} (n°{i + 1}) should be after apogee {apogee} "
                                     f"(n°{i})")
        df.loc[df[apogee:perigee_times[i + 1]].index.values, 'apogee_time'] = apogee
        df.loc[df[apogee:perigee_times[i + 1]].index.values, 'perigee_time'] = perigee_times[i + 1]


def get_ref_MSH_feature(time, apogee_times, perigee_times, df, feature, **kwargs):
    apogee_time = kwargs.get('apogee_time', get_closest(time, apogee_times))
    perigee_time = kwargs.get('perigee_time', get_closest(time, perigee_times))

    if apogee_time <= perigee_time:
        start, stop = apogee_time, perigee_time
    else:
        start, stop = perigee_time, apogee_time
    temp = df[start:stop]
    if len(temp) > 0 :  # The apogee to perigee data portion is not empty
        if feature == 'Np':
            columns = ['Np']
        else:
            columns = [feature, 'Np']

        temp = temp[columns].dropna().sort_values(by='Np')
        temporary = temp[feature].values.flatten()
        # temporary = temporary[~np.isnan(temporary)]
        percentage = kwargs.get('percentage', 0.01)
        ref_feature = np.median(temporary[-int(len(temporary) * percentage):])
        # assert len(temp) > 0, f"{percentage * 100}% of the apogee to perigee data portion is empty"
    else:
        ref_feature = np.nan

    return ref_feature


def compute_ref_MSH_feature_v1(half_orbit_df, feature, percentage):
    assert len(half_orbit_df) > 0, "The apogee to perigee data portion is empty"
    if feature == 'Np':
        columns = ['Np']
    else:
        columns = [feature, 'Np']
    temp = half_orbit_df[columns].dropna().sort_values(by='Np')

    ref_feature = np.nan
    if len(temp) > 0:
        temporary = temp[feature].values.flatten()
        if int(len(temporary) * percentage) > 0:
            ref_feature = np.median(temporary[-int(len(temporary) * percentage):])

    return ref_feature


def compute_ref_MSH_feature(half_orbit_df, feature, percentage):
    if len(half_orbit_df) > 0:   # The apogee to perigee data portion is not empty

        if (feature == 'Np') or (feature == 'Vx'):
            columns = ['Np', 'Vx']
        else:
            columns = [feature, 'Np', 'Vx']

        half_orbit_df = half_orbit_df[columns]
        half_orbit_df[f'ref_MSH_{feature}'] = np.nan

        # Every hour median
        hours = pd.date_range(half_orbit_df.index.values[0],
                              half_orbit_df.index.values[-1] + np.timedelta64(1, 'h'),
                              freq='1H')
        for j in range(len(hours) - 1):
            temp = half_orbit_df.loc[hours[j]:hours[j + 1], columns].dropna()
            temp = temp[np.logical_and(temp['Np'].values > 15, temp['Vx'].values > -300)]  # probably MSH data (or BL)

            if len(temp) > 200:
                temporary = temp[feature].values
                ref_feature = np.median(temporary)
                half_orbit_df.loc[hours[j]:hours[j + 1], f'ref_MSH_{feature}'] = ref_feature

        half_orbit_df = half_orbit_df.interpolate(method='nearest')
        half_orbit_df = half_orbit_df.ffill()  # forward fill for the last nans in the end of the dataframe
        half_orbit_df = half_orbit_df.bfill()  # backward fill for the first nans in the beginning of the dataframe

        if len(half_orbit_df[[f'ref_MSH_{feature}']].dropna()) > 0:
            ref_feature = half_orbit_df[f'ref_MSH_{feature}'].values
        else:
            half_orbit_df = half_orbit_df[columns].dropna().sort_values(by='Np')
            ref_feature = np.nan
            if len(half_orbit_df) > 0:
                temporary = half_orbit_df[feature].values.flatten()
                if int(len(temporary) * percentage) > 0:
                    ref_feature = np.median(temporary[-int(len(temporary) * percentage):])

    else:
        ref_feature = np.nan

    return ref_feature


def get_ref_MSH_feature_over_time(apogee_times, perigee_times, df, feature, **kwargs):
    warnings.filterwarnings("ignore")
    percentage = kwargs.get('percentage', 0.05)

    df['ref_MSH_' + feature] = np.nan

    apogee_times, perigee_times = preprocess_perigees_apogees(apogee_times, perigee_times)

    for i, apogee in enumerate(apogee_times):
        assert perigee_times[i] < apogee, (f"Perigee {perigee_times[i]} (n°{i}) should be before apogee {apogee} "
                                           f"(n°{i})")
        half_orbit_df = df[perigee_times[i]:apogee]
        median = compute_ref_MSH_feature(half_orbit_df, feature, percentage)
        df.loc[df[perigee_times[i]:apogee].index.values, 'ref_MSH_' + feature] = median

        assert perigee_times[
                   i + 1] > apogee, (f"Perigee {perigee_times[i + 1]} (n°{i + 1}) should be after apogee {apogee} "
                                     f"(n°{i})")
        half_orbit_df = df[apogee:perigee_times[i + 1]]
        median = compute_ref_MSH_feature(half_orbit_df, feature, percentage)
        df.loc[df[apogee: perigee_times[i + 1]].index.values, 'ref_MSH_' + feature] = median


def compute_apogees_perigees(df, sat):
    """ This dataframe must contain X, Y, Z the coordinates of the satellite relative to the Earth."""
    apogee_times = find_apogee_times(df)
    print(f'{len(apogee_times)} apogees have been found.')
    pd.to_pickle(apogee_times, f'/home/ghisalberti/make_datasets/MSH_reference/'
                               f'{sat}_all_apogee_times.pkl')
    perigee_times = find_perigee_times(df)
    print(f'{len(perigee_times)} perigees have been found.')
    pd.to_pickle(perigee_times, f'/home/ghisalberti/make_datasets/MSH_reference/'
                                f'{sat}_all_perigee_times.pkl')

    find_closest_apogees_perigees(apogee_times, perigee_times, df)
    return apogee_times, perigee_times


def compute_ref_MSH_values(all_data, sat, path, apogee_times, perigee_times):
    """ The dataframe all_data must already contain the following features : """

    features = ['Vy', 'Vz', 'Vx', 'Vn_MP', 'Vtan1_MP', 'Vtan2_MP', 'V', 'Np', 'Tpara',
                'Tperp', 'Tp', 'logNp', 'logTp', 'anisotropy']

    '''def f_pool(feature):
        get_ref_MSH_feature_over_time(apogee_times, perigee_times, all_data, feature, percentage=0.05)
        all_data.to_pickle(path)
        all_data.to_pickle(path[:-3] + '_copy.pkl')
        pd.to_pickle(all_data[['ref_MSH_' + feature]],
                     f"/home/ghisalberti/make_datasets/MSH_reference/"
                     f"{sat}_ref_MSH_{feature}_hourly.pkl")
        print(f'{feature} for reference MSH is done')

    with Pool(len(features)) as p:
        p.map(f_pool, features)
    '''
    for feature in features:
        get_ref_MSH_feature_over_time(apogee_times, perigee_times, all_data, feature,
                                      percentage=0.05)
        all_data.to_pickle(path)
        all_data.to_pickle(path[:-3] + '_copy.pkl')
        pd.to_pickle(all_data[['ref_MSH_' + feature]],
                     f"/home/ghisalberti/make_datasets/MSH_reference/"
                     f"{sat}_ref_MSH_{feature}_hourly.pkl")
        print(f'{feature} for reference MSH is done')


def compute_gap_to_MSH(all_data, path):
    """ The dataframe all_data must already contain the reference MSH values for all the
    following features, computed with compute_ref_MSH_values."""
    features = ['Np', 'Tpara', 'Tperp', 'Tp', 'logNp', 'logTp', 'anisotropy', 'V', 'Vx', 'Vy',
                'Vz', 'Vtan1_MP', 'Vtan2_MP', 'Vn_MP']

    # Computing relative error
    for feature in features:
        all_data['gap_to_MSH_' + feature] = all_data[feature].values - all_data['ref_MSH_' + feature].values

    for feature in features[:-6]:
        all_data['relative_gap_with_MSH_' + feature] = abs(
            all_data['gap_to_MSH_' + feature].values / all_data['ref_MSH_' + feature].values)

    # For features that can be null, I normalize with the positive norm instead
    for speed in features[-6:]:
        all_data[f'relative_gap_with_MSH_{speed}'] = (all_data[speed].values - all_data[
            f'ref_MSH_{speed}'].values) / all_data.ref_MSH_V.values

    all_data.to_pickle(path)


def normalize_flow_by_Va(all_data, path):
    """ The dataframe all_data must already contain Vtan1_MP, Vtan2_MP and Va."""
    for speed in ['Vtan1_MP', 'Vtan2_MP']:
        all_data[f'gap_to_MSH_{speed}_over_Va'] = (all_data[speed].values - all_data[
            f'ref_MSH_{speed}'].values) / all_data.Va.values
    all_data.to_pickle(path)


def compute_all_ref_MSH_data(all_data, sat, path):
    """ This dataframe must contain the following columns :
    X, Y, Z, Vy, Vz, Vx, Vn_MP, Vtan1_MP, Vtan2_MP, V, Np, Tpara, Tperp, Tp, logNp, logTp, anisotropy, Va """

    # Also saves the apogee and perigee times in pickle files,
    # and adds in all_data the columns closest_apogee and closest_perigee
    apogee_times, perigee_times = compute_apogees_perigees(all_data, sat)
    all_data = all_data[all_data.R.values > 9]
    compute_ref_MSH_values(all_data, sat, path, apogee_times, perigee_times)
    compute_gap_to_MSH(all_data, path)
    normalize_flow_by_Va(all_data, path)
