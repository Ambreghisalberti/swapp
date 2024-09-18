from swapp.engineered_features.gaussian_fits import *
from scipy.signal import find_peaks


def find_apogee_times(df):
    df['R'] = np.sqrt(df.X.values ** 2 + df.Y.values ** 2 + df.Z.values ** 2)
    times = df.index.values
    apogee_times, _ = find_peaks(df.R.values, prominence=(2, None))  # Check if prominence needs to be changed
    return times[apogee_times]


def find_perigee_times(df):
    df['R'] = np.sqrt(df.X.values ** 2 + df.Y.values ** 2 + df.Z.values ** 2)
    times = df.index.values
    perigee_times, _ = find_peaks(-df.R.values, prominence=(2, None))  # Check if prominence needs to be changed
    return times[perigee_times]


def get_closest(time, times):
    dt = times - time
    return times[np.argmin(abs(dt))]


def find_closest_apogees_perigees(apogee_times, perigee_times, df):
    # find dates of closest apogee and perigee for every t (could be made more quickly)
    df['apogee_time'] = np.nan
    df['perigee_time'] = np.nan

    if apogee_times[0] < perigee_times[0]:
        apogee_times = apogee_times[1:]

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
    assert len(temp) > 0, "The apogee to perigee data portion is empty"

    if feature == 'Np':
        columns = ['Np']
    else:
        columns = [feature, 'Np']

    temp = temp[columns].dropna().sort_values(by='Np')
    temporary = temp[feature].values.flatten()
    # temporary = temporary[~np.isnan(temporary)]
    percentage = kwargs.get('percentage', 0.01)
    ref_feature = np.median(temporary[-int(len(temporary) * percentage):])
    assert len(temp) > 0, f"{percentage * 100}% of the apogee to perigee data portion is empty"

    return ref_feature


def compute_ref_MSH_feature(half_orbit_df, feature, percentage):
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


def get_ref_MSH_feature_over_time(apogee_times, perigee_times, df, feature, **kwargs):
    percentage = kwargs.get('percentage', 0.05)

    df['ref_MSH_' + feature] = np.nan

    if apogee_times[0] < perigee_times[0]:
        apogee_times = apogee_times[1:]

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