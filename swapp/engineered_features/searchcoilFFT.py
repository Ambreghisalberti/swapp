import numpy as np
import pandas as pd
from scipy.fft import fft


def download_searchcoil(start, stop, files, starts, stops, path):
    indices = list(np.logical_and(np.logical_and(start >= starts, start <= stops), (stop <= stops)))
    if np.array(indices).sum() == 0:
        print(
            f'The {start}-{stop} interval is not comprised in a single searchcoil file. '
            f'Creating a dedicated file would be faster than merging two files.')
        indices = list(np.logical_and(start >= starts, start <= stops))

    i = indices.index(1)
    print(f'File {i} downloading...')
    searchcoil = pd.read_pickle(path + files[i] + '.pkl')
    print(f'File {i} downloaded')

    if stop < starts[i] or stop > stops[i]:
        print('2 files to download...')
        if i == len(files) - 1:
            raise Exception("One window is after the last time of searchcoil downloaded!")
        searchcoil2 = pd.read_pickle(path + files[i + 1] + '.pkl')
        assert searchcoil.index.values[-1] < searchcoil2.index.values[
            0], ("The 2 datasets to merge are overlapping, so a sort_index and remove duplicates step will be needed, "
                 "which is very long.")
        searchcoil = pd.concat([searchcoil, searchcoil2], axis=0)
        del searchcoil2
        print(f'...File {i + 1} also downloaded')

        searchcoil = searchcoil.sort_index()
        if searchcoil.index.duplicated().sum() > 0:
            searchcoil = searchcoil[~searchcoil.index.duplicated(keep='first')]

    if len(searchcoil.columns) >= 3:
        searchcoil['searchcoil'] = np.sqrt(searchcoil[0] ** 2 + searchcoil[1] ** 2 + searchcoil[2] ** 2)
    elif len(searchcoil.columns) == 1:
        searchcoil = pd.DataFrame(searchcoil.values, index=searchcoil.index.values, columns=['searchcoil'])
    else:
        raise Exception("The searchcoil files has no columns or two columns. It should either have the three "
                        "components of searchcoil measures, or only the total searchcoil measures.")

    print('Searchcoil ready')

    return searchcoil[['searchcoil']]


def compute_fft_mission(sat, **kwargs):
    path, files, dfs = get_info_mission_fft(sat)

    for df in dfs:
        compute_fft_file(df, sat, path, files, 0, **kwargs)


def compute_fft_file(df, sat, path, files, start_index, **kwargs):
    dt = kwargs.get('dt', np.timedelta64(5, 's'))
    window_fft = kwargs.get('window_fft', 30)

    start, stop = str(df.index.values[0])[:4], str(df.index.values[-1])[:4]

    # Initialization
    fft_Bt = initialize_fft(df, start_index, sat)
    B = read_best_file(df.index.values[start_index], dt, path, files)

    i = start_index
    count = 0
    for i in range(start_index, len(df)):
        t = df.index.values[i]
        B = provide_searchcoil_file(t, dt, B, files, path)

        if B is not None:
            Bt = B[t - dt / 2: t + dt / 2].Bt.values
            if len(Bt) < window_fft:
                count += 1
                fft_result = np.nan * np.ones(window_fft)
            else:
                Bt = Bt[:window_fft]
                fft_result = abs(fft(Bt[:window_fft])).flatten()
            del Bt
        else:
            fft_result = np.nan * np.ones(window_fft)
        assert len(fft_result) == window_fft, f"The FFT results around {t} should contain {window_fft} points."

        fft_Bt.loc[t, [f'fft_{i}' for i in range(window_fft)]] = fft_result
        fft_Bt.loc[t, 'fft_Bt'] = fft_result.sum()

        if i % 5000 == 0:
            fft_Bt.to_pickle(f'/home/ghisalberti/make_datasets/B_fluctuations/{sat}_searchcoil_fft_{start}_{stop}.pkl')
            print(f'FFT saved, i={i}.\nThe searchcoil subdataset around a time should contain at least '
                  f'{window_fft} points, but contains less for {count} times out of {i}.')

    fft_Bt.to_pickle(f'/home/ghisalberti/make_datasets/B_fluctuations/{sat}_searchcoil_fft_{start}_{stop}.pkl')
    print(f'FFT saved, i={i}')


def get_info_mission_fft(sat):
    mission = get_mission(sat)
    path = f'/DATA/ghisalberti/Datasets/{mission}/{sat}/searchcoil/'
    df = pd.read_pickle(f'/DATA/ghisalberti/Datasets/{mission}/{sat}/{sat}_interesting_for_BL.pkl')
    # has to be subdata / interesting data
    files = pd.read_pickle(f'/home/ghisalberti/make_datasets/B_fluctuations/{sat}_high_resolution_B_info.pkl')
    return path, files, [df]
    # If the file is too heavy, I will be able to find a way to give a list of several files instead of a merged one.


def get_mission(sat):
    if sat.startswith('MMS'):
        return 'MMS'
    elif sat.startswith('TH'):
        return 'THEMIS'
    else:
        raise Exception('This mission has not been added yet in the list of missions in get_mission function.')


def initialize_fft(df, start_index, sat):
    if start_index == 0:
        fft_Bt = pd.DataFrame([], index=df.index.values, columns=['fft_Bt'])
    else:
        fft_Bt = pd.read_pickle(f'/home/ghisalberti/make_datasets/B_fluctuations/{sat}_searchcoil_fft_'
                                f'{str(df.index.values[0])[:4]}_{str(df.index.values[-1])[:4]}.pkl')
    return fft_Bt


def read_file(path, files, i):
    sp = pd.read_pickle(path + files['files'][i])
    # B = pd.DataFrame(sp.values, index=sp.time, columns=sp.columns)
    return sp


def read_best_file(t, dt, path, files):
    names, starts, stops = files['files'], files['starts'], files['stops']
    files_ok = np.logical_and(t + dt / 2 <= stops, t - dt / 2 >= starts)
    if len(files_ok) > 0:
        nb_file = np.arange(len(names))[files_ok][0]
        B = read_file(path, files, nb_file)
        print(f'File {nb_file} downloaded')
        return B
    else:
        return None


def provide_searchcoil_file(t, dt, current_file, files, path):
    current_start, current_stop = current_file.index.values[0], current_file.index.values[-1]
    if t + dt / 2 > current_stop or t - dt / 2 < current_start:
        current_file = read_best_file(t, dt, path, files)
    return current_file
