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


def compute_searchcoil_fft(all_data, subdata, window_fft, searchcoil_dictionary, files_path, start_index=0,
                           save_name='', resolution=np.timedelta64(5, 's'), verbose=False):
    files, starts, stops = (searchcoil_dictionary['files'], searchcoil_dictionary['starts'],
                            searchcoil_dictionary['stops'])
    save_path = f"/home/ghisalberti/make_datasets/{save_name}_searchcoil_fft.pkl"

    time = subdata.iloc[start_index].name
    searchcoil = download_searchcoil(time - resolution / 2, time + resolution / 2, files, starts, stops, files_path)
    start, stop = searchcoil.index.values[0], searchcoil.index.values[-1]

    if start_index == 0:
        all_data['fft_searchcoil'] = np.nan
    else:
        all_data['fft_searchcoil'] = pd.read_pickle(save_path).fft_searchcoil.values

    for indice in range(start_index, len(subdata)):

        time = subdata.iloc[indice].name
        if time - resolution / 2 < start or time + resolution / 2 > stop:
            searchcoil = download_searchcoil(time - resolution / 2, time + resolution / 2, files, starts, stops,
                                             files_path)
            start, stop = searchcoil.index.values[0], searchcoil.index.values[-1]

        temp = searchcoil[time - resolution / 2: time + resolution / 2].values
        if len(temp) < window_fft - 1:
            if verbose:
                print(
                    f"The sub-searchcoil dataset should contain at least {window_fft - 1} points, "
                    f"but is {len(temp)}.")
            fft_result = np.nan * np.ones(window_fft - 1)
        else:
            start_i = (len(temp) - window_fft + 1)//2
            fft_result = abs(fft(temp[start_i:start_i + window_fft - 1])).flatten()
        del temp
        assert len(fft_result) == window_fft - 1, (f"The FFT results should contain {window_fft - 1} points but "
                                                   f"contains {len(fft_result)} points.")
        all_data.loc[subdata.index.values[indice], 'fft_searchcoil'] = fft_result.sum()

        del fft_result

        if indice % 5000 == 0:
            pd.to_pickle(all_data[['fft_searchcoil']], save_path)
            pd.to_pickle(all_data[['fft_searchcoil']], save_path[:-4]+'_copy.pkl')

            print(indice)

    pd.to_pickle(all_data[['fft_searchcoil']], save_path)
    pd.to_pickle(all_data[['fft_searchcoil']], save_path[:-4] + '_copy.pkl')


def compute_fft_mission(sat, **kwargs):
    mission, path, files, dfs = get_info_mission_fft(sat)

    for df in dfs:
        compute_fft_file(df, mission, path, files, 0, **kwargs)


def compute_fft_file(df, sat, path, files, start_index, **kwargs):
    dt = kwargs.get('dt', np.timedelta64(5, 's'))
    window_fft = kwargs.get('window_fft', 30)

    start, stop = str(df.index.values[0])[:4], str(df.index.values[-1])[:4]

    # Initialization
    fft_Bt = initialize_fft(df, start_index, sat)
    B = read_best_file(df.index.values[start_index], dt, path, files)

    i = start_index
    for i in range(start_index, len(df)):
        t = df.index.values[i]
        B = provide_searchcoil_file(t, dt, B, files, path)

        Bt = B[t - dt / 2: t + dt / 2].Bt.values
        if len(Bt) < window_fft:
            print(
                f"The high resolution B subdataset around {t} should contain at least {window_fft} points, "
                f"but contains {len(Bt)} points.")
            fft_result = np.nan * np.ones(window_fft)
        else:
            Bt = Bt[:window_fft]
            fft_result = abs(fft(Bt[:window_fft])).flatten()
        del Bt
        assert len(fft_result) == window_fft, f"The FFT results around {t} should contain {window_fft} points."

        fft_Bt.loc[t, [f'fft_{i}' for i in range(window_fft)]] = fft_result
        fft_Bt.loc[t, 'fft_Bt'] = fft_result.sum()

        if i % 5000 == 0:
            fft_Bt.to_pickle(f'/home/ghisalberti/make_datasets/B_fluctuations/{sat}_searchcoil_fft_{start}_{stop}.pkl')
            print(f'FFT saved, i={i}')

    fft_Bt.to_pickle(f'/home/ghisalberti/make_datasets/B_fluctuations/{sat}_searchcoil_fft_{start}_{stop}.pkl')
    print(f'FFT saved, i={i}')


def get_info_mission_fft(sat):
    mission = get_mission(sat)
    path = f'/DATA/ghisalberti/Datasets/{mission}/{sat}/searchcoil/'
    df = pd.read_pickle(f'/DATA/ghisalberti/Datasets/{mission}/{sat}/{sat}_interesting_for_BL.pkl')
    # has to be subdata / interesting data
    files = pd.read_pickle(f'/home/ghisalberti/make_datasets/B_fluctuations/{sat}_high_resolution_B_info.pkl')
    return mission, path, files, [df]
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
        fft_Bt = pd.read_pickle(f'/home/ghisalberti/make_datasets/B_fluctuations/{sat}_B_fluctuations_fft.pkl')
    return fft_Bt


def read_file(path, files, i):
    sp = pd.read_pickle(path + files['files'][i])
    B = pd.DataFrame(sp.values, index=sp.time, columns=sp.columns)
    return B


def read_best_file(t, dt, path, files):
    names, starts, stops = files['files'], files['starts'], files['stops']
    files_ok = np.logical_and(t + dt / 2 <= stops, t - dt / 2 >= starts)
    nb_file = np.arange(len(names))[files_ok][0]
    B = read_file(path, files, nb_file)
    print(f'File {nb_file} downloaded')
    return B


def provide_searchcoil_file(t, dt, current_file, files, path):
    current_start, current_stop = current_file.index.values[0], current_file.index.values[-1]
    if t + dt / 2 > current_stop or t - dt / 2 < current_start:
        current_file = read_best_file(t, dt, path, files)
    return current_file
