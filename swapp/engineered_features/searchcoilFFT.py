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
    if len(searchcoil.columns)>=3:
        searchcoil['searchcoil'] = np.sqrt(searchcoil[0] ** 2 + searchcoil[1] ** 2 + searchcoil[2] ** 2)
    elif len(searchcoil.columns)==1:
        searchcoil = pd.DataFrame(searchcoil.values, index = searchcoil.index.values, columns = ['searchcoil'])
    else:
        raise Exception("The searchcoil files has no columns or two columns. It should either have the three "
                        "components of searchcoil measures, or only the total searchcoil measures.")
    
    searchcoil = searchcoil.sort_index()
    if searchcoil.index.duplicated().sum() > 0:
        searchcoil = searchcoil[~searchcoil.index.duplicated(keep='first')]
    print('Searchcoil ready')

    return searchcoil[['searchcoil']]


def compute_searchcoil_fft(all_data, subdata, window_fft, searchcoil_dictionary, files_path, start_index=0,
                           save_name='', resolution=np.timedelta64(5, 's')):
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

        temp = searchcoil[time - resolution / 2:time + resolution / 2].values
        if len(temp) != window_fft - 1 and len(temp) != window_fft:
            print(
                f"The sub-searchcoil dataset should contain {window_fft - 1} or {window_fft} points, "
                f"but is {len(temp)}.")
            fft_result = np.nan * np.ones(window_fft - 1)
        else:
            fft_result = abs(fft(temp[:window_fft - 1])).flatten()
        del temp
        assert len(fft_result) == window_fft - 1, f"The FFT results should contain {window_fft - 1} points."
        all_data.loc[subdata.index.values[indice], 'fft_searchcoil'] = fft_result.sum()

        del fft_result

        if indice % 5000 == 0:
            pd.to_pickle(all_data[['fft_searchcoil']], save_path)
            print(indice)

    pd.to_pickle(all_data[['fft_searchcoil']], save_path)
