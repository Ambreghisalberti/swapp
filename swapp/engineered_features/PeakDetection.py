from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from swapp.windowing.make_windows.utils import time_resolution
from multiprocessing import Pool


def energy_at(indexes, energy_values):
    """
    return the energy interpolated at the
    given decimal indexes from values
    """
    nbr_nrj_channels = len(energy_values)
    energy_idx = np.arange(nbr_nrj_channels)
    interp = interp1d(energy_idx, energy_values)
    return interp(indexes)


def find_populations(energy, flux, **kwargs):
    prominence = kwargs.get('prominence', np.max(flux)/2)
    count = kwargs.get('count_iterations', 0)

    if (count > 200) or (np.max(flux) == 0) or (prominence <= np.max(flux)/100):
        return {"energy": [],
                "width": [],
                "max": [],
                "lefts": [],
                "rights": []}

    peak_indexes, prop = find_peaks(flux,
                                    prominence=(prominence, None),
                                    width=0,
                                    rel_height=0.5,
                                    height=0)

    if len(peak_indexes) > 0:
        lefts = energy_at(prop['left_ips'], energy)
        rights = energy_at(prop['right_ips'], energy)
        widths = rights - lefts

        populations = {"energy": energy[peak_indexes],
                       "width": widths,
                       "max": prop["peak_heights"],
                       "lefts": lefts,
                       "rights": rights}

    else:  # no peak found even though non nul spectro, and high prominence is looked for

        peak_is_last_point = np.argmax(flux) == len(flux) - 1
        peak_is_first_point = np.argmax(flux) == 0

        if peak_is_last_point:
            delta_energy = energy[-1] - energy[-2]
            energy = np.concatenate((energy, (energy[-1] + delta_energy) * np.ones(1)))
            flux = np.concatenate((flux, np.zeros(1)))
            populations = find_populations(energy, flux, prominence=prominence, count_iterations=count+1)

        elif peak_is_first_point:
            delta_energy = energy[1] - energy[0]
            energy = np.concatenate((((energy[0] - delta_energy) * np.ones(1)), energy))
            flux = np.concatenate((np.zeros(1), flux))
            populations = find_populations(energy, flux, prominence=prominence, count_iterations=count+1)

        else:
            populations = find_populations(energy, flux, prominence=prominence/2, count_iterations=count + 1)

    return populations


def energy_flux_at(time, df):
    flux = df.loc[time][[f'spectro_{i}' for i in range(32)]].values.astype('float')
    energy = pd.read_pickle('/home/ghisalberti/make_datasets/datasets/spectro_energies.pkl')['spectro_energies']
    return energy, flux


def plot_populations(ax, populations):
    energies, widths, heights, lefts, rights = populations.values()
    ax.plot(energies, heights, "x")
    ax.hlines(np.array(heights) / 2, lefts, rights, color='red')


def check_pops(energy, flux):
    populations = find_populations(energy, flux)
    fig, ax = plt.subplots()
    ax.plot(energy, flux, "+")
    ax.set_xscale("log")
    plot_populations(ax, populations)
    return fig, ax, populations


def get_pops(start, stop, df):
    times = df[start:stop].index.values
    pop_times = []
    for t in times:
        populations = find_populations(*energy_flux_at(t, df))
        pop_times += [populations]

    energy_main_pop = np.array(
        [pop['energy'][np.argmax(np.array(pop["max"]))] if len(pop['max']) > 0 else np.nan for pop in pop_times])
    width_main_pop = np.array(
        [pop['width'][np.argmax(np.array(pop["max"]))] if len(pop['max']) > 0 else np.nan for pop in pop_times])
    flux_main_pop = np.array([np.max(np.array(pop["max"])) if len(pop['max']) > 0 else np.nan for pop in pop_times])

    info_pop = {'energy_main_pop': energy_main_pop, 'width_main_pop': width_main_pop, 'flux_main_pop': flux_main_pop}
    return info_pop


def temporal_pops(start, stop, df):
    info_pop = get_pops(start, stop, df)
    energy_main_pop, width_main_pop, flux_main_pop = info_pop.values()

    fig, ax = plt.subplots(figsize=(15, 5))
    times = df[start:stop].index.values
    ax.plot(times, flux_main_pop, label='max')
    ax.legend()
    ax1 = ax.twinx()
    ax1.plot(times, energy_main_pop, color='orange', label='center')
    ax1.plot(times, width_main_pop, color='red', label='width')
    ax1.legend()


def get_pops_df(start, stop, df):
    info_pop = get_pops(start, stop, df)
    energy_main_pop, width_main_pop, flux_main_pop = info_pop.values()

    df_temp = df[start:stop][['label_BL']]
    df_temp['energy_main_pop'] = energy_main_pop
    df_temp['width_main_pop'] = width_main_pop
    df_temp['flux_main_pop'] = flux_main_pop

    return df_temp


def save_pops(start, stop, df, **kwargs):
    name = kwargs.get('name', f'{str(start)[:10]}_{str(stop)[:10]}')
    df_temp = get_pops_df(start, stop, df)
    df_temp.to_pickle(f'/home/ghisalberti/make_datasets/detected_peaks/peaks_{name}.pkl')
    #df_temp.to_pickle(f'/home/ghisalberti/make_datasets/detected_peaks/peaks_and_fft_{name}{str(start)[:10]}_'
    #                  f'{str(stop)[:10]}.pkl')
    if kwargs.get('verbose', False):
        print(f'Peaks found and saved from {str(start)[:10]} to {str(stop)[:10]}.')


def save_pops_tuple(inputs):
    if isinstance(inputs[-1], dict):
        save_pops(*inputs[:-1], **inputs[-1])
    else:
        save_pops(*inputs)


def optimized_save_pops(start, stop, df, **kwargs):
    kwargs['name'] = kwargs.pop('name', f'{str(start)[:10]}_{str(stop)[:10]}')

    starts = pd.date_range(start, stop, freq=time_resolution(df)*100000, inclusive='left')
    stops = list(starts[1:])+[stop]
    nb_threads = kwargs.get('nb_threads', 15)
    nb_batches = int(np.ceil(len(starts)/nb_threads))

    for i in range(nb_batches):
        sub_starts = starts[i*nb_threads:(i+1)*nb_threads]
        sub_stops  = stops[ i*nb_threads:(i+1)*nb_threads]

        with Pool(15) as p:
            p.map(save_pops_tuple, zip(sub_starts, sub_stops, [kwargs for _ in range(len(sub_starts))]))


def stat_pops(start, stop, df):
    df_temp = get_pops_df(start, stop, df)
    energy = pd.read_pickle('/home/ghisalberti/make_datasets/spectro_energies.pkl')['spectro_energies']
    bins = np.logspace(np.log10(energy[0]), np.log10(energy[-1]), 30)
    pp = sns.pairplot(df_temp, hue='label_BL', kind='hist', diag_kws={'log_scale': True, 'bins': bins,
                                                                      'stat': 'density'},
                      plot_kws={'alpha': 0.7, 'log_scale': True, 'bins': 20})
    for ax in pp.axes.flat:
        log_columns = ['energy_main_pop', 'width_main_pop']
        if ax.get_xlabel() in log_columns:
            ax.set(xscale="log")
        if ax.get_ylabel() in log_columns:
            ax.set(yscale="log")
    pp.fig.tight_layout()
    return df_temp
