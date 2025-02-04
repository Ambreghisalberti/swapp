import time
from scipy.interpolate import interp1d
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import speasy as spz


def transform_to_mms_energy_channels(channels, fluxes, mms_channels, verbose=False):
    fluxes_df = pd.DataFrame(fluxes, columns=['flux'])
    fluxes_df['channel'] = channels
    fluxes_df = fluxes_df.sort_values(by='channel')

    # Keep only energies to keep (remove the highest energy because the flux is always nan for themis
    # and remove the energies at nan)
    fluxes_df = fluxes_df.dropna()

    if len(fluxes_df) > 0:
        # Find how many mms energies are outside of new range and will have to be extrapolated
        nb_extrapolated_under = (mms_channels < np.min(fluxes_df.channel.values)).sum()
        mms_channels_inside = mms_channels[nb_extrapolated_under:]
        nb_extrapolated_over = (mms_channels > np.max(fluxes_df.channel.values)).sum()
        if nb_extrapolated_over > 0:
            mms_channels_inside = mms_channels_inside[:-nb_extrapolated_over]
        if len(mms_channels_inside) == 0:   # All mms energies channels are outside of current energy channels range"
            spectro_energies_mms = [np.nan for _ in range(len(mms_channels))]
        else:
            # Interpolate
            spectro_energies_mms = interp1d(fluxes_df.channel.values, fluxes_df.flux.values)(mms_channels_inside)
            # Extrapolate by repeating the edge values
            spectro_energies_mms = [spectro_energies_mms[0]] * nb_extrapolated_under + list(spectro_energies_mms)
            spectro_energies_mms += [spectro_energies_mms[-1]] * nb_extrapolated_over

            if verbose:
                plt.figure()
                plt.plot(mms_channels, spectro_energies_mms, label='mms energies')
                plt.plot(channels, fluxes, label='themis energies')
                plt.legend()
                plt.xlabel('Energies')
                plt.ylabel('Ion flux')

    else:
        spectro_energies_mms = [np.nan for _ in range(len(mms_channels))]

    return spectro_energies_mms


def interpolate_spectro(product, interpolated_spectro, energies):
    warnings.simplefilter("ignore")
    interpolated = []
    for i in range(len(product.values)):
        spectro = transform_to_mms_energy_channels(product.axes[1].values[i], product.values[i], energies)
        interpolated += [spectro]
    interpolated_spectro = pd.concat(
        [interpolated_spectro, pd.DataFrame(np.array(interpolated), index=product.time, columns=np.arange(32))])

    return interpolated_spectro


def download_and_interpolate_spectro(inputs):
    product_name, path, start, stop, dt, mission, satellite = inputs
    interpolated_spectro = pd.DataFrame([], columns=np.arange(32))

    energies = pd.read_pickle('/home/ghisalberti/make_datasets/datasets/spectro_energies.pkl')['spectro_energies']
    N = int(np.ceil((stop - start) / dt))
    intervals = [(start + i * dt, start + (i + 1) * dt) for i in range(N)]

    print(product_name + " downloading...\n")
    t1 = time.time()

    product = spz.get_data(path, intervals[0][0], intervals[0][1])
    if product is not None:
        interpolated_spectro = interpolate_spectro(product, interpolated_spectro, energies)
    interpolated_spectro.to_pickle(
        '/DATA/ghisalberti/Datasets/' + mission + f'/{satellite.upper()}/'
                                                  f'{product_name}_interpolated_{start.year}_{stop.year}.pkl')

    for i, interval in enumerate(intervals[1:]):
        prod = spz.get_data(path, interval[0], interval[1])

        if not (prod is None):
            interpolated_spectro = interpolate_spectro(prod, interpolated_spectro, energies)
            interpolated_spectro.to_pickle(
                '/DATA/ghisalberti/Datasets/' + mission + '/' + satellite.upper() +
                f'/{product_name}_interpolated_{start.year}_{stop.year}.pkl')
        # if i%(len(intervals)//100) == 0:
        #    print(f"Interval {i}/{len(intervals)} is downloaded and interpolated", flush = True)

    t2 = time.time()
    print(product_name + f" is downloaded and interpolated in {t2 - t1} seconds!\n", flush=True)

    return None
