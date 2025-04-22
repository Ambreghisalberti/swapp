import numpy as np
import pandas as pd
import speasy as spz
import time
from swapp.catalogues import resolution_to_string
from swapp.windowing.make_windows.utils import time_resolution


def download_product_interval(path, start, stop, **kwargs):
    try:
        product = spz.get_data(path, start, stop)
    except:
        product = None

    if product is not None:
        product = pd.DataFrame(data=product.values, index=product.time)
        resolution = kwargs.get('resolution', time_resolution(product))
        resolution = resolution_to_string(pd.to_timedelta(resolution))
        product = product.resample(resolution).mean()[start:stop]
    else:
        product = pd.DataFrame(data=[])

    return product


def download_product(inputs):
    if len(inputs) == 7:
        product_name, path, start, stop, dt, mission, satellite = inputs
        kwargs = {}
    else:
        product_name, path, start, stop, dt, mission, satellite, kwargs = inputs

    description = kwargs.get('description', '')
    if description != '':
        description = '_' + description

    N = int(np.ceil((stop - start) / dt))
    intervals = [(start + i * dt, start + (i + 1) * dt) for i in range(N)]

    print(product_name + " downloading...\n")
    t1 = time.time()

    product = download_product_interval(path, intervals[0][0], intervals[0][1], **kwargs)
    product.to_pickle(
        '/DATA/ghisalberti/Datasets/' + mission + '/' + satellite + f'/{product_name}_{start.year}_{stop.year}'
                                                                    f'{description}.pkl')

    for interval in intervals[1:]:
        prod = download_product_interval(path, interval[0], interval[1], **kwargs)
        product = pd.concat([product, prod])
        product.to_pickle(
            '/DATA/ghisalberti/Datasets/' + mission + '/' + satellite + f'/{product_name}_'
                                                                        f'{start.year}_{stop.year}'
                                                                        f'{description}.pkl')

    t2 = time.time()
    print(product_name + f" for {satellite} between {start} and {stop} is downloaded in {t2 - t1} "
                         f"seconds!\n",
          flush=True)

    return None


def merge_files(path, name_product, starts, stops, description=''):
    if description != '':
        description = '_' + description

    all_prod = pd.read_pickle(path + name_product + f'_{starts[0].year}_{stops[0].year}{description}.pkl')
    for start, stop in zip(starts[1:], stops[1:]):
        prod = pd.read_pickle(path + name_product + f'_{start.year}_{stop.year}{description}.pkl')
        all_prod = pd.concat([all_prod, prod])

    all_prod.to_pickle(path + name_product + f'_{starts[0].year}_{stops[-1].year}{description}.pkl')
    check_monotony(all_prod)
    check_duplicates(all_prod)


def check_monotony(df):
    assert df.index.is_monotonic_increasing, "The merged product dates are not monotonic increasing."


def check_duplicates(df):
    assert df.index.duplicated.sum() == 0, "The merged product has duplicate dates."


def check_name_unicity(starts, stops):
    start0, stop0 = starts[0], stops[0]
    for start, stop in zip(starts[1:], stops[1:]):
        if (start.year == start0.year) & (stop.year == stop0.year):
            return False
        start0, stop0 = start, stop
    return True
