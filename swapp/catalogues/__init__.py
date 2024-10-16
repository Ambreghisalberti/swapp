import tscat
import json
import pandas as pd
import numpy as np
from datetime import datetime

def create_catalogue(starts, stops, name, author='', **kwargs):
    """
    start and stop are lists or arrays of start and stop times, """
    catalogue = tscat.create_catalogue(name=name, author=author, events=[])
    events = []

    are_tags = False
    if 'tags' in kwargs:
        are_tags = True
        tags = kwargs['tags']
        assert len(tags)==len(starts), "For the current version of this function, you need to give tags for every event."

    for i, (start, stop) in enumerate(zip(starts, stops)):
        start = dates_to_datetime(start, **kwargs)
        stop = dates_to_datetime(stop, **kwargs)
        if are_tags:
            events += [tscat.create_event(start=start, stop=stop, author=author, tags=tags[i])]
        else:
            events += [tscat.create_event(start=start, stop=stop, author=author)]
    tscat.add_events_to_catalogue(catalogue, events)

    return catalogue


def read_catalogue_events(path):
    with tscat.Session():
        with open(path) as json_data:
            d = json.load(json_data)
            catalogue = d['events']
    return catalogue


def resolution_to_string(resolution):
    frequency = str(resolution)  # Ex: numpy.timedelta64(90,'s')
    [nbr, text] = frequency.split(' ')  # Ex: ['90','seconds']
    return nbr + text[0]  # Ex: '90s'


def dates_to_datetime(date, **kwargs):
    if isinstance(date, datetime):
        return date
    elif isinstance(date, str):
        format = kwargs.get('format', '%Y-%m-%dT%H:%M:%S.%f')
        return datetime.strptime(date, format)
    elif isinstance(date, np.datetime64):
        return dates_to_datetime(str(date)[:22],format='%Y-%m-%dT%H:%M:%S.%f')


def catalogue_to_edges_df(path,
                          resolution: np.timedelta64):
    """ Precondition: resolution has to be a string in the format '5S' for 5 seconds for example.
    It will round up dates to the closest 5s"""

    resolution = resolution_to_string(resolution)
    catalogue = read_catalogue_events(path)
    events = pd.DataFrame(columns=['begin', 'end'])
    for ev in catalogue:
        pd.to_datetime(catalogue[0]['start']).round(resolution)
        events.loc[len(events)] = {'begin': pd.to_datetime(ev['start']).round(resolution),
                                   'end': pd.to_datetime(ev['stop']).round(resolution)}

        # events.loc[len(events)] = {'begin':ev['start'][:19], 'end':ev['stop'][:19]}
        '''The [:19] is to have dates in seconds and not in between seconds 
        (for a problem of number of points in make_windows, of which I'm not sure about anymore)'''

    return events


def export_catalogue(catalogue, name, path='/home/ghisalberti/catalogues/'):
    jc = tscat.export_json(catalogue)
    with open(path + name + ".json", "w") as outfile:
        outfile.write(jc)


def merge_catalogues(catalogues, **kwargs):
    """ catalogues must be a list of catalogues.
    The user can specify a new name and author, but by default the name and author
    of the first catalogue will be used for the merged one."""
    author = catalogues[0]['catalogues'][0]['author']
    merged_catalogue = tscat.create_catalogue(name=catalogues[0]['catalogues'][0]['name'], author=author, events=[])
    events = []

    for catalogue in catalogues:
        for ev in catalogue['events']:
            start = dates_to_datetime(ev['start'], **kwargs)
            stop = dates_to_datetime(ev['stop'], **kwargs)
            events += [tscat.create_event(start=start, stop=stop, author=author)]

    tscat.add_events_to_catalogue(merged_catalogue, events)
    return merged_catalogue


def duplicate_catalogue(path_catalogue: str):
    with tscat.Session() as s:
        with open(path_catalogue) as json_data:
            data = json.load(json_data)
            catalogue = data['catalogues'][0]  # Will only duplicate the first catalogue of the file

        duplicate = tscat.create_catalogue(name=catalogue['name'], author=catalogue['author'], tags=catalogue['tags'],
                                           predicate=catalogue['predicate'], events=[])
        events = []

        for ev in data['events']:
            events += [tscat.create_event(start=datetime.strptime(ev['start'], '%Y-%m-%dT%H:%M:%S.%f'),
                                          stop=datetime.strptime(ev['stop'], '%Y-%m-%dT%H:%M:%S.%f'),
                                          author=ev['author'], tags=ev['tags'], products=ev['products'])]
        tscat.add_events_to_catalogue(duplicate, events)

    return duplicate


def export_windows_to_catalogues(df_windows, win_length, time_resolution, win_duration):
    assert len(
        df_windows) % win_length == 0, \
        "The select_window functions does not return a dataframe compatible with win_length"

    starts = df_windows.index.values[::win_length]
    stops = df_windows.index.values[win_length - 1::win_length]
    assert (stops - starts != win_duration - time_resolution).sum() == 0, \
        "The obtained windows don't have the expected duration."

    date = str(datetime.now())[:10]
    dt = str(win_duration).split(' ')
    dt = dt[0] + dt[1]
    catalogue = create_catalogue(starts, stops, f'full_MSP_MSH_windows_dt={dt}_{date}', author='ghisalberti')
    export_catalogue(catalogue, f'full_MSP_MSH_windows_dt={dt}_{date}', path='/home/ghisalberti/catalogues/')