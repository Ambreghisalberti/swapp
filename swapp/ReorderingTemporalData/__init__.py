import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.signal import medfilt
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA, KernelPCA
from scipy.ndimage import gaussian_filter


def find_nearest_neighbour_in_free_list(distance_matrix, indice, free_points_booleans):
    free_indices = np.arange(len(free_points_booleans))[free_points_booleans]
    min_index = np.argmin(distance_matrix[indice][free_points_booleans])
    return free_indices[min_index]


def find_path_nearest_neighbours_locally(distance_matrix):
    free_points_booleans = np.array([True for _ in range(len(distance_matrix))])

    indice = 0
    path = [indice]
    free_points_booleans[indice] = False

    while free_points_booleans.sum() > 0:
        indice = find_nearest_neighbour_in_free_list(distance_matrix, indice, free_points_booleans)
        path += [indice]
        free_points_booleans[indice] = False

    return path


def find_nearest_neighbour_in_taken_list(distance_matrix, indice, taken_points_booleans):
    min_index = np.argmin(distance_matrix[indice][taken_points_booleans])
    return min_index   # Position in taken_points_booleans


def find_path_nearest_neighbours(distance_matrix, **kwargs):
    """Animation of the insertion of every point in the find_path_nearest_neighbours method"""
    verbose = kwargs.get('verbose', False)

    if verbose:
        pause_time = kwargs.get('pause_time', 1)
        points = kwargs.get('points', [[]])
        plt.ion()
        if len(points[0]) != 2:
            raise Exception(
                f"The kwargs verbose=True for this function is designed for 2D data visualization, "
                f"and the data given in input is {len(points[0])}D.")

        fig, ax = plt.subplots()
        ax.scatter(points[:, 0], points[:, 1], marker='+')

    taken_points_booleans = np.array([False for _ in range(len(distance_matrix))])

    path = [0, len(distance_matrix) - 1]
    taken_points_booleans[0] = True
    taken_points_booleans[-1] = True

    if verbose:
        ax.plot(points[:, 0][path], points[:, 1][path])
        plt.show(block=False)
        plt.pause(pause_time)
        plt.close()
        clear_output(wait=True)

    while taken_points_booleans.sum() < len(distance_matrix):

        # indice = np.random.choice(np.arange(len(distance_matrix))[~taken_points_booleans], size=1)[0]
        indice = np.arange(len(distance_matrix))[~taken_points_booleans][0]
        added_distances = []
        for i_path in range(len(path) - 1):
            # Added length if inserting between i_path and i_path + 1
            added_distances += [
                distance_matrix[indice, path[i_path]] + distance_matrix[indice, path[i_path + 1]] - distance_matrix[
                    path[i_path], path[i_path + 1]]]
        best_insertion = np.argmin(added_distances)
        path = path[:best_insertion + 1] + [indice] + path[best_insertion + 1:]
        taken_points_booleans[indice] = True

        if verbose:
            fig, ax = plt.subplots()
            ax.scatter(points[:, 0], points[:, 1], marker='+')
            ax.plot(points[:, 0][path], points[:, 1][path])
            ax.scatter(points[indice, 0], points[indice, 1], s=30, color='red')
            plt.pause(pause_time)
            plt.close()
            clear_output(wait=True)

    if verbose:
        plt.ioff()
        plt.show()

    return path


def find_shortest_path_PCA(df):
    pca = KernelPCA(n_components=1)
    projection = pca.fit_transform(df)

    projection = pd.DataFrame(projection, columns=['projection_1D'], index=np.arange(len(df))).sort_values(
        by='projection_1D')
    path = projection.index.values
    # Automatic way to know if the order is good or opposite
    dist1 = np.sqrt((df.iloc[path[0]] - df.iloc[0])**2).sum()
    dist2 = np.sqrt((df.iloc[path[0]] - df.iloc[-1])**2).sum()
    if dist1 > dist2:
        path = path[::-1]

    return path


def plot_temporal_and_reordered_feature(feature, path, df, **kwargs):
    show_smoothed = kwargs.get('show_smoothed', False)
    kernel_filter = kwargs.get('kernel_filter', 5)

    if ('fig' not in kwargs) or ('ax' not in kwargs):
        fig, ax = plt.subplots()
    else:
        fig, ax = kwargs['fig'], kwargs['ax']

    ax.plot(df.index.values, df[feature].values.flatten(), label='temporal')
    if show_smoothed:
        # ax.plot(smoothed_df.index.values, smoothed_df[feature].values.flatten(), label='Smoothed temporal')
        ax.plot(df.index.values, gaussian_filter(df[feature].values.flatten()[path], kernel_filter), label='Smoothed spatial')
    else:
        ax.plot(df.index.values, df[feature].values.flatten()[path], label='spatial')

    ax.legend()


def plot_temporal_and_reordered(path, df, **kwargs):
    columns = kwargs.pop('columns', df.columns)
    nrows = len(columns)
    fig, ax = plt.subplots(nrows=nrows, figsize=(12, nrows * 1.5))

    for i, col in enumerate(columns):
        plot_temporal_and_reordered_feature(col, path, df, fig=fig, ax=ax[i], **kwargs)
        ax[i].set_ylabel(col)
        ax[i].set_xlabel('Time')
    if 'title' in kwargs:
        fig.suptitle(kwargs['title'])

    plt.tight_layout()


def plot_path_feature(feature, path, df, **kwargs):
    show_smoothed = kwargs.get('show_smoothed', False)
    kernel_filter = kwargs.get('kernel_filter', 5)

    if ('fig' not in kwargs) or ('ax' not in kwargs):
        fig, ax = plt.subplots()
    else:
        fig, ax = kwargs['fig'], kwargs['ax']

    if show_smoothed:
        # ax.plot(smoothed_df.index.values, smoothed_df[feature].values.flatten(), label='Smoothed temporal')
        ax.plot(df.index.values, gaussian_filter(df[feature].values.flatten()[path], kernel_filter),
                label=f"Smoothed {kwargs.get('label','reordered')}")
    else:
        ax.plot(df.index.values, df[feature].values.flatten()[path],
                label=kwargs.get('label','reordered'))


def plot_path(path, df, **kwargs):
    columns = kwargs.pop('columns', df.columns)
    if 'fig' in kwargs and 'ax' in kwargs:
        fig, ax = kwargs.pop('fig'), kwargs.pop('ax')
    else:
        nrows = len(columns)
        fig, ax = plt.subplots(nrows=nrows, figsize=(12, nrows * 1.5))

    for i, col in enumerate(columns):
        plot_path_feature(col, path, df, fig=fig, ax=ax[i], **kwargs)
        ax[i].set_ylabel(col)
        ax[i].set_xlabel('Time')
    if 'title' in kwargs:
        fig.suptitle(kwargs['title'])

    plt.tight_layout()


def reorder(df, columns_to_reorder, **kwargs):
    columns_to_plot = kwargs.pop('columns_to_plot', ['B', 'Bx', 'By', 'Bz', 'Np', 'Vx', 'Vy', 'Tpara', 'Tperp'])

    nrows = len(columns_to_plot)
    fig, ax = plt.subplots(nrows=nrows, figsize=(12, nrows * 1.5))

    values = df.values
    # values = medfilt(values, kernel_size=[kernel_medfilt, 1])
    smoothed_data = pd.DataFrame(values, index=df.index.values, columns=[df.columns])

    values = smoothed_data[columns_to_reorder].values
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(values)

    distance_matrix = cdist(scaled_values, scaled_values)

    if kwargs.get('insertion', False):
        path = find_path_nearest_neighbours(distance_matrix)
        plot_path(path, df, columns=columns_to_plot, ncols=3, label='insertion',
                  fig=fig, ax=ax, **kwargs)

    if kwargs.get('closest', False):
        path = find_path_nearest_neighbours_locally(distance_matrix)
        plot_path(path, df, columns=columns_to_plot, ncols=3, label='closest',
                  fig=fig, ax=ax, **kwargs)

    if kwargs.get('PCA', False):
        path = find_shortest_path_PCA(pd.DataFrame(scaled_values, index=smoothed_data.index.values,
                                                   columns=columns_to_reorder))
        plot_path(path, df, columns=columns_to_plot, ncols=3, label='PCA',
                  fig=fig, ax=ax, **kwargs)

    if kwargs.get('NoverT', False):
        df['numeric_index'] = np.arange(len(df))
        path = df.sort_values(by='logNoverT').numeric_index.values
        if np.nanmean(df['B'].values[:5]) < np.nanmean(df['B'].values[-5:]):  # MSH is on the left and MSP on the right
            path = path[::-1]
        plot_path(path, df, columns=columns_to_plot, ncols=3, label='NoverT',
                  fig=fig, ax=ax, **kwargs)

    plot_path(np.arange(len(df)), df, columns=columns_to_plot, ncols=3, label='temporal',
              fig=fig, ax=ax, **kwargs)

    for a in ax:
        a.legend(bbox_to_anchor=(1.3, 0.5, 0.4, 0.5))    # Changer ça, trouver le meilleur!

    return path
