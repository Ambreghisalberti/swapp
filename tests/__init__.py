from ddt import ddt, data
import unittest
from windows.windowing.make_windows.utils import *
import pandas as pd
from windows.windowing.make_windows import prepare_df
from windows.windowing.make_windows.utils import time_resolution

all_data = pd.read_pickle('/windowing/data/MMS1_data_GSM_5S_2015_2021.pkl')
position = pd.read_pickle('/windowing/data/MMS1_pos_GSM_5S_2015_2021.pkl')
omni = pd.read_pickle('/windowing/data/OMNI_data_5S_2015_2021.pkl')

with open('/windowing/data/data/list_label_catalogues.ts') as file:
    paths = file.read().splitlines()


class BuildWindows():
    def __init__(self, all_data, position, omni, win_duration, paths, **kwargs):
        self.win_duration = win_duration
        self.resolution = time_resolution(all_data)
        self.win_length = durationToNbrPts(self.win_duration, self.resolution)

        processed_data, processed_pos, processed_omni = prepare_df(all_data, position, omni, self.win_length, paths,
                                                                   **kwargs)

        self.data = processed_data
        self.pos = processed_pos
        self.omni = processed_omni
        self.dataset = pd.concat([self.data, self.pos], axis=1)


win_durations = []
test_windowings = []
for win_duration in win_durations:
    test_windowings += [BuildWindows(all_data, position, omni, win_duration, paths)]


@ddt
class TestOriginalData(unittest.TestCase):

    def have_same_index(self, df1, df2):
        self.assertEqual(len(df1), len(df2))
        times1 = df1.index.values
        times2 = df2.index.values
        for i in range(len(times1)):
            self.assertEqual(times1[i], times2[i])


    @data(*test_windowings)
    def test_pos_omni_have_same_index(self):
        self.have_same_index(self.pos, self.omni)


'''
    @data((position, processed_data))
    @unpack
    def test_pos_data_have_same_index(self, pos, df):
        self.have_same_index(pos, df)

    @data((omni_data, processed_data))
    @unpack
    def test_omni_data_have_same_index(self, omni, df):
        self.have_same_index(omni, df)

    @data((processed_data, position))
    @unpack
    def test_more_nans_in_data_than_pos(self, df, pos):
        self.assertLessEqual(len(df.dropna()), len(pos.dropna()))


@ddt
class TestNumberWindows(unittest.TestCase):

    def smaller_than_total_windows(self, df, win_length, column):  # Nom plus court?
        self.assertLessEqual(df[column].sum(), nbr_windows(df, win_length))

    @data((dataset, win_length))
    @unpack
    def test_less_empty_windows_than_total_windows(self, df, win_length):
        self.smaller_than_total_windows(df, win_length, 'isEmpty')

    @data((dataset, win_length))
    @unpack
    def test_less_partial_windows_than_total_windows(self, df, win_length):
        self.smaller_than_total_windows(df, win_length, 'isPartial')

    @data((dataset, win_length))
    @unpack
    def test_less_full_windows_than_total_windows(self, df, win_length):
        self.smaller_than_total_windows(df, win_length, 'isFull')

    @data((dataset, win_length))
    @unpack
    def test_less_windows_close_to_MP_than_total_windows(self, df, win_length):
        self.smaller_than_total_windows(df, win_length, 'isCloseToMP')

    @data((dataset, win_length))
    @unpack
    def test_less_windows_encountering_MSP_MSH_than_total_windows(self, df, win_length):
        self.smaller_than_total_windows(df, win_length, 'encountersMSPandMSH')

    @data((dataset, win_length))
    @unpack
    def test_less_labelled_windows_than_total_windows(self, df, win_length):
        self.smaller_than_total_windows(df, win_length, 'isLabelled')


@ddt
class TestWindows(unittest.TestCase):

    def size_windows(self, df, win_length, win_duration, time_index):
        #resolution = df_resolution(df)
        for t in time_index:
            self.assertEqual(len(get_window(df,t,win_duration)), win_length)


@ddt
class TestSizeWindows(TestWindows, unittest.TestCase):

    @data((dataset, win_length, win_duration))
    @unpack
    def test_size_empty_windows(self, df, win_length, win_duration):
        self.size_windows(df, win_length, win_duration, df[df.isEmpty.values].index.values)

    @data((dataset, win_length, win_duration,))
    @unpack
    def test_size_full_windows(self, df, win_length, win_duration):
        self.size_windows(df, win_length, win_duration, df[df.isFull.values].index.values)

    @data((dataset, win_length, win_duration))
    @unpack
    def test_size_partial_windows(self, df, win_length, win_duration):
        self.size_windows(df, win_length, win_duration, df[df.isPartial.values].index.values)

    @data((dataset, win_length, win_duration))
    @unpack
    def test_size_windows_close_to_MP(self, df, win_length, win_duration):
        self.size_windows(df, win_length, win_duration, df[df.isCloseToMP.values].index.values)

    @data((dataset, win_length, win_duration))
    @unpack
    def test_size_windows_encountering_MSP_MSH(self, df, win_length, win_duration):
        self.size_windows(df, win_length, win_duration, df[df.encountersMSPandMSH.values].index.values)

    @data((dataset, win_length, win_duration))
    @unpack
    def test_size_labelled_windows(self, df, win_length, win_duration):
        self.size_windows(df, win_length, win_duration, df[df.isLabelled.values].index.values)


@ddt
class TestStrictAuxiliaryData(TestWindows, unittest.TestCase):

    @data((dataset, win_duration))
    @unpack
    def test_empty_windows_are_empty(self, df, win_duration):
        self.size_windows(df[['Bx', 'By', 'Bz', 'Np', 'Vx', 'Vy', 'Vz', 'Tp']].dropna(), 0, win_duration,
                          df[df.isEmpty.values].index.values)

    @data((dataset, win_length, win_duration))
    @unpack
    def test_full_windows_are_full(self, df, win_length, win_duration):
        self.size_windows(df[['Bx', 'By', 'Bz', 'Np', 'Vx', 'Vy', 'Vz', 'Tp']].dropna(), win_length, win_duration,
                          df[df.isFull.values].index.values)

    @data((dataset, win_length, win_duration))
    @unpack
    def test_partial_windows_are_partial(self, df, win_length, win_duration):
        resolution = df_resolution(df)
        time_indices = df[df.isPartial.values].index.values[1:]
        for t in time_indices:
            self.assertLess(
                len(get_window_features(df, t, win_duration,
                                        ['Bx', 'By', 'Bz', 'Np', 'Vx', 'Vy', 'Vz', 'Tp']).dropna()),
                win_length)
            self.assertGreaterEqual(
                len(get_window_features(df, t, win_duration, 
                ['Bx', 'By', 'Bz', 'Np', 'Vx', 'Vy', 'Vz', 'Tp']).dropna()),
                0)

    @data((dataset, win_length))
    @unpack
    def test_complementarity_full_empty_partial(self, df, win_length):
        self.assertEqual(df.isFull.sum() + df.isEmpty.sum() + df.isPartial.sum(), nbr_windows(df, win_length))
        self.assertEqual(
            (df.isEmpty.values.astype(int) + df.isFull.values.astype(int) + df.isPartial.values.astype(int) > 1).sum(),
            0)

    @data(dataset)
    def test_regions_are_complementary(self, df):
        nbr_regions = df.isMSP.values + df.isMSH.values + df.isSW.values
        for val in nbr_regions:
            self.assertEqual(val, 1)

    @data((dataset, win_length, win_duration))
    @unpack
    def test_encountering_MSP_and_MSH(self, df, win_length, win_duration):
        resolution = df_resolution(df)
        time_indices = df[df.encountersMSPandMSH.values].index.values
        for t in time_indices:
            [nbr_MSP, nbr_MSH] = get_window_features(df, t, win_duration, ['isMSP','isMSH']).sum()
            self.assertLess(nbr_MSP, win_length)
            self.assertGreater(nbr_MSP, 0)
            self.assertLess(nbr_MSH, win_length)
            self.assertGreater(nbr_MSH, 0)

    @data(dataset)
    def test_BL_points_are_labelled(self, df):
        are_labelled = df[df.label.values.astype(bool)].labelled_data.values
        for val in are_labelled:
            self.assertEqual(val, True)

    @data(dataset)
    def test_windows_with_BL_points_contain_BL(self, df):
        contains_BL = df[df.nbrLabelledBL.values > 0].containsLabelledBL.values
        for val in contains_BL:
            self.assertEqual(val, True)

    @data(dataset)
    def test_windows_containing_BL_have_BL_points(self, df):
        nbr_BL = df[df.containsLabelledBL.values].nbrLabelledBL.values
        for val in nbr_BL:
            self.assertGreaterEqual(val, 0)

    @data(dataset)
    def test_isRegion_really_is_region(self, df):
        values, counts = np.unique(df.regions_pred.values, return_counts=True)
        is_regions = ['isMSP', 'isMSH', 'isSW']
        for i, isRegion in enumerate(is_regions):
            self.assertEqual(counts[i], df[isRegion].sum())

    @data(dataset, win_length, win_duration)
    @unpack
    def test_msp_msh_windows_have_little_sw(self, df, win_length, win_duration):
        time_indices = df[df.encountersMSPandMSH.values].index.values
        for t in time_indices:
            nbr_SW = get_window_features(df, t, win_duration, 'isSW').sum()
            self.assertLess(nbr_SW, win_length * 0.8)

    @data(dataset)
    def test_bl_points_are_close_to_MP(self, df):
        subdf = df[df.label.values.astype(bool)]
        time_indices = subdf.index.values
        relative_dist_to_mp = abs(subdf.r_mp.values - subdf.R.values) / subdf.r_mp.values
        for t, val in zip(time_indices, relative_dist_to_mp):
            self.assertLess(val, 0.4)


@ddt
class TestFlexibleAuxiliaryData(unittest.TestCase):

    def setUp(self):
        self.fails_lots_of_SW = False
        self.times_lots_of_SW = []
        self.fails_BL_far_from_MP = False
        self.times_BL_far_from_MP = []

    def tearDown(self):
        if self.fails_lots_of_SW:
            self.plot(self.times_lots_of_SW)
        if self.fails_BL_far_from_MP:
            self.plot(self.times_BL_far_from_MP)

    @data((dataset, win_length, win_duration))
    @unpack
    def test_msp_msh_windows_have_little_sw(self, df, win_length, win_duration):
        time_indices = df[df.encountersMSPandMSH.values].index.values
        for t in time_indices:
            nbr_SW = get_window_features(df, t, win_duration, 'isSW').sum()

            try:
                self.assertLess(nbr_SW, win_length * 0.8)
            except Exception:
                self.times_lots_of_SW += [t]

        nb_fails = len(self.times_lots_of_SW)
        if nb_fails > 0:
            print(f'test_msp_msh_windows_have_little_sw failed {nb_fails} times.')
            self.fails_lots_of_SW = True
            self.data = df

    @data(dataset)
    def test_bl_points_are_close_to_MP(self, df):
        subdf = df[df.label.values.astype(bool)]
        time_indices = subdf.index.values
        relative_dist_to_mp = abs(subdf.r_mp.values - subdf.R.values) / subdf.r_mp.values
        for t, val in zip(time_indices, relative_dist_to_mp):
            try:
                self.assertLess(val, 0.4)
            except Exception:
                self.times_BL_far_from_MP += [t]

        nb_fails = len(self.times_BL_far_from_MP)
        if nb_fails > 0:
            print(f'test_bl_points_are_close_to_MP failed {nb_fails} times.')
            self.fails_BL_far_from_MP = True
            self.data = df
'''


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
