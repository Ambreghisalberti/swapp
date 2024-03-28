from ddt import ddt, data, unpack
import unittest
from swapp.windowing.make_windows.utils import *
import pandas as pd
from swapp.windowing.make_windows import prepare_df
from swapp.windowing.make_windows.utils import time_resolution

all_data = pd.read_pickle('/windowing/data/MMS1_data_GSM_5S_2015_2021.pkl')
position = pd.read_pickle('/windowing/data/MMS1_pos_GSM_5S_2015_2021.pkl')
omni = pd.read_pickle('/windowing/data/OMNI_data_5S_2015_2021.pkl')

with open('/windowing/data/data/list_label_catalogues.ts') as file:
    paths = file.read().splitlines()

win_durations = [np.timedelta64(8, 'h'), np.timedelta64(4, 'h'),
                 np.timedelta64(2, 'h'), np.timedelta64(30, 'm'),
                 np.timedelta64(10, 'm')]


@ddt
class TestOriginalData(unittest.TestCase):

    @data((position,omni),(position,all_data),(all_data,omni))
    @unpack
    def have_same_index(self, df1, df2):
        self.assertEqual(len(df1), len(df2))
        times1 = df1.index.values
        times2 = df2.index.values
        for i in range(len(times1)):
            self.assertEqual(times1[i], times2[i])

    @data((all_data, position))
    @unpack
    def test_more_nans_in_data_than_pos(self, df, pos):
        self.assertLessEqual(len(df.dropna()), len(pos.dropna()))


def generate_test_class(all_data, pos, omni, win_duration, paths, labelled_days, **kwargs):
    class TestOriginalData(unittest.TestCase):

        @classmethod
        def setUpClass(self):
            self.resolution = time_resolution(all_data)
            self.win_length = time_resolution(win_duration, self.resolution)
            self.data, self.pos, self.omni = prepare_df(all_data, pos, omni, win_duration, paths, labelled_days)


        @data('isFull', 'isEmpty', 'isPartial', 'encountersMSPandMSH', 'isCloseToMP',
              'isLabelled')
        def test_characteristics_multiples_of_winlength(self, column):
            self.assertEqual(self.data[column].sum() % self.win_length, 0)

        def size_windows(self, win, win_length, win_duration):
            self.assertEqual(len(win), win_length)
            self.assertEqual(win.index.values[-1]-win.index.values[0], win_duration - self.resolution)


        @data('isFull', 'isEmpty', 'isPartial', 'encountersMSPandMSH', 'isCloseToMP',
              'isLabelled')
        def test_windows_duration(self, column):
            windows = select_windows(self.data,column)
            starts = windows.index.values[::self.win_length]
            stops = windows.index.values[self.win_length - 1::self.win_length]
            self.assertEqual(len(starts), len(stops))
            self.assertEqual((stops-starts != win_duration - self.resolution).sum(), 0)

        @data('isFull', 'isEmpty', 'isPartial', 'encountersMSPandMSH', 'isCloseToMP',
              'isLabelled')
        def test_windows_size(self, column):
            windows = select_windows(self.data,column)
            starts = windows.index.values[::self.win_length]
            for start in starts:
                win = self.data[start : start + win_duration]
                self.assertEqual(len(win), self.win_length)

        @data('isFull', 'isEmpty', 'isPartial', 'encountersMSPandMSH', 'isCloseToMP',
              'isLabelled')
        def test_windows_have_same_characteristics(self, column):
            windows = select_windows(self.data, column)
            starts = windows.index.values[::self.win_length]
            for t in range(len(starts)):
                win = get_window_features(self.data, t, win_duration).values
                values = np.unique(win)
                self.assertEqual(len(values), 0)




    return TestOriginalData


for s in [10, len(all_data)]:
    Test = returnTest(s)
    unittest.main(argv=[''], exit=False)





'''


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
