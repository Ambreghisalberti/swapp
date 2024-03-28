from ddt import ddt, data, unpack
import sys
import unittest
from swapp.windowing.make_windows.utils import *
import pandas as pd
from swapp.windowing.make_windows import prepare_df
from swapp.windowing.make_windows.utils import time_resolution
import os
__HERE__ = os.path.abspath(__file__)

# path = f'{__HERE__}/../swapp/windowing/data/
path = '/home/ghisalberti/make_datasets/'

all_data = pd.read_pickle(path + 'MMS1_data_GSM_5S_2015_2021.pkl')
position = pd.read_pickle(path + 'MMS1_pos_GSM_5S_2015_2021.pkl')
omni = pd.read_pickle(path + 'OMNI_data_5S_2015_2021.pkl')

with open(path + 'list_label_catalogues.ts') as file:
    paths = file.read().splitlines()

with open(path + 'list_labelled_days.ts') as file:
    labelled_days = file.read().splitlines()

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
              'isLabelled', 'containsLabelledBL')
        def test_characteristics_multiples_of_winlength(self, column):
            self.assertEqual(self.data[column].sum() % self.win_length, 0)


        def make_windows(self, condition):
            windows = select_windows(self.data, condition)
            starts = windows.index.values[::self.win_length]
            stops = windows.index.values[self.win_length - 1::self.win_length]
            return starts, stops


        @data('isFull', 'isEmpty', 'isPartial', 'encountersMSPandMSH', 'isCloseToMP',
              'isLabelled', 'containsLabelledBL')
        def test_windows_duration(self, column):
            starts, stops = self.make_windows(column)
            self.assertEqual(len(starts), len(stops))
            self.assertEqual((stops-starts != win_duration - self.resolution).sum(), 0)


        @data('isFull', 'isEmpty', 'isPartial', 'encountersMSPandMSH', 'isCloseToMP',
              'isLabelled', 'containsLabelledBL')
        def test_windows_size(self, column):
            starts, stops = self.make_windows(column)
            for start, stop in zip(starts, stops):
                win = self.data[start : stop]
                self.assertEqual(len(win), self.win_length)


        @data('isFull', 'isEmpty', 'isPartial', 'encountersMSPandMSH', 'isCloseToMP',
              'isLabelled', 'containsLabelledBL')
        def test_windows_have_same_characteristics(self, column):
            starts, stops = self.make_windows(column)
            for t in range(len(starts)):
                win = get_window_features(self.data, t, win_duration).values
                values = np.unique(win)
                self.assertEqual(len(values), 0)


        def check_number_present_data(self, starts, size):
            for t in range(len(starts)):
                win = get_window_features(self.data, t, win_duration,
                                          ['Bx', 'By', 'Bz', 'Np', 'Vx', 'Vy', 'Vz', 'Tp']).values
                self.assertEqual(len(win.dropna()), size)

        def test_empty_windows_are_empty(self):
            starts, _ = self.make_windows('isEmpty')
            self.check_number_present_data(self, starts, 0)

        def test_full_windows_are_full(self):
            starts, _ = self.make_windows('isFull')
            self.check_number_present_data(self, starts, self.win_length)

        def test_partial_windows_are_partial(self):
            starts, _ = self.make_windows('isPartial')
            for t in range(len(starts)):
                win = get_window_features(self.data, t, win_duration,
                                          ['Bx', 'By', 'Bz', 'Np', 'Vx', 'Vy', 'Vz', 'Tp']).values
                self.assertLess(len(win.dropna()), self.win_length)
                self.assertGreaterEqual(len(win.dropna()), 0)


        def test_complementarity_full_empty_partial(self):
            self.assertEqual(self.data.isFull.sum() + self.data.isEmpty.sum() + self.data.isPartial.sum(),
                             len(self.data))
            self.assertEqual((self.data.isEmpty.values.astype(int) + self.data.isFull.values.astype(int)
                 + self.data.isPartial.values.astype(int) > 1).sum(),0)


        def test_regions_are_complementary(self):
            nbr_regions = (self.data.isMSP.values.astype(int) + self.data.isMSH.values.astype(int)
                           + self.data.isSW.values.astype(int))
            self.assertEqual((nbr_regions != 1).sum(), 0)


        def test_encountering_MSP_and_MSH(self):
            windows = select_windows(self.data, 'encountersMSPandMSH')
            nbr_regions = windows.isMSP.values.astype(int) + windows.isMSH.values.astype(int)
            self.assertEqual((nbr_regions != 1).sum(), 0)
            self.assertGreater(windows.isMSP.sum(), 0)
            self.assertGreater(windows.isMSH.sum(), 0)


        def test_BL_points_are_labelled(self):
            windows = select_windows(self.data, 'label')
            self.assertEqual(windows.isLabelled.sum(), len(windows))

        def test_windows_with_BL_points_contain_BL(self):
            contains_BL = self.data[self.data.nbrLabelledBL.values > 0]
            self.assertEqual((contains_BL.containsLabelledBL.values == False).sum(), 0)

        def test_windows_containing_BL_have_BL_points(self):
            contains_BL = select_windows(self.data, 'containsLabelledBL')
            nbr_BL_points = contains_BL.nbrLabelledBL.values
            self.assertEqual((nbr_BL_points < 1).sum(), 0)

        def test_isRegion_really_is_region(self, df):
            values, counts = np.unique(self.data.regions_pred.values, return_counts=True)
            is_regions = ['isMSP', 'isMSH', 'isSW']
            for i, isRegion in enumerate(is_regions):
                self.assertEqual(counts[i], self.data[isRegion].sum())

        def test_msp_msh_windows_have_little_sw(self):
            starts, stops = self.make_windows('encountersMSPandMSH')
            times_lots_of_SW = []
            for start in starts:
                nbr_SW = get_window_features(self.data, start, win_duration, 'isSW').sum()
                try:
                    self.assertLess(nbr_SW, self.win_length * 0.8)
                except Exception:
                    times_lots_of_SW += [start]

            nb_fails = len(times_lots_of_SW)
            if nb_fails > 0:
                print(f'test_msp_msh_windows_have_little_sw failed {nb_fails} times.')
                # Plot TODO


        @data(position)
        def test_BL_points_are_close_to_MP(self, pos):
            time_indices = all_data[all_data.label.values.astype(bool)].index.values
            subpos = pos.loc[time_indices]
            relative_dist_to_mp = abs(subpos.r_mp.values - subpos.R.values) / subpos.r_mp.values
            times_BL_far_from_MP = []
            for t, val in zip(time_indices, relative_dist_to_mp):
                try:
                    self.assertLess(val, 0.4)
                except Exception:
                    times_BL_far_from_MP += [t]

            nb_fails = len(times_BL_far_from_MP)
            if nb_fails > 0:
                print(f'test_bl_points_are_close_to_MP failed {nb_fails} times.')
                # Plot TODO



    return TestOriginalData

module_obj = sys.modules[__name__]

for i, win_duration in enumerate(win_durations):
    module_obj.__dict__[f"Test_{i}"] = generate_test_class(all_data, position, omni, win_duration, paths, labelled_days)

unittest.main(argv=[''], exit=False)
