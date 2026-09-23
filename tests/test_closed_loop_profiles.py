import unittest

import numpy as np
import pandas as pd

from tools.update_closed_loop_profiles import derive_table


def fixture():
    t = np.r_[np.repeat(np.arange(40., 590., 20.), 20), np.full(35, 598.)]
    command = .01 + .0003*t
    data = pd.DataFrame({'time': np.arange(len(t))*2., 'T': t, 'set_T': np.r_[t[:-35]+2., np.full(35, 600.)],
                         'I': command, 'C_V': command, 'P': command**2*5})
    diagnostics = pd.DataFrame({'time': data.time+.001, 'status': 'accepted', 'range_change_count': 0})
    return data, diagnostics


class ClosedLoopProfileTests(unittest.TestCase):
    def test_accepted_measured_temperature_bins_and_no_hold_contamination(self):
        data, diagnostics = fixture()
        table, evidence = derive_table(data, diagnostics)
        self.assertEqual(table[0], {'temperature_c': 23., 'current_a': .01})
        point = next(p for p in table if p['temperature_c'] == 200)
        self.assertAlmostEqual(point['current_a'], .07)
        self.assertEqual(evidence['late_hold']['mean_temperature_c'], 598)
        self.assertEqual(evidence['late_hold']['temperature_rate_c_min'], 0)
        self.assertFalse(evidence['unmeasured_extension']['measured'])
        self.assertGreater(table[-1]['current_a'], table[-2]['current_a'])
        self.assertEqual(table[-1]['temperature_c'], 600)
        self.assertLess(table[-1]['current_a'], .2)
        self.assertTrue(all(b['temperature_bin_c'][1] <= 600 for b in evidence['bins']))

    def test_invalid_samples_and_range_transition_cannot_change_table(self):
        data, diagnostics = fixture()
        diagnostics.loc[120, 'status'] = 'invalid_hold'
        diagnostics.loc[200:, 'range_change_count'] = 1
        expected, _ = derive_table(data, diagnostics)
        data.loc[[120, 200, 201, 202], 'C_V'] = .9
        actual, _ = derive_table(data, diagnostics)
        self.assertEqual(actual, expected)

    def test_misaligned_logs_insufficient_hold_and_large_reversal_rejected(self):
        data, diagnostics = fixture()
        with self.assertRaisesRegex(ValueError, 'align'):
            derive_table(data, diagnostics.iloc[:-1])
        with self.assertRaisesRegex(ValueError, '40 seconds'):
            derive_table(data.iloc[:-30], diagnostics.iloc[:-30])
        data.loc[data['T'] == 200, 'C_V'] = .2
        with self.assertRaisesRegex(ValueError, 'current step'):
            derive_table(data, diagnostics)

    def test_nicr_uses_same_bounded_extension_without_claiming_600_measurements(self):
        data, diagnostics = fixture()
        data = data[(data['T'] <= 460) | (data.set_T == 600)].reset_index(drop=True)
        data.loc[data.set_T == 600, ['set_T', 'T', 'I', 'C_V', 'P']] = [500, 499, .1597, .1597, .1275]
        diagnostics = pd.DataFrame({'time': data.time+.001, 'status': 'accepted', 'range_change_count': 0})
        table, evidence = derive_table(data, diagnostics, nicr=True)
        self.assertEqual(evidence['derived_temperature_range_c'][1], 499)
        self.assertEqual(evidence['unmeasured_extension']['range_c'], [499, 600])
        self.assertGreater(table[-1]['current_a'], .1597)
        self.assertLess(table[-1]['current_a'], .2)
