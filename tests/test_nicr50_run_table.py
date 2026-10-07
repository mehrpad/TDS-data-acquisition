import unittest

import numpy as np
import pandas as pd

from tools.update_nicr50_run_profile import derive_table, monotonic_squared_currents


def fixture():
    temperatures = np.r_[np.full(100,23.), np.repeat(np.arange(70.,600.,20.),20)]
    applied = .001+.00025*temperatures
    data = pd.DataFrame(dict(time=np.arange(len(temperatures))*2., V=applied*15,
                             I=applied/5.5, P=(applied/5.5)**2*85.,
                             T=temperatures,set_T=temperatures-5.))
    diagnostics = pd.DataFrame(dict(time=data.time+.001,status='accepted',
        raw_temperature_c=temperatures,applied_current_a=applied,
        accepted_current_a=applied+.03,range_change_count=0))
    return data,diagnostics


class NiCr50RunTableTests(unittest.TestCase):
    def test_uses_applied_command_and_indicated_temperature_not_next_command_or_target(self):
        data,diag=fixture()
        table,evidence=derive_table(data,diag)
        point=next(p for p in table if p['temperature_c']==210.)
        self.assertAlmostEqual(point['current_a'], .001+.00025*210.)
        self.assertAlmostEqual(point['wire_current_a'], point['current_a']/5.5)
        self.assertEqual(table[0]['current_a'],.001)
        self.assertFalse(table[0]['measured_on_this_wire'])
        self.assertFalse(table[-1]['measured_on_this_wire'])
        self.assertEqual(table[-1]['temperature_c'],600.)
        self.assertFalse(evidence['measured_final_hold'])

    def test_rejected_sample_and_range_transients_do_not_change_fit(self):
        data,diag=fixture()
        diag.loc[240,'status']='invalid_backoff'
        diag.loc[300:,'range_change_count']=1
        expected,_=derive_table(data,diag)
        data.loc[[240,300,301,302],'I']=1.
        diag.loc[[240,300,301,302],'applied_current_a']=1.
        actual,_=derive_table(data,diag)
        self.assertEqual(expected,actual)

    def test_weighted_monotonic_power_fit_pools_reversal_without_raising_all_later_points(self):
        result=monotonic_squared_currents([.1,.08,.12],[1,3,1])
        self.assertAlmostEqual(result[0],np.sqrt((.1**2+3*.08**2)/4))
        self.assertEqual(result[0],result[1])
        self.assertEqual(result[2],.12)

    def test_misalignment_insufficient_coverage_and_large_reversal_are_rejected(self):
        data,diag=fixture()
        with self.assertRaisesRegex(ValueError,'align'):
            derive_table(data.iloc[:-1],diag)
        with self.assertRaisesRegex(ValueError,'coverage'):
            derive_table(data.iloc[:300],diag.iloc[:300])
        diag.loc[diag.raw_temperature_c.eq(210),'applied_current_a']=.2
        with self.assertRaisesRegex(ValueError,'reversals'):
            derive_table(data,diag)


if __name__=='__main__':
    unittest.main()
