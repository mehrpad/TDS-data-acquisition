import json
import unittest
from pathlib import Path

from tds_control import tds_experiment as ctl
from tds_control.pid import PIDController
from tools.retune_nicr50_after_run129 import retune_profile


PROFILE = Path(__file__).resolve().parents[1]/'files/material_profiles/NiCr_50.json'


class NiCr50StallCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads(PROFILE.read_text())
        # Keep the original plateau regression reproducible after later profiles
        # replace the trial gains; these tests cover the run-129 correction.
        self.profile['current_feedforward_table'] = self.profile['current_feedforward_provenance']['run130_revision']['previous_table']
        self.profile['current_feedforward_provenance']['source_run'] = '127_test'
        self.profile.update(max_current=.18,max_wire_current_a=.03,max_power_w=.1,
                            max_sample_voltage=3.,compliance_voltage=3.)
        self.profile = retune_profile(self.profile)
        self.config = ctl.build_control_config(self.profile)

    def test_sustained_lag_continues_correction_beyond_twenty_ma_with_bounded_slew(self):
        config = self.config
        controller = PIDController(config['pid_kp'],config['pid_ki'],0.,375.,
            output_limits=(.001,config['max_current']),
            integral_limits=(-config['pid_integral_current_limit_a'],config['pid_integral_current_limit_a']))
        controller.integral = .02
        command = .125
        for _ in range(90):
            previous = command
            command = ctl._compute_next_current(controller,272.,375.,command,.02,600.,0.,10.,config,2.)
            self.assertLessEqual(command-previous,config['max_current_step_up']+1e-12)
            self.assertLessEqual(command,config['max_current'])
        self.assertGreater(controller.integral,.04)
        self.assertGreater(command,.15)

    def test_startup_gains_remain_lower_than_high_current_gains(self):
        low = ctl.pid_gains_for_current(self.config,.001)
        high = ctl.pid_gains_for_current(self.config,.1)
        self.assertAlmostEqual(low[0],.0000825)
        self.assertAlmostEqual(low[1],.00000055)
        self.assertAlmostEqual(high[0]/low[0],2.)
        self.assertAlmostEqual(high[1]/low[1],3.)

    def test_flat_bias_is_replaced_by_estimates_with_original_observations_preserved(self):
        revision = self.profile['current_feedforward_provenance']['run129_controller_revision']
        self.assertFalse(revision['run129_used_as_thermal_calibration'])
        points = [p for p in self.profile['current_feedforward_table']
                  if revision['smoothing_range_c'][0] <= p['temperature_c'] <= revision['smoothing_range_c'][1]]
        for a,b in zip(points,points[1:]):
            self.assertGreater((b['current_a']-a['current_a'])/(b['temperature_c']-a['temperature_c']),.00009)
        self.assertTrue(all(not p['measured_on_this_wire'] for p in points[1:-1]))
        self.assertEqual(len(points)-2,len(revision['original_points']))
        self.assertEqual(retune_profile(self.profile),self.profile)

    def test_full_supply_is_not_enabled_for_small_parallel_bank_and_guards_remain(self):
        config = self.config
        rating = self.profile['current_feedforward_provenance']['parallel_divider']['resistor_minimum_power_rating_w']
        self.assertLess(config['compliance_voltage']**2/config['parallel_resistance_ohm'],rating/2)
        with self.assertRaises(ctl.ExperimentSafetyError):
            ctl._enforce_electrical_safety(2.8,.031,config)
        with self.assertRaises(ctl.ExperimentSafetyError):
            ctl._enforce_electrical_safety(3.01,.02,config)
        with self.assertRaises(ctl.ExperimentSafetyError):
            ctl._enforce_temperature_safety(600.1,config)


if __name__ == '__main__':
    unittest.main()
