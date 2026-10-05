import json
import unittest
from pathlib import Path
from unittest.mock import Mock

from tds_control import tds_experiment as ctl
from tds_control.pid import PIDController


PROFILE = Path(__file__).resolve().parents[1] / 'files/material_profiles/NiCr_50.json'


class ParallelDividerTests(unittest.TestCase):
    def setUp(self):
        self.config = ctl.build_control_config(json.loads(PROFILE.read_text()))

    def test_resistance_and_power_use_wire_branch_not_total_psu_current(self):
        instrument = Mock()
        instrument.read_DMM_pair.return_value = (.009, .0001)
        instrument.is_overload_reading.return_value = False
        self.config.update(measurement_pair_samples=1, dmm_staged_ranging_enabled=False)
        result = ctl.measure_resistivity(Mock(), Mock(), instrument, lambda r: 23., config=self.config)
        self.assertAlmostEqual(result[3], 90.)
        self.assertAlmostEqual(ctl._sample_power_w(result[0], result[1]), .0000009)
        self.assertTrue(ctl._is_valid_measurement(*result[:3], self.config))

    def test_lower_wire_cutoff_applies_before_psu_command_ceiling(self):
        ctl._enforce_electrical_safety(2.25, .025, self.config)
        with self.assertRaisesRegex(ctl.ExperimentSafetyError, 'wire current limit'):
            ctl._enforce_electrical_safety(2.79, .031, self.config)
        self.assertEqual(ctl._maximum_wire_current(self.config), .03)
        self.assertEqual(self.config['max_current'], .30)

    def test_wire_current_guard_blocks_total_current_increase(self):
        controller = PIDController(.001, 0., 0., 300.)
        self.config.update(pid_kp=.001, pid_ki=0., pid_integral_time_s=0.,
                           max_current_step_up=.01)
        command = ctl._compute_next_current(controller, 290., 300., .15, .029,
                                            600., None, 10., self.config, 2.)
        self.assertLessEqual(command, .15)
        self.assertTrue(controller.current_limit_active)
        with self.assertRaises(ctl.ExperimentSafetyError):
            ctl._compute_next_current(controller, 290., 300., .20, .031,
                                      600., None, 10., self.config, 2.)

    def test_headroom_uses_wire_fraction_of_total_psu_step(self):
        self.assertAlmostEqual(ctl._wire_current_step_margin(.009, .0001, self.config), .0001)
        direct = ctl.build_control_config({})
        self.assertEqual(ctl._wire_current_step_margin(.009, .0001, direct), direct['max_current_step_up'])

    def test_direct_profiles_keep_previous_limits_and_divider_requires_wire_cutoff(self):
        direct = ctl.build_control_config({'max_current': .10})
        self.assertEqual(ctl._maximum_wire_current(direct), .10)
        for overrides in ({'max_wire_current_a': -.01}, {'max_wire_current_a': float('nan')},
                          {'parallel_resistance_ohm': -10}, {'parallel_resistance_ohm': 10},
                          {'parallel_resistance_ohm': float('inf')}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                ctl.build_control_config(overrides)

    def test_cold_command_stays_at_one_ma_total_and_divider_rating_has_headroom(self):
        self.assertEqual(ctl.current_feedforward_for_temperature(self.config, 23.), .001)
        self.assertAlmostEqual(self.config['current_feedforward_table'][0]['estimated_wire_current_a'], .0001)
        info = self.config['current_feedforward_provenance']['parallel_divider']
        self.assertLess(self.config['compliance_voltage']**2 / self.config['parallel_resistance_ohm'],
                        info['resistor_minimum_power_rating_w'])
        self.assertAlmostEqual(self.config['pid_kp'], .00015)
        self.assertAlmostEqual(self.config['pid_ki'], .000001)
        self.assertEqual(self.config['t0_calibration_current'], .001)


if __name__ == '__main__':
    unittest.main()
