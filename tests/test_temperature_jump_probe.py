import unittest
from unittest.mock import Mock, call, patch

from tds_control import siglent
from tds_control.tds_experiment import (
    CONTROL_DEFAULTS,
    ExperimentSafetyError,
    TemperatureJumpProbe,
    _advance_temperature_jump_probe,
    _confirmed_downward_temperature_jump,
    _confirmed_upward_temperature_jump,
    _temperature_jump_probe_eligible,
)


def _config(**overrides):
    config = dict(CONTROL_DEFAULTS)
    config.update(
        max_current=0.1,
        max_voltage=1.0,
        min_voltage=0.0,
        measurement_voltage_floor=0.01,
    )
    config.update(overrides)
    return config


class TemperatureJumpProbeTests(unittest.TestCase):
    def test_first_control_sample_with_no_previous_resistance_is_not_a_type_error(self):
        config = _config()

        upward = _confirmed_upward_temperature_jump(
            temperature=25.0,
            previous_temperature=23.0,
            measured_resistance=20.0,
            previous_resistance=None,
            measured_current=0.001,
            applied_voltage=0.01,
            resistance_confirmed=True,
            setpoint=40.0,
            config=config,
        )
        downward = _confirmed_downward_temperature_jump(
            temperature=21.0,
            previous_temperature=23.0,
            measured_resistance=19.0,
            previous_resistance=None,
            measured_current=0.001,
            applied_voltage=0.01,
            resistance_confirmed=True,
            setpoint=40.0,
            config=config,
        )

        self.assertFalse(upward)
        self.assertFalse(downward)

    def test_large_downward_jump_uses_small_increase_then_accepts_consensus(self):
        config = _config()
        probe = TemperatureJumpProbe()

        self.assertTrue(
            _temperature_jump_probe_eligible(
                "down", 220.37, 358.92, 7.9877, 11.6676, 0.0342, 0.2778, True, config
            )
        )
        accepted, requested_voltage, attempts = _advance_temperature_jump_probe(
            probe, "down", 220.37, 7.9877, 0.2778, 0.0342, config
        )
        self.assertFalse(accepted)
        self.assertAlmostEqual(requested_voltage, 0.2798)
        self.assertEqual(attempts, 1)

        accepted, requested_voltage, attempts = _advance_temperature_jump_probe(
            probe, "down", 217.89, 8.0000, 0.2798, 0.0340, config
        )
        self.assertTrue(accepted)
        self.assertIsNone(requested_voltage)
        self.assertEqual(attempts, 2)
        self.assertFalse(probe.active)

    def test_large_upward_jump_uses_small_decrease_then_accepts_consensus(self):
        config = _config()
        probe = TemperatureJumpProbe()

        self.assertTrue(
            _temperature_jump_probe_eligible(
                "up", 288.02, 246.31, 9.8984, 8.8081, 0.0368, 0.3678, True, config
            )
        )
        accepted, requested_voltage, _ = _advance_temperature_jump_probe(
            probe, "up", 288.02, 9.8984, 0.3678, 0.0368, config
        )
        self.assertFalse(accepted)
        self.assertAlmostEqual(requested_voltage, 0.3658)

        accepted, requested_voltage, attempts = _advance_temperature_jump_probe(
            probe, "up", 293.03, 10.0747, 0.3658, 0.0351, config
        )
        self.assertTrue(accepted)
        self.assertIsNone(requested_voltage)
        self.assertEqual(attempts, 2)

    def test_temperature_window_stays_centered_on_first_probe_candidate(self):
        config = _config(
            measurement_jump_probe_temperature_tolerance_c=50.0,
            measurement_jump_probe_resistance_ratio=1.0,
        )
        probe = TemperatureJumpProbe()

        accepted, _, _ = _advance_temperature_jump_probe(
            probe, "down", 220.0, 8.0, 0.300, 0.03, config
        )
        self.assertFalse(accepted)
        self.assertEqual(probe.candidate_temperature, 220.0)

        accepted, _, _ = _advance_temperature_jump_probe(
            probe, "down", 275.0, 8.1, 0.302, 0.03, config
        )
        self.assertFalse(accepted)
        self.assertEqual(probe.candidate_temperature, 220.0)
        self.assertEqual(probe.confirmations, 0)

        accepted, _, _ = _advance_temperature_jump_probe(
            probe, "down", 260.0, 8.2, 0.304, 0.03, config
        )
        self.assertFalse(accepted)
        self.assertEqual(probe.confirmations, 1)

        accepted, requested_voltage, attempts = _advance_temperature_jump_probe(
            probe, "down", 255.0, 8.15, 0.306, 0.03, config
        )
        self.assertTrue(accepted)
        self.assertIsNone(requested_voltage)
        self.assertEqual(attempts, 4)

    def test_downward_probe_holds_when_current_is_near_limit(self):
        config = _config()
        probe = TemperatureJumpProbe()
        accepted, requested_voltage, _ = _advance_temperature_jump_probe(
            probe, "down", 220.0, 8.0, 0.3, 0.099, config
        )
        self.assertFalse(accepted)
        self.assertAlmostEqual(requested_voltage, 0.3)

    def test_unstable_probe_stops_with_specific_safety_error(self):
        config = _config(measurement_jump_probe_max_samples=3)
        probe = TemperatureJumpProbe()
        _advance_temperature_jump_probe(probe, "down", 220.0, 8.0, 0.300, 0.03, config)
        _advance_temperature_jump_probe(probe, "down", 200.0, 7.0, 0.302, 0.03, config)

        with self.assertRaisesRegex(ExperimentSafetyError, "did not stabilize"):
            _advance_temperature_jump_probe(probe, "down", 180.0, 6.0, 0.304, 0.03, config)

    def test_voltage_updates_do_not_reassert_output(self):
        ps = Mock()
        with patch.object(siglent.time, "sleep", return_value=None):
            siglent.set_voltage(ps, 0.2)
            self.assertEqual(ps.write.call_args_list, [call("VOLT 0.2")])

            siglent.set_output(ps, "ON")
            self.assertEqual(
                ps.write.call_args_list,
                [call("VOLT 0.2"), call("OUTP CH1,ON")],
            )


if __name__ == "__main__":
    unittest.main()
