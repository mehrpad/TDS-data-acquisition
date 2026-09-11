import unittest
from unittest.mock import Mock, call, patch

from tds_control import siglent
from tds_control.tds_experiment import (
    CONTROL_DEFAULTS,
    TemperatureJumpProbe,
    _advance_temperature_jump_probe,
    _confirmed_downward_temperature_jump,
    _confirmed_upward_temperature_jump,
    _temperature_jump_probe_eligible,
)


def _config(**overrides):
    config = dict(CONTROL_DEFAULTS)
    config.update(
        max_current=1.0,
        min_current=0.0,
        measurement_current_floor=0.01,
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
            applied_current=0.01,
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
            applied_current=0.01,
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

    def test_jump_confirmation_is_eligible_at_currents_common_on_sensitive_wires(self):
        """Reproduces a field lockup: a real, sustained temperature rise past
        the 35 C large-jump threshold was never confirmed, because the probe
        path needed 0.1 A applied current to even start - a threshold carried
        over unchanged from the constant-voltage era, when this key was
        renamed from measurement_jump_confirm_min_voltage.

        At 0.0943 A (the applied current in the field log), eligibility must
        now pass and the probe must be able to run and eventually accept a
        stable new reading, instead of resetting and rejecting every sample
        while the trusted temperature stays frozen and the real one keeps
        climbing.
        """
        config = _config()
        self.assertTrue(
            _temperature_jump_probe_eligible(
                "up",
                temperature=142.77,
                previous_temperature=106.62,
                measured_resistance=3.4705,
                previous_resistance=3.3974,
                measured_current=0.0950,
                applied_current=0.0943,
                resistance_confirmed=True,
                config=config,
            )
        )
        probe = TemperatureJumpProbe()
        accepted, requested_current, attempts = _advance_temperature_jump_probe(
            probe, "up", 142.77, 3.4705, 0.0943, 0.0950, config
        )
        self.assertFalse(accepted)
        self.assertEqual(attempts, 1)
        accepted, requested_current, attempts = _advance_temperature_jump_probe(
            probe, "up", 143.10, 3.4720, requested_current, requested_current, config
        )
        self.assertTrue(accepted)
        self.assertIsNone(requested_current)

    def test_jump_confirmation_still_rejects_truly_low_current_readings(self):
        # Below ignore_invalid_below_current, _is_low_signal_state routes the
        # sample to the separate low-signal path instead; this check staying
        # strict there is not a regression.
        config = _config()
        self.assertFalse(
            _temperature_jump_probe_eligible(
                "up",
                temperature=40.0,
                previous_temperature=23.0,
                measured_resistance=3.0,
                previous_resistance=2.0,
                measured_current=0.01,
                applied_current=0.01,
                resistance_confirmed=True,
                config=config,
            )
        )

    def test_downward_probe_holds_when_current_is_near_limit(self):
        # max_current is both the setpoint ceiling and the safety limit, so the
        # probe must hold once the measurement reaches 95% of it.
        config = _config(max_current=0.1)
        probe = TemperatureJumpProbe()
        accepted, requested_current, _ = _advance_temperature_jump_probe(
            probe, "down", 220.0, 8.0, 0.09, 0.099, config
        )
        self.assertFalse(accepted)
        self.assertAlmostEqual(requested_current, 0.09)

    def test_unstable_probe_gives_up_instead_of_stopping_the_experiment(self):
        # Killing the whole run over one unconfirmed jump is disproportionate:
        # the general invalid-measurement handling already has its own, more
        # patient safety nets. Exhausting the probe's budget on genuinely
        # inconsistent readings should give up on this probe (so the reading
        # is treated as an ordinary invalid measurement) rather than raise.
        config = _config(measurement_jump_probe_max_samples=3)
        probe = TemperatureJumpProbe()
        _advance_temperature_jump_probe(probe, "down", 220.0, 8.0, 0.300, 0.03, config)
        _advance_temperature_jump_probe(probe, "down", 200.0, 7.0, 0.302, 0.03, config)

        accepted, requested_current, attempts = _advance_temperature_jump_probe(
            probe, "down", 180.0, 6.0, 0.304, 0.03, config
        )
        self.assertFalse(accepted)
        self.assertIsNone(requested_current)
        self.assertEqual(attempts, 3)
        self.assertFalse(probe.active)

    def test_slow_continuous_drift_confirms_via_the_sliding_reference(self):
        # Reproduces a field crash: a real, still-settling trend (long thermal
        # tau) keeps every single step small and consistent, but the total
        # drift since the very first probe sample eventually exceeds
        # tolerance. A reference fixed at that first sample would eventually
        # reject every later sample forever and exhaust the probe; sliding the
        # reference forward on each confirmed step must let it accept
        # instead, using the exact field magnitudes (R drifting smoothly
        # 3.4684 -> 3.3828 Ohm, a total change well past the 0.02 ratio
        # tolerance measured against the first sample alone).
        config = _config()
        probe = TemperatureJumpProbe()
        temperature = 171.09
        resistance = 3.4684
        applied_current = 0.0600
        accepted = False
        attempts = 0
        for _ in range(20):
            applied_current += 0.002
            temperature -= 0.1321
            resistance -= 0.00449
            accepted, requested_current, attempts = _advance_temperature_jump_probe(
                probe, "down", temperature, resistance, applied_current, 0.05, config
            )
            if accepted:
                break
        self.assertTrue(accepted)
        self.assertIsNone(requested_current)
        self.assertLess(attempts, 20)

    def test_current_updates_do_not_reassert_output(self):
        ps = Mock()
        ps.query.return_value = "0x0010"
        with patch.object(siglent.time, "sleep", return_value=None):
            siglent.set_current(ps, 0.2)
            self.assertEqual(ps.write.call_args_list, [call("CURR 0.2")])

            siglent.set_output(ps, "ON")
            self.assertEqual(
                ps.write.call_args_list,
                [call("CURR 0.2"), call("OUTP CH1,ON")],
            )

    def test_set_output_retries_with_unlock_when_the_key_lock_blocks_it(self):
        ps = Mock()
        ps.query.side_effect = ["0x0000", "0x0010"]
        with patch.object(siglent.time, "sleep", return_value=None):
            siglent.set_output(ps, "ON")
        self.assertEqual(
            ps.write.call_args_list,
            [call("OUTP CH1,ON"), call("*UNLOCK"), call("OUTP CH1,ON")],
        )

    def test_set_output_reports_an_unswitchable_output(self):
        ps = Mock()
        ps.query.return_value = "0x0000"
        with patch.object(siglent.time, "sleep", return_value=None):
            with self.assertRaises(RuntimeError) as raised:
                siglent.set_output(ps, "ON")
        self.assertIn("Ver/Lock", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
