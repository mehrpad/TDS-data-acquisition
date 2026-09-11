import unittest
from unittest.mock import Mock, call, patch

import numpy as np

from tds_control import siglent
from tds_control.calibration import _calibrated_temperature_spread, _resistance_series_is_stable
from tds_control.tds_experiment import (
    CONTROL_DEFAULTS,
    ExperimentSafetyError,
    LowSignalTemperatureConfirmation,
    LowSignalCurrentRecovery,
    TemperatureProgram,
    _advance_low_signal_current_recovery,
    _enforce_electrical_safety,
    _limit_current_slew,
    _measurement_current_floor,
    _screen_low_signal_temperature,
    _sample_power_w,
    _shutdown_instruments,
    _start_control_at_initial_current,
    _current_ramp_command,
    get_experiment_mode,
    measure_resistivity,
)


def _config(**overrides):
    config = dict(CONTROL_DEFAULTS)
    config.update(
        min_current=0.0,
        max_current=1.0,
        startup_current=0.01,
        t0_current_search_start=0.01,
        measurement_current_floor=0.01,
    )
    config.update(overrides)
    return config


class LinearTemperatureModel:
    def __init__(self, slope=100.0, origin_resistance=19.5, origin_temperature=23.0):
        self.x = np.array([10.0, 30.0])
        self.slope = slope
        self.origin_resistance = origin_resistance
        self.origin_temperature = origin_temperature

    def __call__(self, resistance):
        return self.origin_temperature + self.slope * (float(resistance) - self.origin_resistance)


class LowVoltageStartupTests(unittest.TestCase):
    def test_t0_initial_voltage_is_always_part_of_experiment_floor(self):
        config = _config(startup_current=0.01, t0_current_search_start=0.02)
        self.assertAlmostEqual(_measurement_current_floor(config), 0.02)

    def test_current_slew_uses_a_single_step_limit(self):
        config = _config(t0_current_search_start=0.01, max_current_step_up=0.001, max_current_step_down=0.001)
        self.assertAlmostEqual(_limit_current_slew(0.03, 0.02, 0.01, 1.0, config), 0.021)
        self.assertAlmostEqual(_limit_current_slew(0.01, 0.02, 0.01, 1.0, config), 0.019)

    @patch("tds_control.tds_experiment.time.sleep")
    @patch("tds_control.tds_experiment.siglent.set_current")
    def test_experiment_starts_directly_at_initial_voltage_without_search(self, set_current, sleep):
        power_supply = Mock()
        config = _config(t0_current_search_start=0.02, startup_settle_time_s=1.0)

        voltage, previous_current = _start_control_at_initial_current(
            power_supply,
            config,
            previous_current=None,
            loop_time=1.0,
        )

        self.assertAlmostEqual(voltage, 0.02)
        self.assertAlmostEqual(previous_current, 0.02)
        set_current.assert_called_once_with(power_supply, current=0.02)
        sleep.assert_called_once_with(1.0)

    def test_target_keeps_advancing_by_ramp_speed_even_when_measurement_trails(self):
        program = TemperatureProgram(
            start_T=40.0,
            step_T=500.0,
            target_T=500.0,
            ramp_speed_min=60.0,
            hold_step_time_min=1.0,
            temperature_tolerance_c=2.0,
            hold_entry_tolerance_c=3.0,
        )
        program.initialize(23.0)

        targets = [program.update(measured_temperature=-100.0, dt=1.0)[0] for _ in range(20)]

        self.assertTrue(all(later >= earlier for earlier, later in zip(targets, targets[1:])))
        self.assertAlmostEqual(targets[-1], 43.0)
        self.assertEqual(program.phase, "final_ramp")

    def test_mode_names_and_legacy_values_are_normalized(self):
        self.assertEqual(get_experiment_mode({"experiment_mode": "TEMPERATURE"}), "TEMPERATURE")
        self.assertEqual(get_experiment_mode({"experiment_mode": "VOLTAGE"}), "CURRENT")
        self.assertEqual(get_experiment_mode({"experiment_mode": "CONTROLLED"}), "TEMPERATURE")
        self.assertEqual(get_experiment_mode({"experiment_mode": "CURVE_SWEEP"}), "CURRENT")

    def test_voltage_ramp_uses_volts_per_minute_and_normal_slew_limits(self):
        config = _config(max_current_step_up=0.001, max_current_step_down=0.001)
        command = _current_ramp_command(
            start_current=0.01,
            ramp_speed_min=0.001,
            elapsed_s=60.0,
            applied_current=0.01,
            config=config,
        )
        self.assertAlmostEqual(command, 0.011)

        slew_limited = _current_ramp_command(
            start_current=0.01,
            ramp_speed_min=60.0,
            elapsed_s=1.0,
            applied_current=0.01,
            config=config,
        )
        self.assertAlmostEqual(slew_limited, 0.011)

    def test_max_power_uses_synchronized_sample_voltage_and_current(self):
        config = _config(max_current=3.0, max_power_w=2.5)
        self.assertAlmostEqual(_sample_power_w(2.0, 1.24), 2.48)
        _enforce_electrical_safety(2.0, 1.24, config)
        with self.assertRaisesRegex(ExperimentSafetyError, "sample power"):
            _enforce_electrical_safety(2.0, 1.25, config)

    def test_shared_measurement_path_stops_at_max_power(self):
        siglent_module = Mock()
        siglent_module.read_DMM_pair.return_value = (2.0, 1.25)

        with self.assertRaisesRegex(ExperimentSafetyError, "2.500000 W"):
            measure_resistivity(
                Mock(),
                Mock(),
                siglent_module,
                LinearTemperatureModel(),
                config=_config(max_current=3.0, max_power_w=2.5),
            )

    def test_single_low_signal_92_c_spike_does_not_replace_room_temperature(self):
        confirmation = LowSignalTemperatureConfirmation()

        temperature, pending, confirmed = _screen_low_signal_temperature(
            temperature=92.61,
            resistance=22.05,
            trusted_temperature=23.0,
            confirmation=confirmation,
            config=_config(),
        )

        self.assertTrue(np.isnan(temperature))
        self.assertTrue(pending)
        self.assertFalse(confirmed)
        self.assertEqual(confirmation.confirmations, 1)

    def test_three_consistent_low_signal_samples_can_replace_trusted_temperature(self):
        confirmation = LowSignalTemperatureConfirmation()
        config = _config(
            low_signal_jump_confirm_samples=3,
            low_signal_jump_temperature_tolerance_c=10.0,
            low_signal_jump_resistance_tolerance_ohm=0.015,
        )

        results = [
            _screen_low_signal_temperature(
                temperature=temperature,
                resistance=resistance,
                trusted_temperature=23.0,
                confirmation=confirmation,
                config=config,
            )
            for temperature, resistance in ((92.0, 22.050), (95.0, 22.055), (91.0, 22.052))
        ]

        self.assertTrue(np.isnan(results[0][0]))
        self.assertTrue(np.isnan(results[1][0]))
        self.assertAlmostEqual(results[2][0], (92.0 + 95.0 + 91.0) / 3.0)
        self.assertFalse(results[2][1])
        self.assertTrue(results[2][2])

    def test_stuck_low_signal_recovery_increases_01_v_five_times(self):
        recovery = LowSignalCurrentRecovery()
        config = _config(
            low_signal_recovery_trigger_cycles=2,
            low_signal_recovery_observe_cycles=2,
            low_signal_recovery_current_step=0.01,
            low_signal_recovery_max_attempts=5,
        )
        applied_current = 0.01
        stepped_voltages = []

        for invalid_streak in range(1, 12):
            requested_voltage, stepped = _advance_low_signal_current_recovery(
                recovery=recovery,
                invalid_reuse_streak=invalid_streak,
                low_signal_state=applied_current <= config["ignore_invalid_below_current"],
                applied_current=applied_current,
                measured_current=0.001,
                config=config,
            )
            if requested_voltage is not None:
                applied_current = requested_voltage
            if stepped:
                stepped_voltages.append(applied_current)

        np.testing.assert_allclose(stepped_voltages, [0.02, 0.03, 0.04, 0.05, 0.06])
        self.assertEqual(recovery.attempts, 5)

    def test_low_signal_recovery_does_not_increase_near_current_limit(self):
        recovery = LowSignalCurrentRecovery()
        config = _config(
            low_signal_recovery_trigger_cycles=1,
            low_signal_recovery_observe_cycles=1,
            max_current=0.1,
        )

        requested_current, stepped = _advance_low_signal_current_recovery(
            recovery=recovery,
            invalid_reuse_streak=1,
            low_signal_state=True,
            applied_current=0.01,
            measured_current=0.096,
            config=config,
        )

        self.assertEqual(requested_current, 0.01)
        self.assertFalse(stepped)
        self.assertEqual(recovery.attempts, 0)

    def test_t0_stability_checks_resistance_as_well_as_current(self):
        config = _config()
        self.assertTrue(
            _resistance_series_is_stable([19.50, 19.52, 19.49], config)
        )
        self.assertFalse(
            _resistance_series_is_stable([16.50, 16.83, 17.16], config)
        )

    def test_t0_uncertainty_uses_inverse_calibration_scale(self):
        reference_temperature = lambda resistance: 100.0 * (float(resistance) - 10.0) + 23.0
        spread = _calibrated_temperature_spread(
            [20.00, 20.02],
            scale=2.0,
            reference_temperature_interp=reference_temperature,
        )
        self.assertAlmostEqual(spread, 1.0)

    def test_explicit_fixed_dmm_ranges_override_broad_safety_limits(self):
        dmm = Mock()
        config = _config(
            max_current=3.0,
            dmm_voltage_range_v=0.2,
            dmm_current_range_a=0.2,
        )

        siglent.configure_dc_range_from_config(dmm, "VOLT", config)
        siglent.configure_dc_range_from_config(dmm, "CURR", config)

        self.assertEqual(
            dmm.write.call_args_list,
            [call("CONF:VOLT:DC 0.2"), call("CONF:CURR:DC 0.2")],
        )

    def test_current_range_escalates_early_on_a_tight_range(self):
        """A starting range close to the per-loop step must escalate before overload.

        On the 0.02 A current range with the default 0.01 A step, waiting for
        the flat 80%-of-range threshold (0.016 A) leaves less than one step of
        headroom: the very next commanded step could land at or past 0.02 A
        before the DMM range has caught up. step_margin pulls the threshold
        down to half scale here, so escalation happens a full step early.
        Ranges much larger than the step (0.2 A, 2 A) are unaffected - the
        flat fraction already leaves ample headroom there.
        """
        dmm = Mock()
        config = _config(dmm_current_range_a=0.02)
        siglent.configure_dc_range_from_config(dmm, "CURR", config)

        # Below the tightened threshold (0.01 A): no escalation yet.
        result = siglent.increase_dc_range_if_needed(
            dmm, "CURR", 0.009, config, step_margin=0.01
        )
        self.assertIsNone(result)

        # At the tightened threshold: escalates now, a full step before 0.02 A.
        result = siglent.increase_dc_range_if_needed(
            dmm, "CURR", 0.01, config, step_margin=0.01
        )
        self.assertEqual(result, (0.02, 0.2))
        self.assertEqual(config["_active_dmm_curr_range"], 0.2)

    def test_current_range_margin_does_not_affect_a_range_much_larger_than_the_step(self):
        dmm = Mock()
        config = _config(dmm_current_range_a=2.0)
        siglent.configure_dc_range_from_config(dmm, "CURR", config)

        # 0.667 A is 33% of a 2 A range; nowhere near the flat 80% threshold,
        # and the 0.01 A step margin does not pull that threshold down here.
        result = siglent.increase_dc_range_if_needed(
            dmm, "CURR", 0.667, config, step_margin=0.01
        )
        self.assertIsNone(result)

    def test_voltage_ranging_is_unaffected_by_the_current_step_margin(self):
        dmm = Mock()
        config = _config(dmm_voltage_range_v=20.0)
        siglent.configure_dc_range_from_config(dmm, "VOLT", config)

        # 15 V is 75% of a 20 V range: below the flat 80% threshold, and VOLT
        # calls never pass step_margin, so it stays on this range.
        result = siglent.increase_dc_range_if_needed(dmm, "VOLT", 15.0, config)
        self.assertIsNone(result)

    @patch("tds_control.tds_experiment.time.sleep")
    @patch(
        "tds_control.siglent.read_DMM_pair",
        side_effect=[
            (0.170, 0.00170),
            (0.171, 0.0050),
            (0.172, 0.0050),
            (0.173, 0.0050),
        ],
    )
    def test_staged_fixed_ranges_step_up_and_discard_transition_readings(self, read_pair, sleep):
        voltage_dmm = Mock()
        current_dmm = Mock()
        config = _config(
            max_current=0.1,
            dmm_voltage_range_v=0.2,
            dmm_current_range_a=0.002,
            dmm_range_switch_fraction=0.8,
            dmm_range_settle_time_s=0.3,
            dmm_range_discard_readings=2,
        )
        siglent.configure_dc_range_from_config(voltage_dmm, "VOLT", config)
        siglent.configure_dc_range_from_config(current_dmm, "CURR", config)

        measured_voltage, measured_current, temperature, _ = measure_resistivity(
            voltage_dmm,
            current_dmm,
            siglent,
            lambda resistance: resistance,
            config=config,
        )

        self.assertAlmostEqual(measured_voltage, 0.173)
        self.assertAlmostEqual(measured_current, 0.0050)
        self.assertAlmostEqual(temperature, 34.6)
        self.assertEqual(read_pair.call_count, 4)
        sleep.assert_called_once_with(0.3)
        self.assertIn(call("CONF:VOLT:DC 2.0"), voltage_dmm.write.call_args_list)
        self.assertIn(call("CONF:CURR:DC 0.02"), current_dmm.write.call_args_list)

    def test_sdm3055_overload_responses_are_recognized(self):
        self.assertTrue(siglent.is_overload_reading("+9.90000000E+37"))
        self.assertTrue(siglent.is_overload_reading("overload"))
        self.assertFalse(siglent.is_overload_reading("0.199"))

    @patch("tds_control.tds_experiment.time.sleep")
    @patch(
        "tds_control.siglent.read_DMM_pair",
        side_effect=[
            ("+9.90000000E+37", 0.0010),
            (0.30, 0.0010),
            (0.31, 0.0010),
            ("+9.90000000E+37", 0.0010),
            (5.00, 0.0010),
            (5.01, 0.0010),
            (5.00, 0.0010),
        ],
    )
    def test_overload_recovery_can_cross_multiple_fixed_ranges(self, read_pair, sleep):
        voltage_dmm = Mock()
        current_dmm = Mock()
        config = _config(
            max_current=3.0,
            dmm_voltage_range_v=0.2,
            dmm_current_range_a=0.002,
            dmm_range_recovery_attempts=5,
        )
        siglent.configure_dc_range_from_config(voltage_dmm, "VOLT", config)
        siglent.configure_dc_range_from_config(current_dmm, "CURR", config)

        measured_voltage, measured_current, temperature, _ = measure_resistivity(
            voltage_dmm,
            current_dmm,
            siglent,
            lambda resistance: resistance,
            config=config,
        )

        self.assertAlmostEqual(measured_voltage, 5.0)
        self.assertAlmostEqual(measured_current, 0.001)
        self.assertAlmostEqual(temperature, 5000.0)
        self.assertEqual(read_pair.call_count, 7)
        self.assertEqual(sleep.call_args_list, [call(0.3), call(0.3)])
        self.assertEqual(
            voltage_dmm.write.call_args_list,
            [
                call("CONF:VOLT:DC 0.2"),
                call("CONF:VOLT:DC 2.0"),
                call("CONF:VOLT:DC 20.0"),
            ],
        )

    @patch("tds_control.tds_experiment.time.sleep")
    @patch("tds_control.siglent.read_DMM_pair", return_value=("overload", 0.0010))
    def test_overload_on_largest_range_is_rejected_without_transition_data(self, read_pair, sleep):
        voltage_dmm = Mock()
        current_dmm = Mock()
        config = _config(
            max_current=3.0,
            dmm_voltage_range_v=1000.0,
            dmm_current_range_a=0.002,
            # This test exercises the VOLT overload path only; a tiny step
            # keeps the CURR margin from also escalating the stable 0.001 A
            # reading (see test_current_range_escalates_early_on_a_tight_range).
            max_current_step_up=0.0001,
        )
        siglent.configure_dc_range_from_config(voltage_dmm, "VOLT", config)
        siglent.configure_dc_range_from_config(current_dmm, "CURR", config)

        measured_voltage, measured_current, temperature, _ = measure_resistivity(
            voltage_dmm,
            current_dmm,
            siglent,
            lambda resistance: resistance,
            config=config,
        )

        self.assertTrue(np.isnan(measured_voltage))
        self.assertAlmostEqual(measured_current, 0.001)
        self.assertTrue(np.isnan(temperature))
        read_pair.assert_called_once()
        sleep.assert_not_called()

    def test_synchronized_pair_starts_both_conversions_before_fetching(self):
        voltage_dmm = Mock()
        current_dmm = Mock()
        voltage_dmm.query.return_value = "0.0195"
        current_dmm.query.return_value = "0.0010"

        measured_voltage, measured_current = siglent.read_DMM_pair(voltage_dmm, current_dmm)

        self.assertEqual((measured_voltage, measured_current), ("0.0195", "0.0010"))
        self.assertEqual(
            voltage_dmm.method_calls,
            [call.write("TRIG:SOUR IMM"), call.write("INIT"), call.query("FETCh?")],
        )
        self.assertEqual(
            current_dmm.method_calls,
            [call.write("TRIG:SOUR IMM"), call.write("INIT"), call.query("FETCh?")],
        )

    def test_measurement_prefers_synchronized_pair_reader(self):
        siglent_module = Mock()
        siglent_module.read_DMM_pair.return_value = (0.0195, 0.0010)

        measured_voltage, measured_current, temperature, _ = measure_resistivity(
            Mock(),
            Mock(),
            siglent_module,
            LinearTemperatureModel(),
            config=_config(),
        )

        self.assertAlmostEqual(measured_voltage, 0.0195)
        self.assertAlmostEqual(measured_current, 0.0010)
        self.assertAlmostEqual(temperature, 23.0)
        siglent_module.read_DMM_pair.assert_called_once()
        siglent_module.read_DMM.assert_not_called()


class ShutdownInstrumentsTests(unittest.TestCase):
    """T0 calibration, PI/PID tuning, and every experiment type end here."""

    @patch("tds_control.tds_experiment.time.sleep")
    @patch("tds_control.siglent.set_output")
    @patch("tds_control.siglent.set_current")
    def test_shutdown_zeroes_current_then_switches_output_off(self, set_current, set_output, sleep):
        power_supply = Mock()
        dmm_v = Mock()
        dmm_i = Mock()
        resource_manager = Mock()
        call_order = []
        set_current.side_effect = lambda *a, **k: call_order.append("set_current")
        set_output.side_effect = lambda *a, **k: call_order.append("set_output")

        _shutdown_instruments(dmm_v, dmm_i, power_supply, resource_manager)

        set_current.assert_called_once_with(power_supply, current=0.0)
        set_output.assert_called_once_with(power_supply, state="OFF")
        self.assertEqual(call_order, ["set_current", "set_output"])
        dmm_v.close.assert_called_once()
        dmm_i.close.assert_called_once()
        power_supply.close.assert_called_once()
        resource_manager.close.assert_called_once()

    @patch("tds_control.tds_experiment.time.sleep")
    @patch("tds_control.siglent.set_output")
    @patch("tds_control.siglent.set_current", side_effect=RuntimeError("bus error"))
    def test_shutdown_still_switches_output_off_if_zeroing_current_fails(
        self, set_current, set_output, sleep
    ):
        power_supply = Mock()

        _shutdown_instruments(None, None, power_supply, None)

        set_output.assert_called_once_with(power_supply, state="OFF")

    def test_shutdown_tolerates_a_missing_power_supply(self):
        dmm_v = Mock()

        _shutdown_instruments(dmm_v, None, None, None)

        dmm_v.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
