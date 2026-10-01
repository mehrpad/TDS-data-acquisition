import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from tds_control import tds_experiment as ctl
from tds_control.data_saver import ExperimentDataSaver
from tds_control.pid import PIDController


class LowTcrModel:
    x = np.array([89.05, 92.05])
    temperature_bounds = (0., 600.)

    def __call__(self, resistance):
        return 50. + (float(resistance)-89.3)*200.


def settings(**kwargs):
    return ctl.build_control_config({"experiment_frequency": .5, "measurement_pair_samples": 5,
        "dmm_staged_ranging_enabled": False, "measurement_current_floor": .002,
        "max_current": .1, "max_power_w": .5, "max_temperature_c": 600., **kwargs})


class NiCr50NoiseTests(unittest.TestCase):
    def measure(self, resistances, currents=None, **kwargs):
        instrument = Mock()
        instrument.is_overload_reading.return_value = False
        currents = currents or [.0035]*len(resistances)
        instrument.read_DMM_pair.side_effect = [(r*i, i) for r, i in zip(resistances, currents)]
        config = settings(**kwargs)
        result = ctl.measure_resistivity(Mock(), Mock(), instrument, LowTcrModel(), config=config)
        return result, config

    def test_noise_is_averaged_before_low_tcr_conversion_and_ratio_remains_coherent(self):
        resistances = [89.2, 89.42, 89.22, 89.36, 89.3]
        result, config = self.measure(resistances)
        v, i, t, r = result
        self.assertAlmostEqual(r, np.mean(resistances))
        self.assertAlmostEqual(v/i, r)
        self.assertAlmostEqual(t, 50.)
        batch = config["_last_acquisition"]["pair_batch"]
        self.assertEqual(batch["retained_pairs"], 5)
        self.assertGreater(np.ptp([sample["temperature_c"] for sample in batch["samples"]]), 40.)

    def test_outlier_is_excluded_and_cold_end_noise_does_not_veto_batch(self):
        result, config = self.measure([89.0, 89.3, 89.301, 89.299, 89.302])
        self.assertAlmostEqual(result[3], 89.3005)
        self.assertEqual(config["_last_acquisition"]["pair_batch"]["retained_pairs"], 4)
        self.assertFalse(config["_last_acquisition"]["pair_batch"]["samples"][0]["retained"])

    def test_batch_with_too_few_valid_pairs_remains_invalid(self):
        result, config = self.measure([np.nan, np.nan, np.nan, 89.3, 89.3])
        self.assertTrue(all(np.isnan(value) for value in result))
        self.assertEqual(config["_last_acquisition"]["pair_batch"]["retained_pairs"], 0)

    def test_each_raw_pair_can_trip_temperature_current_and_power_guards(self):
        cases = [("temperature", [89.3, 92.1, 89.3, 89.3, 89.3], [.0035]*5, {}),
                 ("current", [89.3]*5, [.0035, .11, .0035, .0035, .0035], {}),
                 ("power", [89.3]*5, [.0035, .0035, .08, .0035, .0035], {})]
        for label, resistances, currents, overrides in cases:
            with self.subTest(label=label), self.assertRaises(ctl.ExperimentSafetyError):
                self.measure(resistances, currents, **overrides)

    def test_duty_cycled_mode_is_not_silently_multiplied(self):
        config = settings(resistivity_mode="FOUR_WIRE")
        with patch.object(ctl, "_measure_resistivity_once", return_value=(.3, .0035, 50., 89.3)) as single:
            ctl.measure_resistivity(Mock(), Mock(), Mock(), LowTcrModel(), config=config)
        single.assert_called_once()

    def test_t0_does_not_reserve_a_heating_step_on_the_two_ma_range(self):
        config = settings(measurement_pair_samples=1, dmm_staged_ranging_enabled=True,
                          dmm_current_range_a=.002, dmm_voltage_range_v=.2,
                          dmm_range_switch_fraction=.95)
        with patch.object(ctl.siglent, "read_DMM_pair", return_value=(.137, .00154)), \
             patch.object(ctl.siglent, "configure_dc_range") as change:
            ctl.measure_resistivity(Mock(), Mock(), ctl.siglent, LowTcrModel(), calibration=True, config=config)
        change.assert_not_called()

    def test_t0_overload_recovery_is_still_available(self):
        config = settings(measurement_pair_samples=1, dmm_staged_ranging_enabled=True,
                          dmm_current_range_a=.002, dmm_voltage_range_v=.2,
                          dmm_range_switch_fraction=.95,
                          dmm_range_settle_time_s=0., dmm_range_discard_readings=0)
        with patch.object(ctl.siglent, "read_DMM_pair", side_effect=[(.137, 9.9e37), (.137, .00154)]), \
             patch.object(ctl.siglent, "configure_dc_range") as change, \
             patch.object(ctl.siglent, "set_mode_speed"):
            result = ctl.measure_resistivity(Mock(), Mock(), ctl.siglent, LowTcrModel(), calibration=True, config=config)
        self.assertEqual(result[1], .00154)
        self.assertEqual(config["_active_dmm_curr_range"], .02)
        change.assert_called_once()

    def test_sample_count_and_hysteresis_bounds_are_validated(self):
        for field in ("measurement_pair_samples", "t0_pair_samples"):
            for value in (0, 16, 2.5, float("nan"), float("inf")):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    settings(**{field: value})
        for value in (-.0001, .0006, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                settings(current_quantization_hysteresis_a=value)

    def test_hysteresis_reduces_boundary_chatter_but_allows_requested_step(self):
        config = settings(current_quantization_hysteresis_a=.0002)
        controller = PIDController(0., 0., 0., 50.)
        current = .003
        with patch.object(ctl.siglent, "set_current") as commands:
            for request in (.00349, .00351, .00348, .00352, .00369):
                current = ctl._apply_control_current(Mock(), request, current, config, controller, 2.)
                self.assertEqual(current, .003)
            commands.assert_not_called()
            self.assertEqual(ctl._apply_control_current(Mock(), .00371, current, config, controller, 2.), .004)
            commands.assert_called_once()

    def test_integral_can_cross_hysteresis_without_being_unwound_and_backoff_bypasses_it(self):
        config = settings(current_quantization_hysteresis_a=.0002)
        controller = PIDController(0., .00001, 0., 50.)
        current = .003
        for _ in range(4):
            request = controller.compute(40., dt=2., bias=.003)
            current = ctl._apply_control_current(Mock(), request, current, config, controller, 2.)
        self.assertEqual(current, .004)
        with patch.object(ctl.siglent, "set_current") as command:
            backed_off = ctl._invalid_feedback_current(current, config, 2.)
            self.assertEqual(ctl._set_current_if_needed(Mock(), backed_off, current, config, force=True), .003)
            command.assert_called_once()

    def test_hysteresis_cannot_keep_command_outside_new_bounds(self):
        config = settings(current_quantization_hysteresis_a=.0002, max_current=.003)
        controller = PIDController(0., 0., 0., 50.)
        self.assertEqual(ctl._apply_control_current(Mock(), .0039, .004, config, controller, 2.), .003)

    def test_terminal_error_saved_without_inserting_extra_diagnostic_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            saver = ExperimentDataSaver(directory, np.array([[1., 2.], [23., 100.]]))
            saver.save_outcome("error", ctl.ExperimentSafetyError("Temperature feedback remained invalid"))
            saved = json.loads((Path(directory)/"run_outcome.json").read_text())
            self.assertEqual(saved["status"], "error")
            self.assertEqual(saved["error_type"], "ExperimentSafetyError")
            self.assertEqual(saved["error"], "Temperature feedback remained invalid")
            self.assertFalse((Path(directory)/"control_diagnostics.jsonl").exists())
