import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from tds_control import config_io, material_profiles
from tds_control import tds_experiment as ctl


def settings(**overrides):
    return ctl.build_control_config({"experiment_frequency": .5, **overrides})


class Ni50RetryTests(unittest.TestCase):
    @patch.object(ctl.time, "sleep")
    def test_warming_run117_like_pairs_recover_from_old_reference(self, sleep):
        # Run 117's real triggering pair: 18.511 -> 18.817 Ohm. Fresh pairs
        # drift more than 15 mOhm but agree within 0.5% and 5 degrees.
        readings = [(r*.028, .028, 190+(r-18.511)*16, r)
                    for r in (18.817, 18.87, 18.92, 18.97)]
        for ratio, expected in ((0., False), (.005, True)):
            config = settings(measurement_retry_attempts=3, measurement_retry_consensus_ratio=ratio)
            with patch.object(ctl, "measure_resistivity", side_effect=readings):
                v, i, t, r, accepted = ctl._measure_with_retry(
                    Mock(), Mock(), Mock(), lambda r: 190+(r-18.511)*16,
                    config=config, previous_resistance=18.511)
            self.assertEqual(accepted, expected)
            if accepted:
                self.assertAlmostEqual(v/i, r)
                self.assertEqual(r, 18.97)
                self.assertEqual(config["_measurement_retry"]["decision"], "fresh_consensus")

    @patch.object(ctl.time, "sleep")
    def test_relative_agreement_cannot_bypass_low_tcr_temperature_guard(self, sleep):
        convert = lambda r: 23+(r-22)*1000
        readings = [(r*.02, .02, convert(r), r) for r in (22.2, 22.3, 22.35, 22.4)]
        with patch.object(ctl, "measure_resistivity", side_effect=readings):
            result = ctl._measure_with_retry(Mock(), Mock(), Mock(), convert,
                config=settings(measurement_retry_attempts=3), previous_resistance=22.)
        self.assertFalse(result[-1])

    @patch.object(ctl.time, "sleep")
    def test_erratic_retries_remain_invalid(self, sleep):
        readings = [(r*.02, .02, r*10, r) for r in (20., 19., 21., 18.)]
        with patch.object(ctl, "measure_resistivity", side_effect=readings):
            result = ctl._measure_with_retry(Mock(), Mock(), Mock(), lambda r: r*10,
                config=settings(measurement_retry_attempts=3), previous_resistance=10.)
        self.assertFalse(result[-1])

    def test_invalid_consensus_settings_are_rejected(self):
        for ratio in (-.1, .1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                settings(measurement_retry_consensus_ratio=ratio)

    def test_backoff_never_increases_current_and_respects_floor(self):
        config = settings(measurement_current_floor=.005)
        self.assertEqual(ctl._invalid_feedback_current(.028, config, 2), .027)
        self.assertEqual(ctl._invalid_feedback_current(.005, config, 2), .005)
        self.assertEqual(ctl._invalid_feedback_current(.003, config, 2), .003)

    def test_invalid_feedback_reduces_current_pauses_program_and_stops(self):
        config = settings(DMM_v="v", DMM_i="i", PS="ps", DMM_speed=10,
                          measurement_current_floor=.005, measurement_fail_limit=2,
                          minimum_current_change=.01,
                          invalid_measurement_policy="backoff")
        emitter = Mock(); emitter.stopped = False
        saver = Mock()
        clock = [0.]
        def invalid_pair(*args, **kwargs):
            clock[0] += 2.
            return (.5, .028, np.nan, 18., False)
        with patch.object(ctl.pyvisa, "ResourceManager"), patch.object(ctl, "siglent") as instrument, \
             patch.object(ctl.time, "sleep"), patch.object(ctl, "_shutdown_instruments") as shutdown, \
             patch.object(ctl.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(ctl, "_start_control_at_initial_current", return_value=(.028, .028)), \
             patch.object(ctl, "_measure_with_retry", side_effect=invalid_pair), \
             patch.object(ctl, "_compute_next_current") as compute:
            with self.assertRaisesRegex(ctl.ExperimentSafetyError, "without increasing current"):
                ctl.tds(emitter, [dict(start_T=23, step_T=600, target_T=600,
                    ramp_speed_min=10, hold_step_time_min=0)], np.array([[1., 7.], [0., 600.]]), config, 23., saver)
        compute.assert_not_called()
        self.assertEqual([call.kwargs["current"] for call in instrument.set_current.call_args_list[-2:]], [.027, .026])
        records = [call.args[0] for call in saver.enqueue_diagnostics.call_args_list]
        self.assertEqual([r["status"] for r in records], ["invalid_backoff", "invalid_backoff"])
        self.assertEqual([r["setpoint_c"] for r in records], [23., 23.])
        self.assertEqual([r["integral_a"] for r in records], [0., 0.])
        self.assertEqual([r["accepted_current_a"] for r in records], [.027, .026])
        shutdown.assert_called_once()
        status, error = saver.save_outcome.call_args.args
        self.assertEqual(status, "error")
        self.assertIsInstance(error, ctl.ExperimentSafetyError)
        self.assertIn("without increasing current", str(error))
        saver.finalize.assert_called_once()


class Ni50ProfileTests(unittest.TestCase):
    def test_profile_roundtrip_preserves_recovery_and_600_degree_settings(self):
        source = Path(__file__).resolve().parents[1]/"files/material_profiles/Ni_50_200.json"
        original = json.loads(source.read_text())
        self.assertFalse(set(original)-set(material_profiles.PROFILE_FIELDS)-{"profile_name"})
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(material_profiles, "PROFILES_DIR", Path(directory)), \
             patch.object(config_io, "CONFIG_PATH", Path(directory)/"config.toml"), \
             patch.object(config_io, "ensure_runtime_dirs"):
            material_profiles.save_profile("Ni_50_200", original)
            loaded = material_profiles.load_profile("Ni_50_200")
            config_io.save_config(loaded)
            loaded = config_io.load_config()
        self.assertAlmostEqual(loaded["pid_integral_time_s"], original["pid_integral_time_s"])
        original["pid_integral_time_s"] = loaded["pid_integral_time_s"]
        self.assertEqual(loaded, original)
        config = settings(**loaded)
        ctl._validate_trial_program([dict(start_T=23, target_T=600, ramp_speed_min=10)], config)
        table = loaded["current_feedforward_table"]
        self.assertEqual(table[-1]["temperature_c"], 600.)
        self.assertTrue(all(b["current_a"] >= a["current_a"] for a,b in zip(table,table[1:])))
        self.assertLess(table[-1]["current_a"], .05)
        provenance = loaded["current_feedforward_provenance"]
        self.assertFalse(provenance["unmeasured_extension"]["measured"])
        self.assertLess(provenance["derived_temperature_range_c"][-1], 190)
        self.assertEqual(loaded["invalid_measurement_policy"], "backoff")


if __name__ == "__main__":
    unittest.main()
