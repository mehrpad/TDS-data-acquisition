import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tds_control import config_io, material_profiles, tds_experiment as ctl
from tds_control.curve_io import load_resistance_temperature_file
from tools.build_nicr50_profile import build_profile


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT/"files/material_profiles/NiCr_50.json"


class NiCr50ProfileTests(unittest.TestCase):
    def test_run127_table_is_labelled_separately_from_estimated_temperature_curve(self):
        profile = json.loads(PROFILE.read_text())
        provenance = profile["current_feedforward_provenance"]
        self.assertTrue(provenance["measured_on_this_wire"])
        self.assertFalse(provenance['independent_temperature_calibration'])
        self.assertFalse(provenance['measured_final_hold'])
        self.assertFalse(provenance["wire_length_known"])
        self.assertEqual(provenance["source_run"], '130_test')
        self.assertEqual(profile['parallel_resistance_ohm'], 20.)
        self.assertEqual(profile["current_feedforward_table"][0]["current_a"], .001)
        self.assertEqual(profile["profile_name"], "NiCr_50")
        for key in ('startup_current', 'measurement_current_floor', 'tuning_start_current',
                    't0_current_search_start', 't0_calibration_current'):
            self.assertEqual(profile[key], .001)
        points = profile['current_feedforward_table']
        last_measured = provenance['run130_revision']['measured_indicated_range_c'][1]
        self.assertTrue(all(p['measured_on_this_wire'] == (p['temperature_c'] <= last_measured)
                            for p in points[1:-1]))
        self.assertFalse(points[-1]['measured_on_this_wire'])
        self.assertTrue(all(b['temperature_c'] > a['temperature_c'] and b['current_a'] >= a['current_a']
                            for a,b in zip(points,points[1:])))
        self.assertLess(points[-1]['current_a'], .2)
        with (PROFILE.parent/'NiCr_50_R_vs_T_estimated.csv').open() as stream:
            self.assertEqual({r['measured_on_this_wire'] for r in csv.DictReader(stream)}, {'False'})
        with (PROFILE.parent/'NiCr_50_current_table.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows),len(points))
        self.assertEqual({r['measured_on_this_wire'] for r in rows}, {'False','True'})

    def test_profile_and_curve_allow_600_with_headroom_and_preserve_cutoff(self):
        config = ctl.build_control_config(json.loads(PROFILE.read_text()))
        curve, _ = load_resistance_temperature_file(PROFILE.parent/"NiCr_50_R_vs_T_estimated.csv")
        model = ctl.build_temperature_interpolator(curve, config)
        program = [dict(start_T=23, target_T=600, ramp_speed_min=10)]
        ctl._validate_trial_program(program, config)
        ctl._validate_temperature_program_bounds(program, model)
        self.assertEqual(model.temperature_bounds, (0., 600.))
        self.assertEqual(float(model(model.x[-1])), 600.)
        current = ctl.current_feedforward_for_temperature(config, 600.)
        self.assertLess(current, .95*config["max_current"])
        wire_current = config['current_feedforward_table'][-1]['wire_current_a']
        self.assertLess(wire_current, config['max_wire_current_a'])
        self.assertLess(wire_current*model.x[-1], config["max_sample_voltage"])
        self.assertLess(wire_current**2*model.x[-1], config["max_power_w"])
        self.assertEqual(config["invalid_measurement_policy"], "backoff")
        with self.assertRaises(ctl.ExperimentSafetyError):
            ctl._enforce_temperature_safety(600.1, config)
        with self.assertRaises(ValueError):
            ctl._validate_trial_program([dict(start_T=23, target_T=601, ramp_speed_min=10)], config)

    def test_profile_settings_survive_save_load_and_toml(self):
        original = json.loads(PROFILE.read_text())
        self.assertFalse(set(original)-set(material_profiles.PROFILE_FIELDS)-{"profile_name"})
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(material_profiles, "PROFILES_DIR", Path(directory)), \
             patch.object(config_io, "CONFIG_PATH", Path(directory)/"config.toml"), \
             patch.object(config_io, "ensure_runtime_dirs"):
            material_profiles.save_profile(original["profile_name"], original)
            loaded = material_profiles.load_profile(original["profile_name"])
            config_io.save_config(loaded)
            loaded = config_io.load_config()
        self.assertAlmostEqual(loaded["pid_integral_time_s"], original["pid_integral_time_s"])
        original["pid_integral_time_s"] = loaded["pid_integral_time_s"]
        self.assertAlmostEqual(loaded["pid_ki"], original["pid_ki"], places=15)
        original["pid_ki"] = loaded["pid_ki"]
        self.assertEqual(loaded, original)

    def test_builder_rejects_wrong_material_and_unmatched_reference(self):
        with self.assertRaisesRegex(ValueError, "NiCr"):
            build_profile({"profile_name": "Ni_100_152"}, Path("not-read"))
        donor = json.loads((ROOT/"files/material_profiles/NiCr_100_163.json").read_text())
        with self.assertRaisesRegex(ValueError, "does not match"):
            build_profile(donor, PROFILE.parent/"Ni_50_200_R_vs_T.csv")


if __name__ == "__main__":
    unittest.main()
