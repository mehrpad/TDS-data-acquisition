import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tds_control import material_profiles


class MaterialProfilesTests(unittest.TestCase):
    def _use_temp_profiles_dir(self, temporary_directory):
        return patch.object(material_profiles, "PROFILES_DIR", Path(temporary_directory) / "material_profiles")

    def test_save_then_load_round_trips_the_profile_fields(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self._use_temp_profiles_dir(temporary_directory):
                config = {
                    "pid_gain_schedule": [
                        {"current_a": 0.02, "kp": 0.004, "ki": 0.0004, "kd": 0.0},
                        {"current_a": 0.8, "kp": 0.001, "ki": 0.0001, "kd": 0.0},
                    ],
                    "pid_kp": 0.004,
                    "pid_ki": 0.0004,
                    "pid_kd": 0.0,
                    "controller_mode": "PI",
                    "max_current_step_up": 0.05,
                    "max_current_step_down": 0.05,
                    "low_current_max_step_up": 0.02,
                    "low_current_max_step_down": 0.02,
                    "max_current": 1.0,
                    "resistivity_mode": "V_OVER_I",
                    # A field that is NOT in PROFILE_FIELDS should not be saved.
                    "experiment_name": "should_not_be_saved",
                }

                saved_path = material_profiles.save_profile("Ni100", config)
                self.assertTrue(saved_path.exists())

                loaded = material_profiles.load_profile("Ni100")
                self.assertEqual(loaded["pid_gain_schedule"], config["pid_gain_schedule"])
                self.assertEqual(loaded["max_current_step_up"], 0.05)
                self.assertEqual(loaded["profile_name"], "Ni100")
                self.assertNotIn("experiment_name", loaded)

    def test_list_profiles_is_empty_before_anything_is_saved(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self._use_temp_profiles_dir(temporary_directory):
                self.assertEqual(material_profiles.list_profiles(), [])

    def test_list_profiles_returns_saved_names_sorted(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self._use_temp_profiles_dir(temporary_directory):
                material_profiles.save_profile("Ni20Cr100", {"pid_kp": 0.001})
                material_profiles.save_profile("Ni100", {"pid_kp": 0.002})
                self.assertEqual(material_profiles.list_profiles(), ["Ni100", "Ni20Cr100"])

    def test_loading_a_missing_profile_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self._use_temp_profiles_dir(temporary_directory):
                with self.assertRaises(FileNotFoundError):
                    material_profiles.load_profile("does-not-exist")

    def test_profile_names_are_sanitized_for_the_filesystem(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self._use_temp_profiles_dir(temporary_directory):
                material_profiles.save_profile("Ni/Cr 80:20?", {"pid_kp": 0.001})
                names = material_profiles.list_profiles()
                self.assertEqual(len(names), 1)
                # No path separators or other characters that would escape the directory.
                self.assertNotIn("/", names[0])
                self.assertNotIn(":", names[0])

    def test_an_empty_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self._use_temp_profiles_dir(temporary_directory):
                with self.assertRaises(ValueError):
                    material_profiles.save_profile("   ", {"pid_kp": 0.001})

    def test_delete_profile_removes_the_file_and_reports_whether_it_existed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self._use_temp_profiles_dir(temporary_directory):
                material_profiles.save_profile("Ni100", {"pid_kp": 0.001})
                self.assertTrue(material_profiles.delete_profile("Ni100"))
                self.assertEqual(material_profiles.list_profiles(), [])
                self.assertFalse(material_profiles.delete_profile("Ni100"))

    def test_fields_outside_profile_fields_are_never_persisted(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self._use_temp_profiles_dir(temporary_directory):
                material_profiles.save_profile(
                    "Ni100",
                    {"pid_kp": 0.001, "DMM_v": "USB0::secret::INSTR", "PS": "USB0::secret2::INSTR"},
                )
                loaded = material_profiles.load_profile("Ni100")
                self.assertNotIn("DMM_v", loaded)
                self.assertNotIn("PS", loaded)


if __name__ == "__main__":
    unittest.main()
