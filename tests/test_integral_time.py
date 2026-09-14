import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6 import QtWidgets

from tds_control import config_io, material_profiles
from tds_control.app import Ui_TDS
from tds_control.pid import normalize_integral_time
from tds_control.tds_experiment import build_control_config, pid_gains_for_current


class IntegralTimeSettingsTests(unittest.TestCase):
    def test_legacy_gains_derive_time_without_changing_ki(self):
        result = build_control_config({"pid_kp":.0002,"pid_ki":.000002})
        self.assertAlmostEqual(result["pid_integral_time_s"],100)
        self.assertAlmostEqual(result["pid_ki"],.000002)

    def test_explicit_time_controls_runtime_ki_and_zero_disables_it(self):
        result = build_control_config({"pid_kp":.0002,"pid_ki":.1,"pid_integral_time_s":100})
        self.assertAlmostEqual(pid_gains_for_current(result,.1)[1],.000002)
        result["pid_integral_time_s"] = 0
        self.assertEqual(build_control_config(result)["pid_ki"],0)

    def test_i_only_retains_ki_without_finite_integral_time(self):
        result = build_control_config({"pid_kp":0,"pid_ki":.001})
        self.assertEqual(result["pid_ki"],.001)
        self.assertEqual(result["pid_integral_time_s"],0)

    def test_invalid_time_and_overflow_are_rejected(self):
        for time in [-1,float("nan"),float("inf"),1e-320]:
            with self.assertRaises(ValueError):
                normalize_integral_time({"pid_kp":.1,"pid_ki":.001,"pid_integral_time_s":time},True)

    def test_profile_json_and_toml_include_time_and_load_time_only_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(material_profiles,"PROFILES_DIR",Path(directory)/"profiles"), \
                 patch.object(config_io,"CONFIG_PATH",Path(directory)/"config.toml"), \
                 patch.object(config_io,"ensure_runtime_dirs"):
                config = build_control_config({"pid_kp":.0002,"pid_ki":.000002})
                path = material_profiles.save_profile("Ni",config)
                raw = json.loads(path.read_text())
                self.assertAlmostEqual(raw["pid_integral_time_s"],100)
                config_io.save_config(config)
                self.assertAlmostEqual(build_control_config(config_io.load_config())["pid_integral_time_s"],100)
                # Simulate a manually edited JSON containing Kp and Ti only.
                path.write_text(json.dumps({"pid_kp":.0001,"pid_integral_time_s":200}))
                profile = material_profiles.load_profile("Ni")
                self.assertAlmostEqual(profile["pid_ki"],.0000005)
                material_profiles.save_profile("NiCr",{"pid_kp":.0001,"pid_integral_time_s":200})
                self.assertAlmostEqual(material_profiles.load_profile("NiCr")["pid_ki"],.0000005)


class IntegralTimeGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config_patch = patch.object(config_io,"CONFIG_PATH",Path(self.directory.name)/"config.toml")
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)
        self.profile_patch = patch.object(material_profiles,"PROFILES_DIR",Path(self.directory.name)/"profiles")
        self.profile_patch.start()
        self.addCleanup(self.profile_patch.stop)
        self.window = QtWidgets.QMainWindow()
        settings = build_control_config({"experiment_frequency":.5,"DMM_speed":10,
                                        "pid_kp":.0002,"pid_ki":.000002})
        self.ui = Ui_TDS(settings)
        self.ui.setupUi(self.window)
        self.addCleanup(self.ui.timer_error.stop)
        self.addCleanup(self.window.close)

    def test_editing_time_changes_ki_and_persists_json_and_config(self):
        self.assertEqual(self.ui.pid_ti_edit.text(),"100")
        self.ui.pid_ti_edit.setText("200")
        self.assertTrue(self.ui.update_pid_ti())
        self.assertAlmostEqual(float(self.ui.pid_ki_edit.text()),.000001)
        self.assertAlmostEqual(build_control_config(config_io.load_config())["pid_ki"],.000001)
        self.ui.material_profile_combo.setCurrentText("Ni")
        self.ui.save_material_profile()
        self.assertAlmostEqual(material_profiles.load_profile("Ni")["pid_integral_time_s"],200)

    def test_editing_ki_or_kp_recalculates_time_and_zero_is_off(self):
        self.ui.pid_ki_edit.setText("0.000001")
        self.assertTrue(self.ui.update_pid_ki())
        self.assertEqual(self.ui.pid_ti_edit.text(),"200")
        self.ui.pid_kp_edit.setText("0.0001")
        self.assertTrue(self.ui.update_pid_kp())
        self.assertEqual(self.ui.pid_ti_edit.text(),"100")
        self.ui.pid_ti_edit.setText("0")
        self.assertTrue(self.ui.update_pid_ti())
        self.assertEqual(float(self.ui.pid_ki_edit.text()),0)

    def test_invalid_time_restores_field_and_does_not_change_gains(self):
        for time in ["-1","nan","inf","invalid"]:
            self.ui.pid_ti_edit.setText(time)
            self.assertFalse(self.ui.update_pid_ti())
            self.assertEqual(self.ui.pid_ti_edit.text(),"100")
            self.assertAlmostEqual(self.ui.config["pid_ki"],.000002)

    def test_unchanged_time_keeps_schedule_but_edit_clears_it(self):
        self.ui.config["pid_gain_schedule"] = [{"current_a":.1,"kp":.0002,"ki":.000002,"kd":0}]
        self.assertTrue(self.ui.update_pid_ti())
        self.assertTrue(self.ui.config["pid_gain_schedule"])
        self.ui.pid_ti_edit.setText("200")
        self.assertTrue(self.ui.update_pid_ti())
        self.assertEqual(self.ui.config["pid_gain_schedule"],[])

    def test_legacy_profile_and_tuning_refresh_time(self):
        material_profiles.save_profile("old",{"pid_kp":.0001,"pid_ki":.000001})
        self.ui.material_profile_combo.setCurrentText("old")
        self.ui.load_material_profile()
        self.assertEqual(self.ui.pid_ti_edit.text(),"100")
        self.ui.pid_tuning_finished({"schedule":[{"current_a":.1,"kp":.0003,"ki":.000001,"kd":0}],
                                     "max_current_step_up":.001,"max_current_step_down":.001})
        self.assertEqual(self.ui.pid_ti_edit.text(),"300")
        saved = config_io.load_config()
        self.assertAlmostEqual(saved["pid_integral_time_s"],300)
        self.assertEqual(len(saved["pid_gain_schedule"]),1)


if __name__ == "__main__":
    unittest.main()
