import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6 import QtWidgets

from tds_control import calibration, config_io, material_profiles, tds_experiment as ctl
from tds_control.app import Ui_TDS


def config_for_test(overrides=None):
    return ctl.build_control_config({"experiment_frequency": .5, "DMM_speed": 10,
        "DMM_v": "voltage", "DMM_i": "current", "PS": "supply", **(overrides or {})})


class T0CurrentTests(unittest.TestCase):
    def search(self, config, readings, ceiling=.001):
        with patch.object(calibration, "_sleep_with_stop"), \
             patch.object(ctl, "measure_resistivity", side_effect=readings), \
             patch.object(calibration.siglent, "set_current") as command:
            try:
                result = calibration._find_stable_current_setpoint(
                    dmm_v=Mock(), dmm_i=Mock(), power_supply=Mock(),
                    temperature_interp=Mock(), config=config, start_current=.001,
                    max_current=ceiling, step_current=.001, settle_time_s=0,
                    stable_samples=3, minimum_current=.0001, emitter=None, label="T0")
            except ValueError:
                result = None
        return result, [call.kwargs["current"] for call in command.call_args_list]

    def test_search_uses_one_ma_even_with_large_experiment_deadband_and_floor(self):
        config = config_for_test({"minimum_current_change": .01, "measurement_current_floor": .01})
        result, commands = self.search(config, [( .01, .001, 23., 10.)]*3)
        self.assertEqual(commands, [.001])
        self.assertEqual(result[0], .001)
        self.assertEqual(result[1][0]["setpoint_current"], .001)

    def test_invalid_or_noisy_readings_cannot_raise_current_above_fixed_cap(self):
        config = config_for_test()
        for readings in ([(np.nan, np.nan, np.nan, np.nan)]*5,
                         [(.01, .001, 23., r) for r in (10., 11., 12.)]):
            result, commands = self.search(config, readings)
            self.assertIsNone(result)
            self.assertEqual(commands, [.001])

    def test_search_respects_non_grid_ceiling(self):
        readings = [(np.nan, np.nan, np.nan, np.nan)]*10
        result, commands = self.search(config_for_test(), readings, ceiling=.0029)
        self.assertIsNone(result)
        self.assertEqual(commands, [.001, .002])

    def test_invalid_t0_settings_rejected(self):
        for setting in ("t0_current_search_start", "t0_calibration_current", "t0_current_step"):
            for value in (0, .0005, float("nan"), float("inf")):
                with self.subTest(setting=setting, value=value), self.assertRaises(ValueError):
                    ctl.build_control_config({setting: value})
        with self.assertRaisesRegex(ValueError, "ceiling"):
            ctl.build_control_config({"t0_current_search_start": .005, "t0_calibration_current": .001})

    def calibrate(self, reading):
        config = config_for_test({"t0_current_search_start": .001, "t0_calibration_current": .001,
            "t0_dmm_voltage_range_v": .2, "t0_dmm_current_range_a": .002,
            "t0_pair_samples": 9, "measurement_pair_samples": 5,
            "dmm_voltage_range_v": 2., "dmm_current_range_a": .02, "psu_keepalive_current": .01})
        instrument = Mock()
        with patch.object(calibration.pyvisa, "ResourceManager") as manager, \
             patch.object(calibration, "_sleep_with_stop"), patch.object(calibration.time, "sleep"), \
             patch.object(calibration.siglent, "set_current") as command, \
             patch.object(ctl, "measure_resistivity", return_value=reading), \
             patch.object(calibration.siglent, "configure_dc_range_from_config") as ranges, \
             patch.object(calibration.siglent, "set_mode_speed"), \
             patch.object(calibration.siglent, "set_output") as output:
            manager.return_value.open_resource.return_value = instrument
            try:
                result = calibration.calibrate_temperature_curve(np.array([[10., 20.], [23., 100.]]), 23., config)
            except ValueError:
                result = None
        self.assertEqual(config["dmm_voltage_range_v"], 2.)
        self.assertEqual(config["dmm_current_range_a"], .02)
        self.assertEqual(config["psu_keepalive_current"], .01)
        self.assertEqual(config["measurement_pair_samples"], 5)
        self.assertTrue(all(0 <= call.kwargs["current"] <= .001 for call in command.call_args_list))
        self.assertEqual(command.call_args_list[-1].kwargs["current"], 0.)
        self.assertEqual(output.call_args_list[-1].kwargs["state"], "OFF")
        for call in ranges.call_args_list:
            self.assertEqual(call.args[2]["dmm_voltage_range_v"], .2)
            self.assertEqual(call.args[2]["dmm_current_range_a"], .002)
            self.assertEqual(call.args[2]["measurement_pair_samples"], 9)
        return result

    def test_complete_calibration_uses_low_ranges_and_current_without_changing_run_settings(self):
        result = self.calibrate((.010001, .001, 23., 10.001))
        self.assertAlmostEqual(result[0, np.flatnonzero(result[1] == 23.)[0]], 10.001)

    def test_failed_calibration_shuts_off_supply(self):
        self.assertIsNone(self.calibrate((np.nan, np.nan, np.nan, np.nan)))

    def test_all_four_profiles_preserve_t0_settings_through_json_and_toml(self):
        root = Path(__file__).resolve().parents[1]/"files/material_profiles"
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(material_profiles, "PROFILES_DIR", Path(directory)), \
             patch.object(config_io, "CONFIG_PATH", Path(directory)/"config.toml"), \
             patch.object(config_io, "ensure_runtime_dirs"):
            for name, current in (("Ni_100_152", .005), ("NiCr_100_163", .005),
                                  ("Ni_50_200", .001), ("NiCr_50_provisional", .001)):
                profile = json.loads((root/f"{name}.json").read_text())
                material_profiles.save_profile(name, profile)
                config_io.save_config(material_profiles.load_profile(name))
                loaded = ctl.build_control_config(config_io.load_config())
                self.assertEqual(loaded["t0_current_search_start"], current)
                self.assertEqual(loaded["t0_calibration_current"], current)
                for key in (key for key in profile if key.startswith("t0_")):
                    self.assertEqual(loaded[key], profile[key])


class T0GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_initial_current_edit_and_calibration_click_preserve_t0(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(config_io, "CONFIG_PATH", Path(directory)/"config.toml"), \
             patch.object(material_profiles, "PROFILES_DIR", Path(directory)/"profiles"), \
             patch.object(config_io, "ensure_runtime_dirs"), \
             patch("tds_control.app.CalibrationWorkerThread") as worker:
            window = QtWidgets.QMainWindow()
            ui = Ui_TDS(config_for_test({"t0_current_search_start": .001,
                "t0_calibration_current": .001, "startup_current": .005}))
            ui.setupUi(window)
            try:
                self.assertEqual(float(ui.calibration_start_current.text()), .005)
                ui.calibration_start_current.setText("0.01")
                ui.update_calibration_start_current()
                self.assertEqual(ui.config["startup_current"], .01)
                self.assertEqual(ui.config["t0_current_search_start"], .001)
                ui.r_vs_t = np.array([[10., 20.], [23., 100.]])
                ui.calibrate_base_temperature()
                self.assertEqual(worker.call_args.args[-1]["t0_current_search_start"], .001)
                self.assertEqual(worker.call_args.args[-1]["t0_calibration_current"], .001)
                worker.return_value.start.assert_called_once()
            finally:
                ui.timer_error.stop()
                window.close()
