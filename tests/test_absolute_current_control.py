import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from tds_control import calibration, config_io, material_profiles, tds_experiment
from tds_control.pid import PIDController
from tds_control.tds_experiment import (
    TemperatureRateEstimator, _apply_control_current, _compute_next_current,
    _quantized_current, build_control_config, current_feedforward_for_temperature,
)


def config(**overrides):
    values = build_control_config({"experiment_frequency": 0.5})
    values.update(overrides)
    return values


def controller(settings):
    return PIDController(settings["pid_kp"], settings["pid_ki"], 0, 300,
                         output_limits=(0.01, settings["max_current"]),
                         integral_limits=(-1, 1))


def next_current(ctrl, settings, current=0.1, temperature=299, setpoint=300, dt=2, rate=None):
    return _compute_next_current(ctrl, temperature, setpoint, current, current,
                                 400, rate, 10, settings, dt)


class AbsoluteCurrentTests(unittest.TestCase):
    def test_p_only_constant_error_does_not_accumulate_current(self):
        ctrl = PIDController(.001, 0, 0, 300)
        self.assertEqual([ctrl.compute(290, dt=2, bias=.1) for _ in range(20)], [.11] * 20)

    def test_integral_uses_actual_elapsed_time_once(self):
        ctrl = PIDController(0, .001, 0, 300)
        self.assertAlmostEqual(ctrl.compute(299, dt=2, bias=.1), .102)
        self.assertAlmostEqual(ctrl.compute(299, dt=3.5, bias=.1), .1055)

    def test_sub_resolution_integral_accumulates_until_a_real_command_is_sent(self):
        settings = config(pid_kp=0, pid_ki=.0001, temperature_prediction_time_s=0)
        ctrl = controller(settings)
        supply = Mock()
        current = .01
        for _ in range(3):
            request = next_current(ctrl, settings, current, temperature=299)
            current = _apply_control_current(supply, request, current, settings, ctrl, 2)
        self.assertAlmostEqual(current, .011)
        supply.write.assert_called_once_with("CURR 0.011")

    def test_output_tracking_unwinds_a_slew_limited_integrator(self):
        ctrl = PIDController(0, .001, 0, 300, integral_limits=(-1,1))
        ctrl.integral = .3
        ctrl.compute(300, dt=2, bias=.1)
        ctrl.track_output(.05, 2, tracking_time_s=0)
        self.assertAlmostEqual(ctrl.compute(300, dt=2, bias=.1), .05)

    def test_gain_and_phase_changes_preserve_holding_correction(self):
        ctrl = PIDController(.001, .0001, 0, 300)
        before = ctrl.compute(299, dt=2, bias=.1)
        ctrl.set_gains(.0002, .00001, 0)
        self.assertAlmostEqual(ctrl.compute(299, dt=2, bias=.1, integrate=False), before)
        ctrl.reset(measurement=299, preserve_integral=True)
        self.assertAlmostEqual(ctrl.compute(299, dt=2, bias=.1, integrate=False), before)

    def test_below_target_can_reduce_current_and_does_not_force_catchup(self):
        settings = config(pid_kp=.0004, pid_ki=0,
                          current_feedforward_table=[{"temperature_c":300,"current_a":.1}])
        ctrl = controller(settings)
        self.assertAlmostEqual(next_current(ctrl,settings,.15,295), .149)
        self.assertAlmostEqual(next_current(ctrl,settings,.102,295), .102)

    def test_rate_threshold_does_not_force_full_downstep_or_reset(self):
        settings = config(pid_kp=0,pid_ki=0,
                          current_feedforward_table=[{"temperature_c":300,"current_a":.1}])
        ctrl = controller(settings)
        self.assertAlmostEqual(next_current(ctrl,settings,.1,299,rate=15), .1)

    def test_invalid_reading_does_not_integrate_reused_temperature(self):
        settings = config(pid_kp=0,pid_ki=.001)
        ctrl = controller(settings)
        _compute_next_current(ctrl, 299, 300, .1, .1, 400, 0, 10, settings, 2, integrate=False)
        self.assertEqual(ctrl.integral, 0)

    def test_irregular_period_scales_slew_by_elapsed_time(self):
        settings = config(pid_kp=.01,pid_ki=0)
        ctrl = controller(settings)
        self.assertAlmostEqual(next_current(ctrl,settings,.1,280,dt=3.5), .10175)

    def test_transmitted_command_grid_and_software_bounds(self):
        self.assertEqual(_quantized_current(.10049, .01, 1), .1)
        self.assertEqual(_quantized_current(.10051, .01, 1), .101)
        self.assertEqual(_quantized_current(1, .0102, .1007), .1)
        self.assertEqual(_quantized_current(.01, .0102, .1007), .011)
        with self.assertRaises(ValueError):
            _quantized_current(.1, .1002, .1007)

    def test_suppressed_write_tracks_accepted_not_requested_current(self):
        settings = config(minimum_current_change=.002)
        ctrl = controller(settings)
        supply = Mock()
        accepted = _apply_control_current(supply,.101,.1,settings,ctrl,2)
        self.assertEqual(accepted,.1)
        self.assertEqual(ctrl.output,.1)
        supply.write.assert_not_called()

    def test_feedforward_uses_measured_operating_point_for_gain_schedule(self):
        settings = config(current_feedforward_table=[{"temperature_c":300,"current_a":.1}],
                          pid_gain_schedule=[{"current_a":.1,"kp":.001,"ki":0},
                                             {"current_a":.2,"kp":.01,"ki":0}])
        ctrl = controller(settings)
        next_current(ctrl,settings,.2)
        self.assertAlmostEqual(ctrl.kp,.001)

    def test_feedforward_interpolates_and_rejects_duplicate_temperatures(self):
        settings = config(current_feedforward_table=[{"temperature_c":100,"current_a":.02},
                                                     {"temperature_c":300,"current_a":.1}])
        self.assertAlmostEqual(current_feedforward_for_temperature(settings,200),.06)
        self.assertAlmostEqual(current_feedforward_for_temperature(settings,500),.1)
        settings["current_feedforward_table"].append({"temperature_c":100,"current_a":.03})
        with self.assertRaises(ValueError):
            current_feedforward_for_temperature(settings,200)

    def test_quantized_delayed_plant_tracks_ramp_then_holds_without_current_cycling(self):
        # Independent first-order plant: equilibrium T = 220 + 800*I,
        # tau=12 s and one-sample actuator delay, with 0.15 C sensor noise.
        settings = config(pid_kp=.0005,pid_ki=.00003,
                          current_feedforward_table=[{"temperature_c":240,"current_a":.025},
                                                     {"temperature_c":320,"current_a":.125}])
        ctrl = controller(settings)
        supply = Mock()
        temperature, current, delayed_current = 260., .05, .05
        estimator = TemperatureRateEstimator(8)
        errors = []
        timestamp = 0.0
        rng = np.random.default_rng(7)
        for step in range(300):
            dt = (2, 2.1, 3)[step % 3]
            timestamp += dt
            temperature += (1-np.exp(-dt/12))*(220+800*delayed_current-temperature)
            measured = temperature + rng.normal(0,.15)
            setpoint = min(280+step/3,300)
            rate = estimator.update(timestamp,measured)
            request = next_current(ctrl,settings,current,measured,setpoint,dt,rate)
            delayed_current = current
            current = _apply_control_current(supply,request,current,settings,ctrl,dt)
            if step >= 200:
                errors.append(temperature-setpoint)
        self.assertLess(abs(np.mean(errors)),.5)
        self.assertLess(np.ptp(errors),2)


class ConfigurationAndTuningTests(unittest.TestCase):
    def test_measured_tables_survive_toml_save_and_material_profile(self):
        settings = config(current_feedforward_table=[{"temperature_c":300,"current_a":.1}],
                          pid_gain_schedule=[{"current_a":.1,"kp":1e-6,"ki":1e-7,"kd":0}])
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(config_io,"CONFIG_PATH",Path(directory)/"config.toml"), \
                 patch.object(config_io,"ensure_runtime_dirs"), \
                 patch.object(material_profiles,"PROFILES_DIR",Path(directory)/"profiles"):
                config_io.save_config(settings)
                saved = config_io.load_config()
                material_profiles.save_profile("NiCr",saved)
                profile = material_profiles.load_profile("NiCr")
        self.assertEqual(profile["current_feedforward_table"],settings["current_feedforward_table"])
        self.assertEqual(profile["pid_gain_schedule"],settings["pid_gain_schedule"])

    def test_tuning_keeps_small_high_gain_wire_gains_and_slow_integral(self):
        times = np.arange(2,122,2)
        response = [{"elapsed_s":t,"temperature":23+10*(1-np.exp(-t/12))} for t in times]
        gains = calibration._estimate_pid_from_step(response,23,.001,2,3)
        self.assertGreater(gains["Kp"],0)
        self.assertLess(gains["Kp"],.001)
        self.assertLess(gains["Ki"],1e-5)
        self.assertGreaterEqual(gains["Kp"]/gains["Ki"],30)
        low_gain = calibration._estimate_pid_from_step(response,23,100,2,3)
        self.assertAlmostEqual(low_gain["Kp"],.05)
        self.assertGreaterEqual(low_gain["Kp"]/low_gain["Ki"],30)

    def test_live_loop_tracks_quantized_current_through_invalid_reading_recovery(self):
        settings = config(DMM_v="fake-v",DMM_i="fake-i",PS="fake-ps",DMM_speed=10,
                          current_feedforward_table=[{"temperature_c":100,"current_a":.1}])
        emitter = Mock(stopped=False)
        samples = []
        supply = Mock()
        resource_manager = Mock()
        resource_manager.open_resource.return_value = supply

        def emitted(*args):
            samples.append(args)
            if len(samples) == 3:
                emitter.stopped = True

        readings = [(1,.01,99,100,True), (1,.01,float("nan"),100,True),
                    (1,.01,99,100,True)]
        parameters = [{"start_T":100,"step_T":100,"target_T":200,
                       "ramp_speed_min":10,"hold_step_time_min":0}]
        with patch.object(tds_experiment.pyvisa,"ResourceManager",return_value=resource_manager), \
             patch.object(tds_experiment,"prepare_power_supply_output"), \
             patch.object(tds_experiment,"siglent",Mock()), \
             patch.object(tds_experiment,"_shutdown_instruments"), \
             patch.object(tds_experiment,"_measure_with_retry",side_effect=readings), \
             patch.object(tds_experiment,"_emit_measurement",side_effect=emitted), \
             patch.object(tds_experiment.time,"sleep"), \
             patch.object(tds_experiment.time,"monotonic",side_effect=np.arange(100,dtype=float)), \
             patch.object(tds_experiment.TemperatureProgram,"update",return_value=(100,"hold",False)):
            tds_experiment.tds(emitter,parameters,np.array([[1,5],[23,400]]),settings,100)
        self.assertEqual(len(samples),3)
        self.assertTrue(np.isnan(samples[1][2]))
        for sample in samples:
            self.assertAlmostEqual(sample[5]*1000,round(sample[5]*1000))
        self.assertGreaterEqual(samples[2][5],.01)


class RateEstimatorTests(unittest.TestCase):
    def test_regression_handles_irregular_timestamps_and_reset(self):
        estimator = TemperatureRateEstimator(8)
        self.assertIsNone(estimator.update(0,100))
        self.assertIsNone(estimator.update(2,100.4))
        self.assertAlmostEqual(estimator.update(5,101),12)
        self.assertIsNone(estimator.update(6,200,reset=True))
        self.assertIsNone(estimator.update(7,float("nan")))


if __name__ == "__main__":
    unittest.main()
