import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from tds_control import material_profiles, config_io
from tds_control import tds_experiment as ctl
from tds_control.pid import PIDController
from tds_control.measurement_quality import ResistancePowerGuard
from tds_control.data_saver import ExperimentDataSaver
from tds_control.curve_io import load_resistance_temperature_file
from tools.build_ramp_profiles import ramp_points


def settings(**overrides):
    return ctl.build_control_config(dict(experiment_frequency=.5, **overrides))


class LogicalFixTests(unittest.TestCase):
    def test_prediction_changes_output_but_not_measured_error_or_integral(self):
        config = settings(pid_kp=.001, pid_ki=.00001,
                          current_feedforward_table=[{"temperature_c": 100, "current_a": .05}])
        ctrl = PIDController(.001, .00001, 0, 100, output_limits=(.01, 1))
        ctrl.previous_setpoint = 100
        ctl._compute_next_current(ctrl, 99, 100, .05, .05, 200, 90, 0, config, 2)
        self.assertAlmostEqual(ctrl.integral, .00002)
        self.assertEqual(ctrl.previous_error, 1)
        self.assertEqual(ctrl.previous_measurement, 99)
        self.assertAlmostEqual(ctrl.requested_output, .05 + .001 + .00002 - .003)

    def test_prediction_is_included_in_actuator_tracking(self):
        ctrl = PIDController(.001, .00001, 0, 100, output_limits=(.01, 1))
        ctrl.compute(99, dt=2, bias=.05, output_correction=-.003)
        before = ctrl.integral
        ctrl.track_output(ctrl.requested_output, 2)
        self.assertEqual(ctrl.integral, before)

    def test_failed_half_pair_retries_both_channels(self):
        module = Mock()
        module.is_overload_reading.return_value = False
        module.read_DMM_pair.side_effect = [("0.3", "bad"), ("0.4", "0.02")]
        module.increase_dc_range_if_needed.return_value = None
        v, i, t, r = ctl.measure_resistivity(Mock(), Mock(), module, lambda x: x,
                                           config=settings())
        self.assertEqual((v, i, r), (.4, .02, 20))
        self.assertEqual(module.read_DMM_pair.call_count, 2)
        module.read_DMM.assert_not_called()

    def test_pair_failure_never_uses_sequential_fallback(self):
        module = Mock()
        module.is_overload_reading.return_value = False
        module.read_DMM_pair.side_effect = [RuntimeError("USB"), ("bad", ".02")]
        module.increase_dc_range_if_needed.return_value = None
        values = ctl.measure_resistivity(Mock(), Mock(), module, lambda x: x, config=settings())
        self.assertTrue(all(np.isnan(v) for v in values))
        module.read_DMM.assert_not_called()

    @patch("tds_control.tds_experiment.time.sleep")
    def test_retry_works_with_dynamic_guard_off_and_returns_a_coherent_pair(self, sleep):
        module = Mock()
        module.is_overload_reading.return_value = False
        # Initial outlier, then two new agreeing resistance samples with differing currents.
        module.read_DMM_pair.side_effect = [(".9", ".01"), (".4", ".01"), (".8", ".02")]
        module.increase_dc_range_if_needed.return_value = None
        v, i, t, r, ok = ctl._measure_with_retry(Mock(), Mock(), module, lambda x: x,
            config=settings(measurement_temperature_jump_guard_enabled=False), previous_resistance=2.)
        self.assertTrue(ok)
        self.assertAlmostEqual(v/i, r)
        self.assertEqual(r, 40.)
        self.assertEqual(module.read_DMM_pair.call_count, 3)

    @patch("tds_control.tds_experiment.time.sleep")
    def test_low_tcr_outlier_retries_even_below_ohmic_threshold(self, sleep):
        module = Mock()
        module.is_overload_reading.return_value = False
        module.read_DMM_pair.side_effect = [(".2202", ".01"), (".22", ".01")]
        module.increase_dc_range_if_needed.return_value = None
        v, i, t, r, ok = ctl._measure_with_retry(Mock(), Mock(), module,
            lambda x: 23+(x-22)*1000, config=settings(), previous_resistance=22.)
        self.assertTrue(ok)
        self.assertAlmostEqual(t, 23.)
        self.assertEqual(module.read_DMM_pair.call_count, 2)

    def test_inverse_flat_section_is_order_independent_and_exposes_ambiguity(self):
        curve = np.array([[1.,2.,2.,2.,3.], [20.,40.,45.,50.,80.]])
        a = ctl.build_temperature_interpolator(curve, settings())
        b = ctl.build_temperature_interpolator(curve[:,::-1], settings())
        self.assertEqual(float(a(2)), 45)
        self.assertEqual(float(b(2)), 45)
        self.assertEqual(a.plateau_intervals, ((2.,40.,50.),))

    def test_inverse_extrapolation_keeps_measured_endpoint_anchor(self):
        curve = np.array([[1.,2.,2.,3.],[20.,40.,50.,100.]])
        model = ctl._build_temperature_interpolator_from_curve(curve,(20,100),(20,50))
        self.assertEqual(float(model(2.5)),75.)

    def test_provisional_profiles_reject_unmeasured_target_or_wrong_ramp(self):
        config=settings(trial_max_temperature_c=110.,
            current_feedforward_provenance={"kind":"provisional_ramp","ramp_rate_c_min":10.})
        with self.assertRaisesRegex(ValueError,"limited to 110"):
            ctl._validate_trial_program([dict(start_T=23,target_T=150,ramp_speed_min=10)],config)
        with self.assertRaisesRegex(ValueError,"10 C/min"):
            ctl._validate_trial_program([dict(start_T=23,target_T=100,ramp_speed_min=30)],config)
        ctl._validate_trial_program([dict(start_T=23,target_T=100,ramp_speed_min=10)],config)

    def test_guard_detects_sustained_collapse_but_not_normal_cooling_or_a_spike(self):
        config=settings(resistance_power_guard_enabled=True)
        for mode in ("collapse","cooldown","spike"):
            guard=ResistancePowerGuard(config)
            reasons=[]
            for k in range(21):
                t=300-k*1.5 if mode != "spike" else (240 if k==15 else 300)
                current=.05+k*.001
                power=.1+k*.003
                target=300-k*2 if mode=="cooldown" else 300+k
                reasons.append(guard.update(k*2,t,22+t*.002,current,power,target))
            self.assertEqual(any(reasons),mode=="collapse")

    def test_final_hold_runs_for_simple_and_stepped_programs(self):
        for step in (0, 10):
            program = ctl.TemperatureProgram(23, step, 43, 60, 1, 2, 3)
            program.initialize(23)
            target, phase, done = program.update(23, 0)
            if step:
                target, phase, done = program.update(33, 10)
                self.assertEqual(phase, "hold")
                target, phase, done = program.update(33, 60)
                self.assertFalse(done)
                target, phase, done = program.update(43, 10)
            else:
                target, phase, done = program.update(43, 20)
            self.assertEqual(target, 43)
            self.assertEqual(phase, "hold")
            self.assertFalse(done)
            self.assertFalse(program.update(43, 59)[2])
            self.assertTrue(program.update(43, 1)[2])

    def test_zero_hold_still_finishes_on_arrival(self):
        program=ctl.TemperatureProgram(23,0,43,60,0,2,3)
        program.initialize(23)
        program.update(23,0)
        self.assertTrue(program.update(43,20)[2])

    @patch("tds_control.tds_experiment.time.sleep")
    @patch("tds_control.tds_experiment.time.monotonic", return_value=10.1)
    def test_measurement_waits_for_command_settling(self, monotonic, sleep):
        module=Mock()
        module.is_overload_reading.return_value=False
        module.read_DMM_pair.return_value=(".2",".01")
        module.increase_dc_range_if_needed.return_value=None
        config=settings(current_settle_time_s=.5)
        config["_current_command_time"]=10.
        ctl.measure_resistivity(Mock(),Mock(),module,lambda r:r,config=config)
        sleep.assert_called_once()
        self.assertAlmostEqual(sleep.call_args.args[0],.4)

    def test_invalid_feedback_holds_current_pauses_target_then_shuts_down(self):
        config=settings(DMM_v="v",DMM_i="i",PS="ps",DMM_speed=10,
                        measurement_fail_limit=2,invalid_measurement_policy="hold")
        emitter=Mock();emitter.stopped=False
        saver=Mock()
        ctrl_compute=Mock(wraps=ctl._compute_next_current)
        with patch.object(ctl.pyvisa,"ResourceManager"), patch.object(ctl,"siglent"), \
             patch.object(ctl.time,"sleep"), patch.object(ctl,"_shutdown_instruments") as shutdown, \
             patch.object(ctl,"_start_control_at_initial_current",return_value=(.01,.01)), \
             patch.object(ctl,"_measure_with_retry",return_value=(.2,.01,np.nan,20.,False)), \
             patch.object(ctl,"_compute_next_current",ctrl_compute), \
             patch.object(ctl,"_apply_control_current") as apply:
            with self.assertRaisesRegex(ctl.ExperimentSafetyError,"without increasing current"):
                ctl.tds(emitter,[dict(start_T=23,step_T=100,target_T=100,ramp_speed_min=10,
                                    hold_step_time_min=0)], np.array([[1.,3.],[0.,200.]]),
                        config,23.,saver)
        ctrl_compute.assert_not_called()
        apply.assert_not_called()
        shutdown.assert_called_once()
        saver.finalize.assert_called_once()
        self.assertEqual([c.args[0][1] for c in saver.enqueue.call_args_list],[23.,23.])
        self.assertTrue(all(c.args[0][6]==.01 for c in saver.enqueue.call_args_list))


class ExportAndProfileTests(unittest.TestCase):
    def test_source_bounds_and_diagnostics_survive_export_reload(self):
        source=np.array([[1.,2.],[23.,200.]])
        extended=np.array([[1.,2.,6.],[23.,200.,1000.]])
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            saver=ExperimentDataSaver(folder,extended,source_r_vs_t=source).start()
            record={"raw":np.nan,"nested":{"first":np.float64(2)}}
            saver.enqueue_diagnostics(record)
            record["nested"]["first"]=99
            saver.enqueue([0,23,23,0,.01,.01,.01,.0001,1.])
            saver.finalize()
            curve, info=load_resistance_temperature_file(folder/"r_vs_t.csv")
            np.testing.assert_allclose(curve,source)
            self.assertEqual(info["restored_source_bounds_c"],[23,200])
            saved=json.loads((folder/"control_diagnostics.jsonl").read_text())
            self.assertIsNone(saved["raw"])
            self.assertEqual(saved["nested"]["first"],2)
            with (folder/"r_vs_t_source.csv").open("a") as stream:
                stream.write("3,500\n")
            with self.assertRaisesRegex(ValueError,"hashes"):
                load_resistance_temperature_file(folder/"r_vs_t.csv")

    def test_trial_profiles_survive_application_save_and_toml_roundtrip(self):
        root=Path(__file__).resolve().parents[1]
        for name in ("Ni_100_152","NiCr_100_163"):
            original=json.loads((root/"files"/"material_profiles"/(name+".json")).read_text())
            missing=set(original)-set(material_profiles.PROFILE_FIELDS)-{"profile_name"}
            self.assertEqual(missing,set())
            with tempfile.TemporaryDirectory() as directory, \
                 patch.object(material_profiles,"PROFILES_DIR",Path(directory)), \
                 patch.object(config_io,"CONFIG_PATH",Path(directory)/"config.toml"), \
                 patch.object(config_io,"ensure_runtime_dirs"):
                material_profiles.save_profile(name,original)
                loaded=material_profiles.load_profile(name)
                config_io.save_config(loaded)
                loaded=config_io.load_config()
                # Ti is recomputed from Kp/Ki; allow floating-point division noise.
                self.assertAlmostEqual(loaded['pid_integral_time_s'], original['pid_integral_time_s'])
                loaded['pid_integral_time_s'] = original['pid_integral_time_s']
                self.assertEqual(loaded,original)

    def test_ramp_binning_uses_complete_target_bins_and_rms_current(self):
        import pandas as pd
        data=pd.DataFrame({"set_T":np.repeat([35.,45.],20),
                           "T":np.repeat([32.,42.],20),
                           "C_V":np.tile([.02,.04],20)})
        points,evidence=ramp_points(data,30,50,10)
        self.assertEqual(len(points),2)
        self.assertAlmostEqual(points[0]["current_a"],np.sqrt((.02**2+.04**2)/2),places=6)
        self.assertEqual(points[0]["temperature_c"],32)
        self.assertEqual(evidence[0]["samples"],20)


class ExtendedTrialProfileTests(unittest.TestCase):
    def test_600_degree_program_and_placeholder_preserve_limits(self):
        root = Path(__file__).resolve().parents[1]
        for name, power in (("Ni_100_152", .821924605868587), ("NiCr_100_163", .25)):
            profile = json.loads((root / "files/material_profiles" / (name + ".json")).read_text())
            config = ctl.build_control_config(profile)
            program = [dict(start_T=40, step_T=200, target_T=600,
                            ramp_speed_min=10, hold_step_time_min=1)]
            ctl._validate_trial_program(program, config)
            curve = ctl.build_temperature_interpolator(np.array([[1., 2., 3.], [23., 100., 200.]]), config)
            ctl._validate_temperature_program_bounds(program, curve)
            with self.assertRaises(ValueError):
                ctl._validate_trial_program([dict(program[0], target_T=601)], config)
            table = profile["current_feedforward_table"]
            self.assertEqual(table[-1]["temperature_c"], 600)
            self.assertEqual(table[-1]["current_a"], table[-2]["current_a"])
            self.assertLessEqual(ctl.current_feedforward_for_temperature(config, 600), profile["max_current"])
            self.assertEqual(profile["max_current"], .305911385 if name == "Ni_100_152" else .1)
            self.assertEqual(profile["max_power_w"], power)
            if "unmeasured_extension" in profile["current_feedforward_provenance"]:
                self.assertFalse(profile["current_feedforward_provenance"]["unmeasured_extension"]["measured"])
            self.assertLess(profile["current_feedforward_provenance"]["derived_temperature_range_c"][1], 300)


class CurrentRampUpdateTests(unittest.TestCase):
    def test_preserves_cold_interpolation_excludes_invalid_and_preserves_limits(self):
        import pandas as pd
        from tools.update_current_ramp_profile import update_profile
        base = {"max_current": .1, "max_power_w": .05,
                "current_feedforward_provenance": {},
                "current_feedforward_table": [
                    {"temperature_c": 23., "current_a": .01},
                    {"temperature_c": 90., "current_a": .055},
                    {"temperature_c": 120., "current_a": .061},
                    {"temperature_c": 600., "current_a": .061}]}
        data = pd.DataFrame({"time": np.arange(10),
                             "T": [80., 110., 120., 130., 140., 420., 440., 480., float('nan'), 580.],
                             "I": [.01, .09, .1, .11, .12, .17, .18, .19, .2, .22],
                             "C_V": [.01, .09, .1, .11, .12, .17, .18, .19, .2, .22],
                             "P": [.001]*10})
        result = update_profile(base, data, excluded_rows=(7,))
        for t in np.linspace(23, 100, 101):
            self.assertAlmostEqual(ctl.current_feedforward_for_temperature(settings(**base), t),
                                   ctl.current_feedforward_for_temperature(settings(**result), t))
        self.assertEqual((result['max_current'], result['max_power_w']), (.1, .05))
        used = [r for b in result['current_feedforward_provenance']['current_ramp_update']['bins'] for r in b['csv_data_rows']]
        self.assertEqual(used, [1, 2, 3, 4, 5, 6, 9])
        self.assertEqual(ctl.current_feedforward_for_temperature(settings(**result), 600), .1)
        self.assertEqual(update_profile(result, data, excluded_rows=(7,)), result)

    def test_current_guard_remains_enforced_and_is_reported(self):
        config = settings(max_current=.1, pid_kp=.001, pid_ki=.00001, max_current_step_up=.001, max_current_step_down=.001)
        controller = PIDController(.001, .00001, 0, 400, output_limits=(.01, .1))
        with patch('builtins.print') as output:
            current = ctl._compute_next_current(controller, 284, 400, .095, .096,
                                                600, 0, 10, config, 2)
        self.assertLessEqual(current, .095)
        self.assertTrue(controller.current_limit_active)
        self.assertTrue(any('CURRENT LIMIT' in str(c) for c in output.call_args_list))
        saver = Mock()
        ctl._record_control_diagnostics(saver, config, controller, 284, 284, 400, .095, current, 2, 'accepted')
        record = saver.enqueue_diagnostics.call_args.args[0]
        self.assertTrue(record['current_limit_active'])
        self.assertEqual(record['measured_current_increase_guard_a'], .095)


class MaximumTemperatureTests(unittest.TestCase):
    def test_rejects_invalid_limits_and_programs_above_limit(self):
        for value in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                settings(max_temperature_c=value)
        with self.assertRaisesRegex(ValueError, 'Maximum Temperature'):
            ctl._validate_trial_program([dict(start_T=23, target_T=301)], settings(max_temperature_c=300))
        ctl._validate_trial_program([dict(start_T=23, target_T=300)], settings(max_temperature_c=300))

    def test_raw_overtemperature_stops_before_out_of_range_masking(self):
        config = settings(max_temperature_c=600, curve_extrapolation_enabled=False)
        curve = ctl.build_temperature_interpolator(np.array([[1., 2.], [0., 600.]]), config)
        module = Mock()
        module.read_DMM_pair.return_value = (.021, .01)  # R=2.1 implies 660 C, outside curve
        module.is_overload_reading.return_value = False
        module.increase_dc_range_if_needed.return_value = None
        with self.assertRaisesRegex(ctl.ExperimentSafetyError, 'Maximum Temperature 600'):
            ctl.measure_resistivity(Mock(), Mock(), module, curve, config=config)
        ctl._enforce_temperature_safety(600, config)

    def test_current_ramp_shuts_instruments_down_on_temperature_cutoff(self):
        config = settings(DMM_v='v', DMM_i='i', PS='ps', DMM_speed=10, max_temperature_c=600)
        emitter = Mock(); emitter.stopped = False
        saver = Mock()
        with patch.object(ctl.pyvisa, 'ResourceManager'), patch.object(ctl, 'siglent'), \
             patch.object(ctl.time, 'sleep'), patch.object(ctl, 'prepare_power_supply_output'), \
             patch.object(ctl, '_set_current_if_needed', return_value=.01), \
             patch.object(ctl, '_shutdown_instruments') as shutdown, \
             patch.object(ctl, '_measure_with_retry', side_effect=ctl.ExperimentSafetyError('Maximum Temperature')):
            with self.assertRaisesRegex(ctl.ExperimentSafetyError, 'Maximum Temperature'):
                ctl.current_ramp(emitter, {'ramp_speed_min': .1}, np.array([[1., 2.], [23., 600.]]), config, saver)
        shutdown.assert_called_once()
        saver.finalize.assert_called_once()
