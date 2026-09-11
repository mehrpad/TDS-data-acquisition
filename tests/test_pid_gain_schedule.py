import unittest
from unittest.mock import Mock, patch

import numpy as np

from tds_control import calibration
from tds_control.calibration import (
    _cap_next_tuning_target,
    _suggest_current_step,
    _tuning_schedule_targets,
    tune_pid_schedule,
)
from tds_control.tds_experiment import CONTROL_DEFAULTS, build_control_config, pid_gains_for_current


def _config(**overrides):
    config = dict(CONTROL_DEFAULTS)
    config.update(
        max_current=1.0,
        min_current=0.0,
        measurement_current_floor=0.005,
        temperature_tolerance_c=2.0,
        minimum_current_change=0.001,
    )
    config.update(overrides)
    return config


class PidGainsForCurrentTests(unittest.TestCase):
    def test_falls_back_to_flat_gains_without_a_schedule(self):
        config = _config(pid_kp=0.002, pid_ki=0.0003, pid_kd=0.0, controller_mode="PI")
        self.assertEqual(pid_gains_for_current(config, 0.5), (0.002, 0.0003, 0.0))

    def test_clamps_at_the_ends_of_the_schedule(self):
        schedule = [
            {"current_a": 0.02, "kp": 0.004, "ki": 0.0004, "kd": 0.0},
            {"current_a": 0.5, "kp": 0.001, "ki": 0.00005, "kd": 0.0},
        ]
        config = _config(pid_gain_schedule=schedule, controller_mode="PI")
        self.assertEqual(pid_gains_for_current(config, 0.001), (0.004, 0.0004, 0.0))
        self.assertEqual(pid_gains_for_current(config, 5.0), (0.001, 0.00005, 0.0))

    def test_interpolates_linearly_between_two_points(self):
        schedule = [
            {"current_a": 0.0, "kp": 0.0, "ki": 0.0, "kd": 0.0},
            {"current_a": 1.0, "kp": 1.0, "ki": 1.0, "kd": 1.0},
        ]
        config = _config(pid_gain_schedule=schedule, controller_mode="PID")
        kp, ki, kd = pid_gains_for_current(config, 0.25)
        self.assertAlmostEqual(kp, 0.25)
        self.assertAlmostEqual(ki, 0.25)
        self.assertAlmostEqual(kd, 0.25)

    def test_kd_is_zeroed_in_pi_mode_even_with_a_nonzero_schedule_kd(self):
        schedule = [{"current_a": 0.1, "kp": 0.002, "ki": 0.0002, "kd": 0.5}]
        config = _config(pid_gain_schedule=schedule, controller_mode="PI")
        self.assertEqual(pid_gains_for_current(config, 0.1), (0.002, 0.0002, 0.0))

    def test_three_point_schedule_interpolates_within_the_correct_segment(self):
        schedule = [
            {"current_a": 0.02, "kp": 0.004, "ki": 0.0004, "kd": 0.0},
            {"current_a": 0.4, "kp": 0.002, "ki": 0.0002, "kd": 0.0},
            {"current_a": 0.8, "kp": 0.001, "ki": 0.0001, "kd": 0.0},
        ]
        config = _config(pid_gain_schedule=schedule, controller_mode="PI")
        kp, _, _ = pid_gains_for_current(config, 0.6)
        # Midway between the 0.4 A and 0.8 A points (0.002 and 0.001).
        self.assertAlmostEqual(kp, 0.0015)


class SuggestCurrentStepTests(unittest.TestCase):
    def test_returns_minimum_change_for_a_non_positive_gain(self):
        config = _config()
        self.assertEqual(
            _suggest_current_step(0.0, 50.0, 2.0, config),
            config["minimum_current_change"],
        )
        self.assertEqual(
            _suggest_current_step(float("nan"), 50.0, 2.0, config),
            config["minimum_current_change"],
        )

    def test_a_slower_process_gets_a_bigger_suggested_step(self):
        config = _config()
        fast = _suggest_current_step(process_gain=800.0, time_constant_s=5.0, loop_time=2.0, config=config)
        slow = _suggest_current_step(process_gain=800.0, time_constant_s=100.0, loop_time=2.0, config=config)
        self.assertGreater(slow, fast)

    def test_result_is_clipped_to_the_configured_bounds(self):
        config = _config(minimum_current_change=0.001, max_current=1.0)
        # An enormous gain would otherwise suggest a step far below the floor.
        tiny = _suggest_current_step(process_gain=1e9, time_constant_s=100.0, loop_time=2.0, config=config)
        self.assertAlmostEqual(tiny, config["minimum_current_change"])
        # A vanishingly small gain would otherwise suggest an enormous step.
        huge = _suggest_current_step(process_gain=1e-6, time_constant_s=100.0, loop_time=2.0, config=config)
        self.assertAlmostEqual(huge, 0.1 * config["max_current"])


class TuningScheduleTargetsTests(unittest.TestCase):
    def test_three_well_separated_points_for_a_normal_max_current(self):
        config = _config(max_current=1.0, tuning_start_current=0.005)
        targets = _tuning_schedule_targets(config)
        names = [name for name, _ in targets]
        currents = [current for _, current in targets]
        self.assertEqual(names, ["low", "mid", "high"])
        self.assertAlmostEqual(currents[0], 0.005)
        self.assertAlmostEqual(currents[1], 0.4)
        self.assertAlmostEqual(currents[2], 0.8)

    def test_collapses_to_fewer_points_when_max_current_is_small(self):
        config = _config(max_current=0.01, tuning_start_current=0.005)
        targets = _tuning_schedule_targets(config)
        # 40%/80% of 0.01 A land too close to 0.005 A and to each other.
        self.assertLess(len(targets), 3)
        currents = [current for _, current in targets]
        self.assertEqual(currents, sorted(currents))


class CapNextTuningTargetTests(unittest.TestCase):
    """Reproduces the real failure: a high-gain wire and a naive 40% jump.

    A wire measured at ~800 C/A on-device sent a naive jump to 40% of a 1 A
    max_current past 500 C before the first stability sample was taken.
    """

    def test_a_high_gain_point_sharply_caps_the_next_target(self):
        config = _config(tuning_schedule_jump_max_rise_c=10.0, minimum_current_change=0.001)
        capped = _cap_next_tuning_target(
            desired_current=0.4, previous_current=0.017, previous_gain=804.0, config=config
        )
        # 10 C allowed / 804 C/A ~= 0.0124 A of headroom above the low point.
        self.assertLess(capped, 0.017 + 0.02)
        self.assertGreater(capped, 0.017)

    def test_a_low_gain_point_barely_caps_the_next_target(self):
        config = _config(tuning_schedule_jump_max_rise_c=10.0)
        capped = _cap_next_tuning_target(
            desired_current=0.4, previous_current=0.02, previous_gain=15.0, config=config
        )
        # 10 C / 15 C/A = 0.667 A of headroom - comfortably clears 0.4 A.
        self.assertAlmostEqual(capped, 0.4)

    def test_never_returns_less_than_the_previous_current(self):
        config = _config(tuning_schedule_jump_max_rise_c=10.0, minimum_current_change=0.001)
        capped = _cap_next_tuning_target(
            desired_current=0.4, previous_current=0.9, previous_gain=804.0, config=config
        )
        self.assertGreaterEqual(capped, 0.9)

    def test_takes_a_small_fixed_first_step_without_a_prior_gain(self):
        config = _config(tuning_current_step=0.001)
        capped = _cap_next_tuning_target(
            desired_current=0.4, previous_current=0.02, previous_gain=None, config=config
        )
        self.assertAlmostEqual(capped, 0.02 + 0.01)


class FirstOrderPlant:
    """A minimal simulated wire: R(T) linear, T(I) first-order with dead time."""

    def __init__(self, gain_c_per_a, tau_s, dead_time_s, base_temperature, base_resistance, ohm_per_c):
        self.gain = gain_c_per_a
        self.tau = tau_s
        self.dead_time = dead_time_s
        self.base_temperature = base_temperature
        self.base_resistance = base_resistance
        self.ohm_per_c = ohm_per_c
        self.current = 0.0
        self.temperature = base_temperature
        self.time_since_step = 1e9

    def set_current(self, current):
        if current != self.current:
            self.time_since_step = 0.0
        self.current = float(current)

    def advance(self, dt):
        self.time_since_step += dt
        effective_t = max(self.time_since_step - self.dead_time, 0.0)
        target_temperature = self.base_temperature + self.gain * self.current
        self.temperature = target_temperature + (self.base_temperature - target_temperature) * np.exp(
            -effective_t / self.tau
        )
        return self.temperature

    def resistance(self):
        return self.base_resistance + self.ohm_per_c * (self.temperature - self.base_temperature)

    def read_pair(self):
        r = self.resistance()
        v = self.current * r
        return v, self.current


def _temperature_interp_for_plant(plant):
    def interp(resistance):
        return plant.base_temperature + (float(resistance) - plant.base_resistance) / plant.ohm_per_c

    interp.x = np.array([plant.base_resistance - 1.0, plant.base_resistance + 50.0])
    interp.temperature_bounds = (-50.0, 2000.0)
    return interp


class TunePidScheduleIntegrationTests(unittest.TestCase):
    """End-to-end check against a simulated first-order wire, no real hardware."""

    def test_tune_pid_schedule_produces_an_ascending_three_point_schedule(self):
        plant = FirstOrderPlant(
            gain_c_per_a=15.0,
            tau_s=0.3,
            dead_time_s=0.05,
            base_temperature=23.0,
            base_resistance=2.0,
            ohm_per_c=0.01,
        )
        power_supply = Mock()
        power_supply.write_termination = None
        power_supply.read_termination = None

        siglent_double = Mock()
        siglent_double.set_current.side_effect = lambda _ps, current: plant.set_current(current)

        def read_pair(*_args, **_kwargs):
            plant.advance(0.05)
            return plant.read_pair()

        siglent_double.read_DMM_pair.side_effect = read_pair
        siglent_double.is_overload_reading.return_value = False
        siglent_double.increase_dc_range_if_needed.return_value = None
        siglent_double.configure_dc_range_from_config.return_value = None
        siglent_double.set_mode_speed.return_value = None
        siglent_double.set_output.return_value = None
        siglent_double.unlock_panel.return_value = None
        siglent_double.set_compliance_voltage.return_value = None

        resource_manager = Mock()
        resource_manager.open_resource.return_value = power_supply

        config = _config(
            DMM_v="fake-dmm-v",
            DMM_i="fake-dmm-i",
            PS="fake-ps",
            DMM_speed=10,
            max_current=1.0,
            tuning_start_current=0.02,
            controller_mode="PI",
            experiment_frequency=20.0,
            resistivity_mode="V_OVER_I",
            tuning_settle_time_s=0.0,
            tuning_between_attempts_s=0.0,
            tuning_baseline_samples=2,
            tuning_stable_current_samples=2,
            tuning_stable_current_a=1e-5,
            tuning_max_duration_s=20.0,
            tuning_min_temperature_rise_c=0.1,
            tuning_target_rise_c=0.5,
            tuning_min_observable_rise_c=0.05,
            tuning_no_response_timeout_s=10.0,
            tuning_plateau_timeout_s=10.0,
            tuning_plateau_idle_timeout_s=5.0,
            resistance_outlier_min_ohm=0.5,
            stable_resistance_spread_ohm=0.5,
            measurement_fail_limit=200,
            temperature_tolerance_c=20.0,
            safety_temp_margin_c=100.0,
        )

        temperature_interp = _temperature_interp_for_plant(plant)

        with patch.object(calibration, "pyvisa") as pyvisa_module, patch.object(
            calibration, "siglent", siglent_double
        ), patch.object(
            calibration, "_prepare_curve_interpolators", return_value=(None, None, temperature_interp)
        ), patch.object(
            calibration.tds_experiment, "siglent", siglent_double
        ):
            pyvisa_module.ResourceManager.return_value = resource_manager
            result = tune_pid_schedule(
                experiment_params={"target_T": 300.0},
                config=config,
                r_vs_t=np.array([[1.0, 2.0], [20.0, 30.0]]),
                base_temperature_hint=23.0,
                emitter=None,
            )

        schedule = result["schedule"]
        self.assertGreaterEqual(len(schedule), 2)
        currents = [point["current_a"] for point in schedule]
        self.assertEqual(currents, sorted(currents))
        for point in schedule:
            self.assertGreater(point["kp"], 0.0)
            self.assertGreater(point["ki"], 0.0)
        self.assertGreater(result["max_current_step_up"], 0.0)
        self.assertGreater(result["low_current_max_step_up"], 0.0)
        # dmm_v/dmm_i/power_supply all resolve to this same mock in the test
        # double, so _shutdown_instruments closing each of them closes it 3x.
        self.assertEqual(power_supply.close.call_count, 3)

    def test_a_high_gain_wire_gets_a_capped_mid_target_instead_of_a_runaway(self):
        """Reproduces the real failure with the fix in place.

        A wire whose low-current gain is ~800 C/A must not be sent straight
        to 40% of max_current: tune_pid_schedule should cap that jump using
        the gain just measured at the low point, and still finish rather than
        tripping the baseline-search safety check.
        """
        plant = FirstOrderPlant(
            gain_c_per_a=800.0,
            tau_s=0.2,
            dead_time_s=0.02,
            base_temperature=23.0,
            base_resistance=2.0,
            ohm_per_c=0.001,
        )
        power_supply = Mock()
        power_supply.write_termination = None
        power_supply.read_termination = None

        siglent_double = Mock()
        siglent_double.set_current.side_effect = lambda _ps, current: plant.set_current(current)

        def read_pair(*_args, **_kwargs):
            plant.advance(0.05)
            return plant.read_pair()

        siglent_double.read_DMM_pair.side_effect = read_pair
        siglent_double.is_overload_reading.return_value = False
        siglent_double.increase_dc_range_if_needed.return_value = None
        siglent_double.configure_dc_range_from_config.return_value = None
        siglent_double.set_mode_speed.return_value = None
        siglent_double.set_output.return_value = None
        siglent_double.unlock_panel.return_value = None
        siglent_double.set_compliance_voltage.return_value = None

        resource_manager = Mock()
        resource_manager.open_resource.return_value = power_supply

        config = _config(
            DMM_v="fake-dmm-v",
            DMM_i="fake-dmm-i",
            PS="fake-ps",
            DMM_speed=10,
            max_current=1.0,
            tuning_start_current=0.02,
            controller_mode="PI",
            experiment_frequency=20.0,
            resistivity_mode="V_OVER_I",
            tuning_settle_time_s=0.0,
            tuning_between_attempts_s=0.0,
            tuning_baseline_samples=2,
            tuning_stable_current_samples=2,
            tuning_stable_current_a=1e-5,
            tuning_max_duration_s=20.0,
            tuning_min_temperature_rise_c=0.05,
            tuning_target_rise_c=0.2,
            tuning_min_observable_rise_c=0.02,
            tuning_no_response_timeout_s=10.0,
            tuning_plateau_timeout_s=10.0,
            tuning_plateau_idle_timeout_s=5.0,
            resistance_outlier_min_ohm=0.5,
            stable_resistance_spread_ohm=0.5,
            measurement_fail_limit=200,
            # A tight, realistic window - this is what a naive 40%/80% jump
            # blew straight through in the field.
            temperature_tolerance_c=2.0,
            safety_temp_margin_c=15.0,
            tuning_temperature_window_c=40.0,
            tuning_schedule_jump_max_rise_c=10.0,
        )

        temperature_interp = _temperature_interp_for_plant(plant)

        with patch.object(calibration, "pyvisa") as pyvisa_module, patch.object(
            calibration, "siglent", siglent_double
        ), patch.object(
            calibration, "_prepare_curve_interpolators", return_value=(None, None, temperature_interp)
        ), patch.object(
            calibration.tds_experiment, "siglent", siglent_double
        ):
            pyvisa_module.ResourceManager.return_value = resource_manager
            result = tune_pid_schedule(
                experiment_params={"target_T": 300.0},
                config=config,
                r_vs_t=np.array([[1.0, 2.0], [20.0, 30.0]]),
                base_temperature_hint=23.0,
                emitter=None,
            )

        schedule = result["schedule"]
        self.assertGreaterEqual(len(schedule), 1)
        # The naive 40% target (0.4 A) must not have been used directly: at
        # 800 C/A that step alone would already exceed the safety window.
        for point in schedule:
            if point is not schedule[0]:
                self.assertLess(point["current_a"], 0.4)


class RunPidTuningAttemptMinimumSamplesTests(unittest.TestCase):
    """Reproduces the field bug: a high-gain point accepted after one sample.

    On device, the "mid" schedule point's very first post-step reading
    already exceeded the rise threshold, so the response was accepted with
    a single sample. _estimate_pid_from_step then collapsed dead_time_s and
    time_constant_s toward loop_time (times[0] - dead_time_s == 0), reporting
    tau=2.0 s for a wire whose true tau (measured properly at the low point
    moments earlier) was 23.8 s - and the resulting over-aggressive Ki got
    clamped across the rest of the schedule's current range.
    """

    def _run_attempt(self, plant, config, required_rise=3.0, smoothed_required_rise=1.95):
        power_supply = Mock()
        power_supply.write_termination = None
        power_supply.read_termination = None

        siglent_double = Mock()
        siglent_double.set_current.side_effect = lambda _ps, current: plant.set_current(current)

        def read_pair(*_args, **_kwargs):
            plant.advance(0.05)
            return plant.read_pair()

        siglent_double.read_DMM_pair.side_effect = read_pair
        siglent_double.is_overload_reading.return_value = False
        siglent_double.increase_dc_range_if_needed.return_value = None
        siglent_double.configure_dc_range_from_config.return_value = None
        siglent_double.set_mode_speed.return_value = None

        temperature_interp = _temperature_interp_for_plant(plant)

        with patch.object(calibration, "siglent", siglent_double):
            plant.set_current(config["_response_current"])
            return calibration._run_pid_tuning_attempt(
                dmm_v="fake-dmm-v",
                dmm_i="fake-dmm-i",
                power_supply=power_supply,
                temperature_interp=temperature_interp,
                config=config,
                emitter=None,
                baseline_current=config["_baseline_current"],
                response_current=config["_response_current"],
                base_temperature=plant.base_temperature,
                desired_rise=8.0,
                required_rise=required_rise,
                smoothed_required_rise=smoothed_required_rise,
                safe_temperature_limit=plant.base_temperature + 60.0,
                temperature_lower_bound=None,
                loop_time=0.05,
            )

    def test_a_fast_first_sample_still_waits_for_the_minimum_sample_count(self):
        # gain=162 C/A, tau=23.8 s: the real field values for the schedule
        # point this reproduces. A small enough rise threshold relative to
        # that gain (as happened on device) lets even the first, barely-risen
        # sample clear it - the actual mechanism does not matter, only that
        # it can happen well before the response curve has revealed its
        # shape.
        plant = FirstOrderPlant(
            gain_c_per_a=162.0,
            tau_s=23.8,
            dead_time_s=0.1,
            base_temperature=31.0,
            base_resistance=2.1,
            ohm_per_c=0.02,
        )
        config = _config(
            resistivity_mode="V_OVER_I",
            tuning_stable_current_a=1e-5,
            tuning_max_duration_s=2.0,
            tuning_min_response_samples=5,
            measurement_fail_limit=200,
            _baseline_current=0.0481,
            _response_current=0.0722,
        )

        result = self._run_attempt(
            plant, config, required_rise=0.01, smoothed_required_rise=0.005
        )

        self.assertEqual(result["status"], "usable_response")
        self.assertGreaterEqual(len(result["response"]), config["tuning_min_response_samples"])

        estimate = calibration._estimate_pid_from_step(
            response=result["response"],
            base_temperature=plant.base_temperature,
            step_current=config["_response_current"] - config["_baseline_current"],
            loop_time=0.05,
            min_temp_rise=0.01,
            controller_mode="PI",
        )
        # With only one sample, dead_time_s and time_constant_s both collapse
        # toward loop_time (0.05 s here, 2.0 s on device) - see
        # _estimate_pid_from_step. Requiring enough samples first should let
        # the estimate reflect the response actually unfolding, not that
        # floor.
        self.assertGreater(estimate["time_constant_s"], 0.05)


if __name__ == "__main__":
    unittest.main()
