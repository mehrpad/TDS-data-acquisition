import unittest
from unittest.mock import Mock

import numpy as np

from tds_control.tds_experiment import (
    CONTROL_DEFAULTS,
    ExperimentSafetyError,
    _limit_current_slew,
    build_control_config,
    get_resistivity_mode,
    measure_resistivity,
    resistivity_loop_time,
    current_step_scale,
)


def _config(**overrides):
    config = build_control_config(
        {
            "max_current": 30.0,
            "max_current": 3.0,
            "max_power_w": 2.5,
            "experiment_frequency": 0.5,
            "resistivity_output_settle_s": 0.0,
            "resistivity_measure_time_s": 0.0001,
            "resistivity_heat_time_s": 0.0,
            "dmm_voltage_range_v": 0.2,
            "dmm_current_range_a": 0.002,
            "dmm_staged_ranging_enabled": False,
        }
    )
    config.update(overrides)
    return config


class IdentityTemperatureModel:
    temperature_bounds = (-1000.0, 1000.0)

    def __call__(self, resistance):
        return float(resistance)


def _siglent_double(pair_reading, resistance_reading=None):
    module = Mock()
    module.read_DMM_pair.side_effect = pair_reading
    module.is_overload_reading.return_value = False
    module.increase_dc_range_if_needed.return_value = None
    module.read_DMM_resistance.return_value = resistance_reading
    return module


class ResistivityModeTests(unittest.TestCase):
    def test_v_over_i_mode_leaves_the_output_alone(self):
        siglent = _siglent_double([("0.0200", "0.0100")])
        voltage, current, temperature, resistance = measure_resistivity(
            Mock(), Mock(), siglent, IdentityTemperatureModel(), config=_config()
        )
        self.assertAlmostEqual(resistance, 2.0)
        self.assertAlmostEqual(voltage, 0.02)
        self.assertAlmostEqual(current, 0.01)
        self.assertAlmostEqual(temperature, 2.0)
        siglent.set_output.assert_not_called()

    def test_offset_corrected_mode_subtracts_the_quiet_window_thermal_emf(self):
        # 0.0200 V while heating, 0.0040 V of contact thermal EMF with no current.
        siglent = _siglent_double([("0.0200", "0.0100"), ("0.0040", "0.0000")])
        power_supply = Mock()
        voltage, current, _, resistance = measure_resistivity(
            Mock(),
            Mock(),
            siglent,
            IdentityTemperatureModel(),
            config=_config(resistivity_mode="OFFSET_CORRECTED"),
            power_supply=power_supply,
        )
        self.assertAlmostEqual(voltage, 0.016)
        self.assertAlmostEqual(current, 0.01)
        self.assertAlmostEqual(resistance, 1.6)
        self.assertEqual(
            [call.kwargs["state"] for call in siglent.set_output.call_args_list],
            ["OFF", "ON"],
        )

    def test_four_wire_mode_reads_resistance_directly_and_restores_the_dc_range(self):
        siglent = _siglent_double([("0.0200", "0.0100")], resistance_reading="3.5000")
        power_supply = Mock()
        _, _, temperature, resistance = measure_resistivity(
            Mock(),
            Mock(),
            siglent,
            IdentityTemperatureModel(),
            config=_config(resistivity_mode="FOUR_WIRE", dmm_resistance_range_ohm=200.0),
            power_supply=power_supply,
        )
        self.assertAlmostEqual(resistance, 3.5)
        self.assertAlmostEqual(temperature, 3.5)
        siglent.configure_fres_range.assert_called_once()
        siglent.configure_dc_range.assert_called_once()
        self.assertEqual(
            [call.kwargs["state"] for call in siglent.set_output.call_args_list],
            ["OFF", "ON"],
        )

    def test_the_output_is_restored_even_when_the_quiet_window_read_fails(self):
        siglent = _siglent_double([("0.0200", "0.0100")])
        siglent.read_DMM_resistance.side_effect = RuntimeError("meter offline")
        with self.assertRaises(RuntimeError):
            measure_resistivity(
                Mock(),
                Mock(),
                siglent,
                IdentityTemperatureModel(),
                config=_config(resistivity_mode="FOUR_WIRE"),
                power_supply=Mock(),
            )
        self.assertEqual(
            [call.kwargs["state"] for call in siglent.set_output.call_args_list],
            ["OFF", "ON"],
        )

    def test_a_duty_cycled_mode_without_a_power_supply_is_rejected(self):
        siglent = _siglent_double([("0.0200", "0.0100")])
        with self.assertRaisesRegex(ValueError, "power-supply session"):
            measure_resistivity(
                Mock(),
                Mock(),
                siglent,
                IdentityTemperatureModel(),
                config=_config(resistivity_mode="OFFSET_CORRECTED"),
            )

    def test_loop_period_follows_the_duty_cycle_windows(self):
        continuous = _config()
        self.assertAlmostEqual(resistivity_loop_time(continuous), 2.0)

        duty_cycled = _config(
            resistivity_mode="FOUR_WIRE",
            resistivity_heat_time_s=3.0,
            resistivity_measure_time_s=2.0,
        )
        self.assertAlmostEqual(resistivity_loop_time(duty_cycled), 5.0)

    def test_a_resistance_below_the_curve_survives_so_t0_can_still_anchor_it(self):
        """An out-of-range resistance is a good measurement with no conversion.

        T0 calibration exists to scale an unanchored curve, and its
        current-only fallback needs the resistance that the curve cannot yet
        convert. Nulling it here stalls the calibration that would fix it.
        """
        curve = IdentityTemperatureModel()
        curve.x = np.array([21.0, 24.5])
        siglent = _siglent_double([("0.0091", "0.000425")], resistance_reading="20.5840")

        _, _, temperature, resistance = measure_resistivity(
            Mock(),
            Mock(),
            siglent,
            curve,
            calibration=True,
            config=_config(resistivity_mode="FOUR_WIRE"),
            power_supply=Mock(),
        )
        self.assertTrue(np.isnan(temperature))
        self.assertAlmostEqual(resistance, 20.584)

    def test_slew_limits_hold_the_same_ramp_rate_across_loop_periods(self):
        """A longer duty-cycled loop must not shrink the volts-per-minute.

        The step limits are per loop but were chosen for the
        1/experiment_frequency period, so a 5 s cycle has to be allowed 2.5x
        the step a 2 s cycle gets or the controller cannot follow the setpoint.
        """
        continuous = _config()
        duty_cycled = _config(
            resistivity_mode="FOUR_WIRE",
            resistivity_heat_time_s=4.0,
            resistivity_measure_time_s=1.0,
        )
        self.assertAlmostEqual(current_step_scale(continuous), 1.0)
        self.assertAlmostEqual(current_step_scale(duty_cycled), 2.5)

        def ramp_rate_v_per_min(config):
            reached = _limit_current_slew(1.0, 0.01, 0.0, 30.0, config)
            return (reached - 0.01) * 60.0 / resistivity_loop_time(config)

        self.assertAlmostEqual(ramp_rate_v_per_min(continuous), ramp_rate_v_per_min(duty_cycled))

    def test_an_open_contact_is_caught_by_the_sample_voltage_ceiling(self):
        """Constant current turns a bad contact into a voltage runaway.

        The supply raises its terminal voltage to hold the set current, so the
        sample voltage climbs toward compliance rather than the current falling.
        """
        siglent = _siglent_double([("12.5000", "0.0100")])
        with self.assertRaisesRegex(ExperimentSafetyError, "max_sample_voltage"):
            measure_resistivity(
                Mock(),
                Mock(),
                siglent,
                IdentityTemperatureModel(),
                config=_config(max_sample_voltage=10.0, max_power_w=1000.0),
            )

    def test_legacy_voltage_config_keys_are_migrated(self):
        migrated = build_control_config(
            {
                "startup_voltage": 0.02,
                "t0_voltage_search_start": 0.03,
                "max_voltage": 30.0,
                "max_voltage_step_up": 0.002,
            }
        )
        self.assertEqual(migrated["startup_current"], 0.02)
        self.assertEqual(migrated["t0_current_search_start"], 0.03)
        self.assertEqual(migrated["compliance_voltage"], 30.0)
        self.assertEqual(migrated["max_current_step_up"], 0.002)
        self.assertNotIn("startup_voltage", migrated)
        self.assertNotIn("max_voltage", migrated)
        # The ceiling is a current now, and comes from the defaults.
        self.assertEqual(migrated["max_current"], CONTROL_DEFAULTS["max_current"])

    def test_an_unknown_mode_falls_back_to_the_continuous_measurement(self):
        self.assertEqual(get_resistivity_mode({"resistivity_mode": "nonsense"}), "V_OVER_I")
        self.assertEqual(get_resistivity_mode({"resistivity_mode": "four_wire"}), "FOUR_WIRE")


if __name__ == "__main__":
    unittest.main()
