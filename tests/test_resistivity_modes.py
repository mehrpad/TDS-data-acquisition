import unittest
from unittest.mock import Mock

import numpy as np

from tds_control.tds_experiment import (
    build_control_config,
    get_resistivity_mode,
    measure_resistivity,
    resistivity_loop_time,
)


def _config(**overrides):
    config = build_control_config(
        {
            "max_voltage": 30.0,
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

    def test_an_unknown_mode_falls_back_to_the_continuous_measurement(self):
        self.assertEqual(get_resistivity_mode({"resistivity_mode": "nonsense"}), "V_OVER_I")
        self.assertEqual(get_resistivity_mode({"resistivity_mode": "four_wire"}), "FOUR_WIRE")


if __name__ == "__main__":
    unittest.main()
