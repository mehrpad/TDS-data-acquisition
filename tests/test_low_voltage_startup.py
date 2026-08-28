import unittest
from unittest.mock import Mock, call

import numpy as np

from tds_control import siglent
from tds_control.calibration import _calibrated_temperature_spread, _resistance_series_is_stable
from tds_control.tds_experiment import (
    CONTROL_DEFAULTS,
    _limit_voltage_slew,
    _measurement_voltage_floor,
    _summarize_initial_measurements,
    measure_resistivity,
)


def _config(**overrides):
    config = dict(CONTROL_DEFAULTS)
    config.update(
        min_voltage=0.0,
        max_voltage=1.0,
        max_current=0.1,
        startup_voltage=0.01,
        t0_voltage_search_start=0.01,
        measurement_voltage_floor=0.01,
    )
    config.update(overrides)
    return config


class LinearTemperatureModel:
    def __init__(self, slope=100.0, origin_resistance=19.5, origin_temperature=23.0):
        self.x = np.array([10.0, 30.0])
        self.slope = slope
        self.origin_resistance = origin_resistance
        self.origin_temperature = origin_temperature

    def __call__(self, resistance):
        return self.origin_temperature + self.slope * (float(resistance) - self.origin_resistance)


class LowVoltageStartupTests(unittest.TestCase):
    def test_t0_initial_voltage_is_always_part_of_experiment_floor(self):
        config = _config(startup_voltage=0.01, t0_voltage_search_start=0.02)
        self.assertAlmostEqual(_measurement_voltage_floor(config), 0.02)

    def test_low_voltage_slew_uses_one_millivolt_steps(self):
        config = _config(t0_voltage_search_start=0.01)
        self.assertAlmostEqual(_limit_voltage_slew(0.03, 0.02, 0.01, 1.0, config), 0.021)
        self.assertAlmostEqual(_limit_voltage_slew(0.01, 0.02, 0.01, 1.0, config), 0.019)

    def test_stable_startup_uses_median_resistance(self):
        samples = [
            (0.01830, 0.000938, 19.50),
            (0.01832, 0.000939, 19.52),
            (0.01831, 0.000937, 19.49),
            (0.01833, 0.000940, 19.51),
            (0.01831, 0.000938, 19.50),
            (0.01830, 0.000938, 19.50),
            (0.01832, 0.000939, 19.51),
            (0.01831, 0.000937, 19.49),
            (0.01831, 0.000938, 19.50),
        ]
        summary = _summarize_initial_measurements(samples, LinearTemperatureModel(), _config())
        self.assertIsNotNone(summary)
        self.assertAlmostEqual(summary[2], 23.0)
        self.assertAlmostEqual(summary[3], 19.50)

    def test_single_startup_outlier_is_ignored(self):
        samples = [
            (0.01830, 0.000938, 19.50),
            (0.01832, 0.000939, 19.52),
            (0.01831, 0.000937, 19.49),
            (0.01831, 0.000938, 19.50),
            (0.01830, 0.000938, 19.50),
            (0.01832, 0.000939, 19.51),
            (0.01831, 0.000937, 19.49),
            (0.01831, 0.000938, 19.50),
            (0.02000, 0.000900, 22.00),
        ]
        summary = _summarize_initial_measurements(samples, LinearTemperatureModel(), _config())
        self.assertIsNotNone(summary)
        self.assertAlmostEqual(summary[3], 19.50)

    def test_low_tcr_temperature_spread_is_rejected_even_when_resistance_looks_stable(self):
        samples = [
            (0.01830, 0.000938, resistance)
            for resistance in (19.45, 19.47, 19.49, 19.50, 19.51, 19.53, 19.55, 19.48, 19.52)
        ]
        self.assertIsNone(
            _summarize_initial_measurements(
                samples,
                LinearTemperatureModel(slope=100.0),
                _config(startup_temperature_spread_c=5.0),
            )
        )

    def test_t0_stability_checks_resistance_as_well_as_current(self):
        config = _config()
        self.assertTrue(
            _resistance_series_is_stable([19.50, 19.52, 19.49], config)
        )
        self.assertFalse(
            _resistance_series_is_stable([16.50, 16.83, 17.16], config)
        )

    def test_t0_uncertainty_uses_inverse_calibration_scale(self):
        reference_temperature = lambda resistance: 100.0 * (float(resistance) - 10.0) + 23.0
        spread = _calibrated_temperature_spread(
            [20.00, 20.02],
            scale=2.0,
            reference_temperature_interp=reference_temperature,
        )
        self.assertAlmostEqual(spread, 1.0)

    def test_explicit_fixed_dmm_ranges_override_broad_safety_limits(self):
        dmm = Mock()
        config = _config(
            max_voltage=30.0,
            max_current=3.0,
            dmm_voltage_range_v=0.2,
            dmm_current_range_a=0.2,
        )

        siglent.configure_dc_range_from_config(dmm, "VOLT", config)
        siglent.configure_dc_range_from_config(dmm, "CURR", config)

        self.assertEqual(
            dmm.write.call_args_list,
            [call("CONF:VOLT:DC 0.2"), call("CONF:CURR:DC 0.2")],
        )

    def test_synchronized_pair_starts_both_conversions_before_fetching(self):
        voltage_dmm = Mock()
        current_dmm = Mock()
        voltage_dmm.query.return_value = "0.0195"
        current_dmm.query.return_value = "0.0010"

        measured_voltage, measured_current = siglent.read_DMM_pair(voltage_dmm, current_dmm)

        self.assertEqual((measured_voltage, measured_current), ("0.0195", "0.0010"))
        self.assertEqual(
            voltage_dmm.method_calls,
            [call.write("TRIG:SOUR IMM"), call.write("INIT"), call.query("FETCh?")],
        )
        self.assertEqual(
            current_dmm.method_calls,
            [call.write("TRIG:SOUR IMM"), call.write("INIT"), call.query("FETCh?")],
        )

    def test_measurement_prefers_synchronized_pair_reader(self):
        siglent_module = Mock()
        siglent_module.read_DMM_pair.return_value = (0.0195, 0.0010)

        measured_voltage, measured_current, temperature = measure_resistivity(
            Mock(),
            Mock(),
            siglent_module,
            LinearTemperatureModel(),
            config=_config(),
        )

        self.assertAlmostEqual(measured_voltage, 0.0195)
        self.assertAlmostEqual(measured_current, 0.0010)
        self.assertAlmostEqual(temperature, 23.0)
        siglent_module.read_DMM_pair.assert_called_once()
        siglent_module.read_DMM.assert_not_called()


if __name__ == "__main__":
    unittest.main()
