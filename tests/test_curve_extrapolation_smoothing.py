import unittest

import numpy as np

from tds_control.calibration import _scale_and_anchor_curve
from tds_control.tds_experiment import (
    CONTROL_DEFAULTS,
    _extend_curve_for_configured_extrapolation,
    _validate_temperature_program_bounds,
    build_temperature_interpolator,
)


def _config(**overrides):
    config = dict(CONTROL_DEFAULTS)
    config.update(overrides)
    return config


class CurveExtrapolationSmoothingTests(unittest.TestCase):
    def test_extrapolation_is_disabled_by_default(self):
        curve = np.array(
            [
                [1.0, 2.0, 3.0],
                [25.0, 100.0, 300.0],
            ]
        )

        unchanged, bounds, source_bounds = _extend_curve_for_configured_extrapolation(
            curve,
            _config(),
        )

        np.testing.assert_array_equal(unchanged, curve)
        self.assertEqual(bounds, (25.0, 300.0))
        self.assertEqual(source_bounds, (25.0, 300.0))

    def test_disabled_extrapolation_rejects_target_outside_measured_curve(self):
        curve = np.array(
            [
                [1.0, 2.0, 3.0],
                [25.0, 100.0, 300.0],
            ]
        )
        model = build_temperature_interpolator(curve, _config())
        experiment_params = [{"start_T": 25.0, "target_T": 500.0}]

        with self.assertRaisesRegex(ValueError, "outside the allowed R vs. T conversion range"):
            _validate_temperature_program_bounds(experiment_params, model)

    def test_t0_anchor_extends_only_to_measured_room_temperature(self):
        curve = np.array(
            [
                [10.0, 20.0, 30.0],
                [25.0, 100.0, 300.0],
            ]
        )
        calibrated = _scale_and_anchor_curve(
            curve,
            scale=2.0,
            anchor_temperature=23.0,
            anchor_resistance=19.5,
        )
        model = build_temperature_interpolator(calibrated, _config())

        self.assertEqual(model.temperature_bounds, (23.0, 300.0))
        self.assertAlmostEqual(float(model(19.5)), 23.0)
        _validate_temperature_program_bounds(
            [{"start_T": 23.0, "target_T": 300.0}],
            model,
        )
        with self.assertRaisesRegex(ValueError, "outside the allowed R vs. T conversion range"):
            _validate_temperature_program_bounds(
                [{"start_T": 23.0, "target_T": 500.0}],
                model,
            )

    def test_dense_noisy_curve_is_smoothed_and_extrapolated(self):
        temperatures = np.linspace(25.0, 300.0, 2400)
        trend = 20.0 + 0.012 * temperatures
        noise = 0.025 * np.sin(temperatures * 3.1) + 0.012 * np.sin(temperatures * 17.0)
        resistances = trend + noise
        resistances[::173] += 0.20
        resistances[71::211] -= 0.18
        curve = np.vstack((resistances, temperatures))

        extended, bounds, source_bounds = _extend_curve_for_configured_extrapolation(
            curve,
            _config(curve_extrapolation_enabled=True),
        )

        self.assertEqual(bounds, (0.0, 600.0))
        self.assertGreater(source_bounds[0], 24.0)
        self.assertLess(source_bounds[1], 301.0)
        self.assertLess(extended.shape[1], curve.shape[1])
        self.assertTrue(np.all(np.diff(extended[0, :]) >= -1e-12))

        model = build_temperature_interpolator(curve, _config(curve_extrapolation_enabled=True))
        resistance_600 = float(np.interp(600.0, extended[1, :], extended[0, :]))
        self.assertAlmostEqual(float(model(resistance_600)), 600.0, places=6)

    def test_dense_noisy_curve_is_smoothed_without_extrapolation(self):
        temperatures = np.linspace(25.0, 300.0, 2400)
        trend = 20.0 + 0.012 * temperatures
        noise = 0.025 * np.sin(temperatures * 3.1) + 0.012 * np.sin(temperatures * 17.0)
        resistances = trend + noise
        resistances[::173] += 0.20
        resistances[71::211] -= 0.18
        curve = np.vstack((resistances, temperatures))

        conditioned, bounds, source_bounds = _extend_curve_for_configured_extrapolation(
            curve,
            _config(curve_extrapolation_enabled=False),
        )

        self.assertEqual(bounds, source_bounds)
        self.assertGreater(bounds[0], 24.0)
        self.assertLess(bounds[1], 301.0)
        self.assertLess(conditioned.shape[1], curve.shape[1])
        self.assertTrue(np.all(np.diff(conditioned[0, :]) >= -1e-12))

    def test_sparse_curve_with_large_reversal_is_still_rejected(self):
        curve = np.array(
            [
                [1.0, 1.4, 1.1, 1.8, 2.2],
                [20.0, 50.0, 80.0, 120.0, 160.0],
            ]
        )

        with self.assertRaisesRegex(ValueError, "resistance reversals"):
            _extend_curve_for_configured_extrapolation(
                curve,
                _config(curve_extrapolation_enabled=True),
            )

    def test_smoothing_can_be_disabled_for_strict_validation(self):
        temperatures = np.linspace(20.0, 200.0, 400)
        resistances = 10.0 + 0.01 * temperatures + 0.08 * np.sin(temperatures)
        curve = np.vstack((resistances, temperatures))

        with self.assertRaisesRegex(ValueError, "resistance reversals"):
            _extend_curve_for_configured_extrapolation(
                curve,
                _config(curve_extrapolation_enabled=True, curve_smoothing_enabled=False),
            )


if __name__ == "__main__":
    unittest.main()
