import unittest

import numpy as np
import pandas as pd

from tds_control.curve_io import _extract_curve_from_frames


class CurveImportTests(unittest.TestCase):
    def test_accepts_saved_experiment_temperature_and_resistance_headers(self):
        frame = pd.DataFrame(
            [
                ["time", "T", "V", "I", "R_ohm"],
                [1.0, 23.0, 0.01, 0.0005, 20.0],
                [2.0, 100.0, 0.02, 0.0008, 25.0],
            ]
        )

        curve, source = _extract_curve_from_frames({"Experiment Data": frame})

        np.testing.assert_allclose(curve, [[20.0, 25.0], [23.0, 100.0]])
        self.assertEqual(source["resistance_column"], "R_ohm")
        self.assertEqual(source["temperature_column"], "T")

    def test_finds_header_after_title_and_uses_common_long_names(self):
        frame = pd.DataFrame(
            [
                ["Corrected wire calibration", None],
                ["Temperature (°C)", "Corrected resistance (Ohm)"],
                [23.0, 19.5],
                [200.0, 24.0],
            ]
        )

        curve, source = _extract_curve_from_frames({"Corrected R vs T": frame})

        np.testing.assert_allclose(curve, [[19.5, 24.0], [23.0, 200.0]])
        self.assertEqual(source["header_row"], 2)

    def test_searches_later_excel_sheets(self):
        unrelated = pd.DataFrame([["notes"], ["nothing here"]])
        calibration = pd.DataFrame(
            [["resistivity", "temperature [C]"], [1.0, 20.0], [2.0, 100.0]]
        )

        curve, source = _extract_curve_from_frames(
            {"Notes": unrelated, "Calibration": calibration}
        )

        np.testing.assert_allclose(curve, [[1.0, 2.0], [20.0, 100.0]])
        self.assertEqual(source["sheet"], "Calibration")

    def test_rejects_unlabeled_numeric_columns_instead_of_guessing(self):
        frame = pd.DataFrame([["x", "y"], [1.0, 20.0], [2.0, 100.0]])

        with self.assertRaisesRegex(ValueError, "No usable resistance/resistivity"):
            _extract_curve_from_frames({"Sheet1": frame})


if __name__ == "__main__":
    unittest.main()
