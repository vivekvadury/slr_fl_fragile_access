"""Regression checks for the model-summary publication gate (no input data needed)."""
import importlib.util
from pathlib import Path
import unittest

import pandas as pd

spec = importlib.util.spec_from_file_location("compare_models", Path(__file__).with_name("07_compare_models.py"))
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


class ModelValidationTests(unittest.TestCase):
    def setUp(self):
        self.prod = pd.DataFrame({"transition": ["Redundant -> Fragile"], "term": ["z_pct_black_nh"],
                                  "estimate": [0.2], "std.error": [0.03]})
        self.est = pd.DataFrame({"rep": ["canonical", "000", "001"],
                                 "transition": ["Redundant -> Fragile"] * 3,
                                 "term": ["z_pct_black_nh"] * 3, "coef": [0.2, 0.21, 0.19],
                                 "coef_se_clustered": [0.03] * 3, "fit_error": [False] * 3})

    def test_complete_reproduced_fits_pass(self):
        compare.validate_estimates(self.est, self.prod, 2)

    def test_failed_fit_stops(self):
        self.est.loc[1, "fit_error"] = True
        with self.assertRaisesRegex(ValueError, "fits failed"):
            compare.validate_estimates(self.est, self.prod, 2)

    def test_missing_replication_stops(self):
        with self.assertRaisesRegex(ValueError, "replications"):
            compare.validate_estimates(self.est.iloc[:2], self.prod, 2)

    def test_incomplete_coefficients_stop(self):
        self.est.loc[1, "term"] = "wrong_term"
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            compare.validate_estimates(self.est, self.prod, 2)

    def test_duplicate_coefficients_stop(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            compare.validate_estimates(pd.concat([self.est, self.est.iloc[:1]]), self.prod, 2)

    def test_nonfinite_fit_stops(self):
        self.est.loc[1, "coef"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            compare.validate_estimates(self.est, self.prod, 2)

    def test_canonical_drift_stops(self):
        for column in ["coef", "coef_se_clustered"]:
            with self.subTest(column=column):
                changed = self.est.copy()
                changed.loc[0, column] += 0.01
                with self.assertRaisesRegex(ValueError, "does not reproduce"):
                    compare.validate_estimates(changed, self.prod, 2)


if __name__ == "__main__":
    unittest.main()
