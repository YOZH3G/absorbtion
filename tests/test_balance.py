import unittest

import numpy as np

from app.calculations import DEFAULT_MODEL_VALUES, absorption_balance


class BalanceTests(unittest.TestCase):
    def test_reference_values_and_mass_conservation(self):
        result = absorption_balance(**DEFAULT_MODEL_VALUES)
        expected = {"j": 400, "gog": 600, "gna": 7800,
                    "xog": 1 / 6, "xna": 0.3,
                    "mass_residual": 0, "component_residual": 0}
        for key, value in expected.items():
            self.assertAlmostEqual(float(result[key]), value)

    def test_no_extraction_and_no_gas_component(self):
        for updates in ({"eta": 0.0}, {"xg": 0.0}):
            values = dict(DEFAULT_MODEL_VALUES, **updates)
            result = absorption_balance(**values)
            self.assertEqual(result["j"], 0)
            self.assertEqual(result["xog"], values["xg"])
            self.assertEqual(result["xna"], values["xa"])

    def test_balances_and_gas_flow_independence_on_varied_inputs(self):
        for xg in (0, 0.1, 0.5, 1):
            for eta in (0, 0.25, 0.8, 1):
                if xg == eta == 1:
                    continue
                first = absorption_balance(1000, xg, 7400, 0.2, eta)
                second = absorption_balance(2500, xg, 10000, 0.7, eta)
                self.assertAlmostEqual(first["xog"], second["xog"])
                for result in (first, second):
                    self.assertAlmostEqual(result["mass_residual"], 0, delta=1e-10)
                    self.assertAlmostEqual(result["component_residual"], 0, delta=1e-10)
                    self.assertTrue(0 <= result["xog"] <= 1)
                    self.assertTrue(0 <= result["xna"] <= 1)

    def test_invalid_inputs_and_empty_gas_outlet_are_rejected(self):
        for updates in ({"xg": 1, "eta": 1}, {"gg": 0}, {"ga": -1},
                        {"xa": 1.1}, {"eta": -0.1}, {"xg": np.nan}, {"gg": np.inf}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                absorption_balance(**dict(DEFAULT_MODEL_VALUES, **updates))

    def test_vectorized_balance_has_identical_scalar_results(self):
        values = dict(DEFAULT_MODEL_VALUES, eta=np.linspace(0, 1, 11))
        result = absorption_balance(**values)
        for index, eta in enumerate(values["eta"]):
            scalar = absorption_balance(**dict(DEFAULT_MODEL_VALUES, eta=eta))
            for key in result:
                self.assertAlmostEqual(result[key][index], scalar[key])
