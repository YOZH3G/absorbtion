import unittest

import numpy as np

from app.calculations import DEFAULT_MODEL_VALUES, absorption_balance
from app.simulation import (LEAN_GAS, RICH_ABSORBENT, eta_gain,
                            run_simulation, tune_balanced_controller)


DYNAMICS = dict(kind="step", start_time=10.0, effect_duration=10.0,
                simulation_duration=150.0, time_constant=10.0, delay=2.0)


def controller(chain, kind="PI", **updates):
    tuning = tune_balanced_controller(chain, DEFAULT_MODEL_VALUES, kind, 10, 2)
    values = dict(controller_type=kind, controller_gain=tuning["proportional_gain"],
                  integral_time=tuning["integral_time"] or 1.0,
                  derivative_time=tuning["derivative_time"] or 0.0,
                  control_limit=1.0, setpoint=1 / 6 if chain == LEAN_GAS else 0.3)
    return dict(values, **updates)


class EtaControllerTests(unittest.TestCase):
    def test_local_gain_matches_independent_finite_difference(self):
        for chain, output in ((LEAN_GAS, "xog"), (RICH_ABSORBENT, "xna")):
            high = absorption_balance(**dict(DEFAULT_MODEL_VALUES, eta=0.800001))[output]
            low = absorption_balance(**dict(DEFAULT_MODEL_VALUES, eta=0.799999))[output]
            self.assertAlmostEqual(eta_gain(chain, DEFAULT_MODEL_VALUES), (high - low) / 2e-6, places=8)

    def test_all_autotuned_types_reduce_error_for_both_outputs(self):
        for chain in (LEAN_GAS, RICH_ABSORBENT):
            uncontrolled = run_simulation(chain, DEFAULT_MODEL_VALUES, 0.1, 0, DYNAMICS)
            for kind in ("P", "PI", "PD", "PID"):
                with self.subTest(chain=chain, kind=kind):
                    closed = run_simulation(chain, DEFAULT_MODEL_VALUES, 0.1, 0,
                                            DYNAMICS, controller(chain, kind))
                    self.assertLess(abs(closed["error"][-1]),
                                    abs(closed["baseline"] - uncontrolled["final_response"][-1]))
                    self.assertTrue(np.all(np.isfinite(closed["control"])))
                    self.assertTrue(np.all((closed["control"] >= 0) & (closed["control"] <= 1)))
                    if "I" in kind:
                        self.assertLess(abs(closed["error"][-1]), 1e-5)

    def test_unreachable_liquid_assignment_saturates_without_fabricating_target(self):
        result = run_simulation(RICH_ABSORBENT, DEFAULT_MODEL_VALUES, 0, 0, DYNAMICS,
                                controller(RICH_ABSORBENT, setpoint=0.31))
        self.assertFalse(result["setpoint_reachable"])
        self.assertAlmostEqual(result["control"][-1], 1)
        self.assertAlmostEqual(result["metrics"]["steady_state"], 0.30886075949367087)
        self.assertGreater(result["metrics"]["static_error"], 0)

    def test_saturation_recovers_after_temporary_disturbance(self):
        dynamics = dict(DYNAMICS, kind="rectangle", effect_duration=80.0,
                        simulation_duration=300.0)
        result = run_simulation(LEAN_GAS, DEFAULT_MODEL_VALUES, -0.8, 0, dynamics,
                                controller(LEAN_GAS))
        self.assertTrue(np.any(result["control"] == 0))
        self.assertAlmostEqual(result["control"][-1], 0.8, delta=1e-4)
        self.assertLess(abs(result["error"][-1]), 1e-5)

    def test_zero_gain_matches_free_reaction_and_keeps_eta_bias(self):
        for chain in (LEAN_GAS, RICH_ABSORBENT):
            free = run_simulation(chain, DEFAULT_MODEL_VALUES, 0.1, 0, DYNAMICS)
            closed = run_simulation(chain, DEFAULT_MODEL_VALUES, 0.1, 0, DYNAMICS,
                                    controller(chain, controller_gain=0))
            np.testing.assert_array_equal(closed["control"], np.full_like(closed["time"], 0.8))
            np.testing.assert_allclose(closed["final_response"], np.interp(
                closed["time"], free["time"], free["final_response"]), atol=2e-6)

    def test_p_and_pd_final_outputs_match_nonlinear_equilibrium(self):
        for chain in (LEAN_GAS, RICH_ABSORBENT):
            for kind in ("P", "PD"):
                result = run_simulation(chain, DEFAULT_MODEL_VALUES, 0.1, 0, DYNAMICS,
                                        controller(chain, kind))
                self.assertAlmostEqual(result["final_response"][-1],
                                       result["metrics"]["steady_state"], delta=1e-6)

    def test_no_control_authority_prevents_autotuning(self):
        for chain, updates in ((LEAN_GAS, {"xg": 0}), (RICH_ABSORBENT, {"xa": 1})):
            with self.assertRaisesRegex(ValueError, "не зависит"):
                tune_balanced_controller(chain, dict(DEFAULT_MODEL_VALUES, **updates), "PI", 10, 2)

    def test_invalid_actuator_bounds_and_concentrations_are_rejected(self):
        for updates in ({"control_limit": 0}, {"control_limit": 1.1}, {"setpoint": 1.1},
                        {"controller_gain": -1}, {"integral_time": 0}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                run_simulation(LEAN_GAS, DEFAULT_MODEL_VALUES, 0.1, 0, DYNAMICS,
                               controller(LEAN_GAS, **updates))
        with self.assertRaisesRegex(ValueError, "не определена"):
            run_simulation(RICH_ABSORBENT, dict(DEFAULT_MODEL_VALUES, xg=1), 0, 0,
                           DYNAMICS, controller(RICH_ABSORBENT))

    def test_gas_flow_disturbance_changes_liquid_but_not_gas(self):
        result = run_simulation(LEAN_GAS, DEFAULT_MODEL_VALUES, 0, 0.3, DYNAMICS)
        np.testing.assert_allclose(result["phase_responses"]["xog"], 1 / 6, atol=1e-15)
        self.assertGreater(result["phase_responses"]["xna"][-1], 0.3)

    def test_delayed_eta_step_moves_both_outputs_in_opposite_directions(self):
        result = run_simulation(LEAN_GAS, DEFAULT_MODEL_VALUES, 0, 0, DYNAMICS,
                                controller(LEAN_GAS, setpoint=0.15))
        for key, baseline in (("xog", 1 / 6), ("xna", 0.3)):
            np.testing.assert_allclose(result["phase_responses"][key][result["time"] <= 2], baseline)
        self.assertLess(result["phase_responses"]["xog"][-1], 1 / 6)
        self.assertGreater(result["phase_responses"]["xna"][-1], 0.3)
