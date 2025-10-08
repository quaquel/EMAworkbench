"""Tests for optimization functionality."""

from ema_workbench.em_framework.optimization import (
    Problem,
)
from ema_workbench import Constraint, ScalarOutcome, RealParameter, IntegerParameter, BooleanParameter, CategoricalParameter

# Created on 6 Jun 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>


def test_problem(mocker):
    """Test problem class."""
    # evil way to mock super
    searchover = "levers"
    parameters = [RealParameter("a", 0, 1),
                  IntegerParameter("b", 1, 10),
                  BooleanParameter("c", ),
                  CategoricalParameter("d", ["a", "b", "c"]),]
    outcomes = [
        ScalarOutcome("x", kind=ScalarOutcome.MAXIMIZE),
        ScalarOutcome("y", kind=ScalarOutcome.MAXIMIZE),
    ]

    constraints = [Constraint("n", function=lambda x: x)]

    problem = Problem(searchover, parameters, outcomes, constraints)

    assert searchover == problem.searchover
    assert parameters == problem.decision_variables
    assert outcomes == problem.objectives
    assert constraints == problem.ema_constraints
    assert [c.name for c in constraints] == problem.constraint_names

    searchover = "uncertainties"
    problem = Problem(searchover, parameters, outcomes, constraints)

    assert searchover == problem.searchover
    assert parameters == problem.decision_variables
    assert outcomes == problem.objectives
    assert constraints == problem.ema_constraints
    assert [c.name for c in constraints] == problem.constraint_names

    """Test robust problem class."""
    parameters = [RealParameter("a", 0, 1),
                  IntegerParameter("b", 1, 10),
                  BooleanParameter("c", ),
                  CategoricalParameter("d", ["a", "b", "c"]),]

    scenarios = 10
    robustness_functions = [
        ScalarOutcome("x_prime", kind=ScalarOutcome.MAXIMIZE, function=mocker.Mock()),
        ScalarOutcome("y_prime", kind=ScalarOutcome.MAXIMIZE, function=mocker.Mock()),
    ]

    constraints = [Constraint("n", function=lambda x: x)]

    searchover = "robust"
    problem = Problem(searchover, parameters, robustness_functions, constraints, 10)

    assert problem.searchover == "robust"
    assert parameters == problem.decision_variables
    assert robustness_functions == problem.objectives
    assert constraints == problem.ema_constraints
    assert [c.name for c in constraints] == problem.constraint_names


# class TestOptimization(unittest.TestCase):
#     @mock.patch("ema_workbench.em_framework.optimization.platypus")
#     def test_to_dataframe(self, mocked_platypus):
#         problem = mock.Mock()
#         type = mocked_platypus.Real  # @ReservedAssignment
#         type.decode.return_value = 0
#         problem.types = [type, type]
#
#         result1 = mock.Mock()
#         result1.variables = [0, 0]
#         result1.objectives = [1, 1]
#         result1.problem = problem
#
#         result2 = mock.Mock()
#         result2.variables = [0, 0]
#         result2.objectives = [0, 0]
#         result2.problem = problem
#
#         data = [result1, result2]
#
#         mocked_platypus.unique.return_value = data
#         optimizer = mock.Mock()
#         optimizer.results = data
#
#         dvnames = ["a", "b"]
#         outcome_names = ["x", "y"]
#
#         df = to_dataframe(optimizer, dvnames, outcome_names)
#         self.assertListEqual(list(df.columns.values), ["a", "b", "x", "y"])
#
#         for i, entry in enumerate(data):
#             self.assertListEqual(list(df.loc[i, dvnames].values), entry.variables)
#             self.assertListEqual(
#                 list(df.loc[i, outcome_names].values), entry.objectives
#             )
#
#     @mock.patch("ema_workbench.em_framework.optimization.platypus")
#     def test_to_platypus_types(self, mocked_platypus):
#         dv = [
#             RealParameter("real", 0, 1),
#             IntegerParameter("integer", 0, 10),
#             CategoricalParameter("categorical", ["a", "b"]),
#         ]
#
#         types = to_platypus_types(dv)
#         self.assertTrue(str(types[0]).find("platypus.Real") != -1)
#         self.assertTrue(str(types[1]).find("platypus.Integer") != -1)
#         self.assertTrue(str(types[2]).find("platypus.Subset") != -1)
#
#     @mock.patch("ema_workbench.em_framework.optimization.platypus")
#     def test_to_problem(self, mocked_platypus):
#         mocked_model = Model("test", function=mock.Mock())
#         mocked_model.levers = [RealParameter("a", 0, 1), RealParameter("b", 0, 1)]
#         mocked_model.uncertainties = [
#             RealParameter("c", 0, 1),
#             RealParameter("d", 0, 1),
#         ]
#         mocked_model.outcomes = [ScalarOutcome("x", kind=1), ScalarOutcome("y", kind=1)]
#
#         searchover = "levers"
#         problem = to_problem(mocked_model, searchover)
#         assert searchover, problem.searchover)
#
#         for entry in problem.parameters:
#             self.assertIn(entry.name, mocked_model.levers.keys())
#             self.assertIn(entry, list(mocked_model.levers))
#         for entry in problem.outcome_names:
#             self.assertIn(entry, mocked_model.outcomes.keys())
#
#         searchover = "uncertainties"
#         problem = to_problem(mocked_model, searchover)
#
#         assert searchover, problem.searchover)
#         for entry in problem.parameters:
#             self.assertIn(entry.name, mocked_model.uncertainties.keys())
#             self.assertIn(entry, list(mocked_model.uncertainties))
#         for entry in problem.outcome_names:
#             self.assertIn(entry, mocked_model.outcomes.keys())
#
#     def test_process_levers(self):
#         pass
#
#     def test_process_uncertainties(self):
#         pass
#
#
# class TestRobustOptimization(unittest.TestCase):
#     @mock.patch("ema_workbench.em_framework.optimization.platypus")
#     def test_to_robust_problem(self, mocked_platypus):
#         mocked_model = Model("test", function=mock.Mock())
#         mocked_model.levers = [RealParameter("a", 0, 1), RealParameter("b", 0, 1)]
#         mocked_model.uncertainties = [
#             RealParameter("c", 0, 1),
#             RealParameter("d", 0, 1),
#         ]
#         mocked_model.outcomes = [ScalarOutcome("x"), ScalarOutcome("y")]
#
#         scenarios = 5
#         robustness_functions = [
#             ScalarOutcome(
#                 "mean_x", variable_name="x", function=mock.Mock(), kind="maximize"
#             ),
#             ScalarOutcome(
#                 "mean_y", variable_name="y", function=mock.Mock(), kind="maximize"
#             ),
#         ]
#
#         problem = to_robust_problem(mocked_model, scenarios, robustness_functions)
#
#         assert "robust", problem.searchover)
#         for entry in problem.parameters:
#             self.assertIn(entry.name, mocked_model.levers.keys())
#         assert ["mean_x", "mean_y"], problem.outcome_names)
#
#     def test_process_robust(self):
#         pass
