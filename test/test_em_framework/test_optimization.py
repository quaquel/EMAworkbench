"""Tests for optimization functionality."""

import numpy as np
import pandas as pd
import pytest

from ema_workbench import (
    BooleanParameter,
    CategoricalParameter,
    Constraint,
    IntegerParameter,
    RealParameter,
    Sample,
    ScalarOutcome,
)
from ema_workbench.em_framework.optimization import (
    Problem,
    _evaluate_constraints,
    evaluate,
    process_jobs,
    to_dataframe,
)
from ema_workbench.em_framework.points import SampleCollection

# Created on 6 Jun 2017
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>


def test_problem(mocker):
    """Test problem class."""
    # evil way to mock super
    searchover = "levers"
    parameters = [
        RealParameter("a", 0, 1),
        IntegerParameter("b", 1, 10),
        BooleanParameter(
            "c",
        ),
        CategoricalParameter("d", ["a", "b", "c"]),
    ]
    outcomes = [
        ScalarOutcome("x", kind=ScalarOutcome.MAXIMIZE),
        ScalarOutcome("y", kind=ScalarOutcome.MAXIMIZE),
    ]

    constraints = [Constraint("n", lambda x: x, parameter_names="n")]

    problem = Problem(searchover, parameters, outcomes, constraints)

    assert searchover == problem.searchover
    assert parameters == problem.decision_variables
    assert outcomes == problem.objectives
    assert constraints == problem.ema_constraints
    assert [c.name for c in constraints] == problem.constraint_names
    assert [o.name for o in outcomes] == problem.outcome_names
    assert [d.name for d in parameters] == problem.parameter_names

    searchover = "uncertainties"
    problem = Problem(searchover, parameters, outcomes, constraints)

    assert searchover == problem.searchover
    assert parameters == problem.decision_variables
    assert outcomes == problem.objectives
    assert constraints == problem.ema_constraints
    assert [c.name for c in constraints] == problem.constraint_names

    parameters = [
        RealParameter("a", 0, 1),
        IntegerParameter("b", 1, 10),
        BooleanParameter(
            "c",
        ),
        CategoricalParameter("d", ["a", "b", "c"]),
    ]

    scenarios = 10
    robustness_functions = [
        ScalarOutcome("x_prime", kind=ScalarOutcome.MAXIMIZE, function=mocker.Mock()),
        ScalarOutcome("y_prime", kind=ScalarOutcome.MAXIMIZE, function=mocker.Mock()),
    ]

    constraints = [Constraint("n", lambda x: x, parameter_names="n")]

    searchover = "robust"
    problem = Problem(
        searchover, parameters, robustness_functions, constraints, scenarios
    )

    assert problem.searchover == "robust"
    assert parameters == problem.decision_variables
    assert robustness_functions == problem.objectives
    assert constraints == problem.ema_constraints
    assert [c.name for c in constraints] == problem.constraint_names

    for scenario in [1, Sample(a=1)]:

        with pytest.raises(ValueError):
            Problem(searchover, parameters, robustness_functions, constraints, scenario)

    with pytest.raises(ValueError):
        robustness_functions[0].kind = 0  # set to  info
        Problem(searchover, parameters, robustness_functions, constraints, 10)


def test_to_dataframe(mocker):
    """Test to_dataframe function."""
    mocked_platypus = mocker.patch("ema_workbench.em_framework.optimization.platypus")
    problem = mocker.Mock()
    type = mocked_platypus.Real  # @ReservedAssignment
    type.decode.return_value = 0
    problem.types = [type, type]
    problem.decision_variables = [RealParameter("a", 0, 1), RealParameter("b", 1, 10)]
    problem.objectives = [
        ScalarOutcome("x", kind=ScalarOutcome.MAXIMIZE),
        ScalarOutcome("y", kind=ScalarOutcome.MAXIMIZE),
    ]

    result1 = mocker.Mock()
    result1.variables = [0, 0]
    result1.objectives = [1, 1]
    result1.problem = problem

    result2 = mocker.Mock()
    result2.variables = [0, 0]
    result2.objectives = [0, 0]
    result2.problem = problem

    data = [result1, result2]

    mocked_platypus.unique.return_value = data
    optimizer = mocker.Mock()
    optimizer.results = data

    dvnames = ["a", "b"]
    outcome_names = ["x", "y"]

    df = to_dataframe(optimizer, dvnames, outcome_names)
    assert list(df.columns.values) == ["a", "b", "x", "y"]

    for i, entry in enumerate(data):
        assert list(df.loc[i, dvnames].values) == entry.variables
        assert list(df.loc[i, outcome_names].values) == entry.objectives


def test_process_jobs(mocker):
    """Test process_jobs function."""
    decision_variables = [
        RealParameter("a", 0, 1),
    ]
    outcomes = [
        ScalarOutcome("x", kind=ScalarOutcome.MAXIMIZE),
        ScalarOutcome("y", kind=ScalarOutcome.MAXIMIZE),
    ]

    problem = Problem("levers", decision_variables, outcomes)

    jobs = []
    for _ in range(10):
        job = mocker.Mock()
        job.solution = mocker.Mock()
        job.solution.problem = problem
        job.solution.variables = [
            0.5,
        ]
        jobs.append(job)

    scenarios, policies = process_jobs(jobs)
    assert scenarios == 1
    assert len(policies) == len(jobs)

    problem = Problem("uncertainties", decision_variables, outcomes)
    jobs = []
    for _ in range(10):
        job = mocker.Mock()
        job.solution = mocker.Mock()
        job.solution.problem = problem
        job.solution.variables = [
            0.5,
        ]
        jobs.append(job)

    scenarios, policies = process_jobs(jobs)
    assert len(scenarios) == len(jobs)
    assert policies == 1

    problem = Problem("robust", decision_variables, outcomes, reference=10)
    jobs = []
    for _ in range(10):
        job = mocker.Mock()
        job.solution = mocker.Mock()
        job.solution.problem = problem
        job.solution.variables = [
            0.5,
        ]
        jobs.append(job)

    scenarios, policies = process_jobs(jobs)
    assert scenarios == 10
    assert len(policies) == 10

    with pytest.raises(ValueError):
        problem = Problem("wrong value", decision_variables, outcomes, reference=10)
        jobs = []
        for _ in range(10):
            job = mocker.Mock()
            job.solution = mocker.Mock()
            job.solution.problem = problem
            job.solution.variables = [
                0.5,
            ]
            jobs.append(job)

        process_jobs(jobs)


def test_evaluate(mocker):
    """Test evaluate function."""
    decision_variables = [
        RealParameter("a", 0, 1),
        RealParameter("b", 0, 1),
    ]
    objectives = [
        ScalarOutcome("x", kind=ScalarOutcome.MAXIMIZE),
        ScalarOutcome("y", kind=ScalarOutcome.MAXIMIZE),
    ]

    rng = np.random.default_rng(42)
    samples = rng.random((10, 2))
    samples_collection = SampleCollection(samples, decision_variables)

    # search over levers
    problem = Problem("levers", decision_variables, objectives)

    jobs = []
    scenario_names = []
    for sample in samples_collection:
        job = mocker.Mock()
        job.solution = sample._to_platypus_solution(problem)
        jobs.append((sample, job))
        scenario_names.append(sample.name)

    experiments = pd.DataFrame(
        {
            "a": samples[:, 0],
            "b": samples[:, 1],
            "scenario": np.zeros(
                10,
            ),
            "policy": scenario_names,
        }
    )
    outcomes = {
        "x": rng.random((10,)),
        "y": rng.random((10,)),
    }

    evaluate(jobs, experiments, outcomes, problem)

    # check if all jobs are evaluated
    for i, (_, job) in enumerate(jobs):
        assert job.solution.evaluated
        assert np.all(
            np.isclose(job.solution.objectives, [v[i] for v in outcomes.values()])
        )

    # search over uncertainties
    problem = Problem("uncertainties", decision_variables, objectives)
    jobs = []
    scenario_names = []
    for sample in samples_collection:
        job = mocker.Mock()
        job.solution = sample._to_platypus_solution(problem)
        jobs.append((sample, job))
        scenario_names.append(sample.name)

    experiments = pd.DataFrame(
        {
            "a": samples[:, 0],
            "b": samples[:, 1],
            "scenario": scenario_names,
            "policy": np.zeros(
                10,
            ),
        }
    )
    outcomes = {
        "x": rng.random((10,)),
        "y": rng.random((10,)),
    }

    evaluate(jobs, experiments, outcomes, problem)

    # check if all jobs are evaluated
    for i, (_, job) in enumerate(jobs):
        assert job.solution.evaluated
        assert np.all(
            np.isclose(job.solution.objectives, [v[i] for v in outcomes.values()])
        )

    # robust optimization
    n_scenarios = 2
    n_solutions = 10

    objectives = [
        ScalarOutcome(
            "x_robust",
            kind=ScalarOutcome.MAXIMIZE,
            variable_name="x",
            function=lambda x: np.mean(x),
        ),
        ScalarOutcome(
            "y_robust",
            kind=ScalarOutcome.MAXIMIZE,
            variable_name="y",
            function=lambda x: np.mean(x),
        ),
    ]

    reference = pd.DataFrame(
        {
            "u_1": rng.random((n_scenarios,)),
            "u_2": rng.random((n_scenarios,)),
            "scenario": np.arange(n_scenarios),
        }
    )
    problem = Problem("robust", decision_variables, objectives, reference=n_scenarios)
    jobs = []
    sample_experiments = []
    for sample in samples_collection:
        job = mocker.Mock()
        job.solution = sample._to_platypus_solution(problem)
        jobs.append((sample, job))

        a = reference.copy()
        a["policy"] = sample.name
        sample_experiments.append(a)

    experiments = pd.concat(sample_experiments)
    outcomes = {
        "x": rng.random((n_scenarios * n_solutions,)),
        "y": rng.random((n_scenarios * n_solutions,)),
    }

    evaluate(jobs, experiments, outcomes, problem)

    # check if all jobs are evaluated
    for _, job in jobs:
        assert job.solution.evaluated


def test_evaluate_constraints():
    """Tests for _evaluate_constraints."""
    experiment = pd.Series({"a": 1, "b": 2})
    outcomes = {"x": 0.5, "y": 2}

    def constraint_func(data):
        return data - 0.5

    constraints = [
        Constraint("c_a", constraint_func, parameter_names="a"),
        Constraint("c_x", constraint_func, outcome_names="x"),
    ]

    scores = _evaluate_constraints(experiment, outcomes, constraints)

    assert scores == [0.5, 0]


# @mock.patch("ema_workbench.em_framework.optimization.platypus")
# def test_to_platypus_types(self, mocked_platypus):
#     dv = [
#         RealParameter("real", 0, 1),
#         IntegerParameter("integer", 0, 10),
#         CategoricalParameter("categorical", ["a", "b"]),
#     ]
#
#     types = to_platypus_types(dv)
#     self.assertTrue(str(types[0]).find("platypus.Real") != -1)
#     self.assertTrue(str(types[1]).find("platypus.Integer") != -1)
#     self.assertTrue(str(types[2]).find("platypus.Subset") != -1)
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
