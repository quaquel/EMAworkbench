"""Tests for optimization functionality."""

import tarfile

import numpy as np
import pandas as pd
import platypus
import pytest
from platypus import AbstractGeneticAlgorithm, Variator

from ema_workbench import (
    BooleanParameter,
    CategoricalParameter,
    Constraint,
    EMAError,
    IntegerParameter,
    RealParameter,
    Sample,
    ScalarOutcome,
    SequentialEvaluator,
)
from ema_workbench.em_framework import Category
from ema_workbench.em_framework.optimization import (
    ArchiveStorageExtension,
    CombinedVariator,
    Problem,
    ProgressBarExtension,
    _evaluate_constraints,
    _optimize,
    epsilon_nondominated,
    evaluate,
    process_jobs,
    rebuild_platypus_population,
    to_dataframe,
)
from ema_workbench.em_framework.points import SampleCollection
from ema_workbench.em_framework.util import ProgressTrackingMixIn

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


def test_progress_bar_extension(mocker):
    """Tests for ProgressBarExtension   ."""
    mocked_pb = mocker.patch(
        "ema_workbench.em_framework.optimization.ProgressTrackingMixIn"
    )
    mocked_mixin = mocker.Mock(spec=ProgressTrackingMixIn)
    mocked_mixin.i = 0
    mocked_pb.return_value = mocked_mixin

    progress_bar = ProgressBarExtension(1000, frequency=100)

    algorithm = mocker.Mock(spec=platypus.EpsNSGAII)
    algorithm.nfe = 100

    progress_bar.do_action(algorithm)

    assert mocked_mixin.call_count == 1
    assert mocked_mixin.call_args_list[0][0] == (100,)


def test_archive_storage_extension(mocker):
    """Tests for ArchiveStorageExtension."""
    mocked_tarfile = mocker.patch("ema_workbench.em_framework.optimization.tarfile")
    mocked_archive = mocker.Mock(spec=tarfile.TarFile)
    mocked_tarfile.open.return_value.__enter__.return_value = mocked_archive

    mocked_to_dataframe = mocker.patch(
        "ema_workbench.em_framework.optimization.to_dataframe"
    )
    mocked_to_dataframe.return_value = pd.DataFrame(
        {"u1": [0, 1], "u2": [1, 2], "o1": [3, 4]}
    )

    decision_variables = ["u1", "u2"]
    outcome_names = ["o1"]
    directory = "."
    filename = "sometarbal.tar.gz"
    frequency = 100
    by_nfe = True

    storage_extension = ArchiveStorageExtension(
        decision_variables,
        outcome_names,
        directory,
        filename,
        frequency=frequency,
        by_nfe=by_nfe,
    )

    algorithm = mocker.Mock(spec=platypus.EpsNSGAII)
    algorithm.nfe = 100
    algorithm.archive = []
    storage_extension.do_action(algorithm)

    assert mocked_archive.addfile.call_count == 1

    # test for algorithms without an archive
    mocked_archive.reset_mock()
    algorithm = mocker.Mock(spec=platypus.NSGAII)
    algorithm.nfe = 100
    algorithm.result = []
    storage_extension.do_action(algorithm)
    assert mocked_archive.addfile.call_count == 1

    # test FileExistError
    mocked_os = mocker.patch("ema_workbench.em_framework.optimization.os")

    mocked_os.path.exists.return_value = True

    with pytest.raises(FileExistsError):
        decision_variables = ["u1", "u2"]
        outcome_names = ["o1"]
        directory = "."
        filename = "sometarbal.tar.gz"
        frequency = 100
        by_nfe = True

        ArchiveStorageExtension(
            decision_variables,
            outcome_names,
            directory,
            filename,
            frequency=frequency,
            by_nfe=by_nfe,
        )


def test_epsilon_dominated():
    """Tests for epsilon_dominated."""
    rng = np.random.default_rng(42)
    n = 10

    results = []
    for _ in range(5):
        results.append(
            pd.DataFrame(
                {
                    "u1": rng.random(n),
                    "o1": rng.random(n),
                    "o2": rng.random(n),
                }
            )
        )

    decision_variables = [
        RealParameter("u1", 0, 1),
    ]
    objectives = [
        ScalarOutcome("o1", kind=ScalarOutcome.MAXIMIZE),
        ScalarOutcome("o2", kind=ScalarOutcome.MAXIMIZE),
    ]
    problem = Problem("uncertainties", decision_variables, objectives)

    dataframe = epsilon_nondominated(results, [0.05, 0.05], problem)
    assert dataframe.columns.tolist() == ["u1", "o1", "o2"]

    with pytest.raises(ValueError):
        epsilon_nondominated(results, [0.05, 0.05, 0.05], problem)


def test_rebuild_platypus_population():
    """Tests for rebuild_platypus_population."""
    rng = np.random.default_rng(42)

    n = 10
    data = pd.DataFrame(
        {"a": rng.random(n), "b": 1 + rng.random(n), "c": 2 + rng.random(n)}
    )

    decision_variables = [
        RealParameter("a", 0, 1),
        RealParameter("b", 1, 2),
    ]
    objectives = [
        ScalarOutcome("c", kind=ScalarOutcome.MAXIMIZE),
    ]

    problem = Problem("uncertainties", decision_variables, objectives)

    population = rebuild_platypus_population(data, problem)
    assert len(population) == n

    for i, row in data.iterrows():
        solution = population[i]
        assert row["a"] == solution.variables[0]
        assert row["b"] == solution.variables[1]
        assert row["c"] == solution.objectives[0]

    # 4 columns expected but data only contains 3
    with pytest.raises(EMAError):
        decision_variables = [
            RealParameter("a", 0, 1),
            RealParameter("b", 1, 2),
            RealParameter("x", 1, 2),
        ]

        problem = Problem("uncertainties", decision_variables, objectives)
        rebuild_platypus_population(data, problem)

    # decision variable in problem but not in data
    with pytest.raises(EMAError):
        decision_variables = [
            RealParameter("a", 0, 1),
            RealParameter("b", 1, 2),
            RealParameter("x", 1, 2),
        ]

        problem = Problem("uncertainties", decision_variables, [])
        rebuild_platypus_population(data, problem)

    # objective in problem but not in data
    with pytest.raises(EMAError):
        objectives = [
            ScalarOutcome("c", kind=ScalarOutcome.MAXIMIZE),
            ScalarOutcome("d", kind=ScalarOutcome.MAXIMIZE),
            ScalarOutcome("e", kind=ScalarOutcome.MAXIMIZE),
        ]

        problem = Problem("uncertainties", [], objectives)
        rebuild_platypus_population(data, problem)


def test_combined_variator(mocker):
    """Tests for CombinedVariator."""
    # this test only ensures everything runs, not
    # that it's correct, but code has been taken from platypus
    mocked_random = mocker.patch("ema_workbench.em_framework.optimization.random")
    mocked_random.random.return_value = 0.1
    mocked_random.choice.return_value = Category("some_value", "some value")

    decision_variables = [
        RealParameter("a", 0, 1),
        IntegerParameter("b", 1, 10),
        CategoricalParameter("c", [1, 6, "c"]),
    ]
    objectives = [
        ScalarOutcome("c", kind=ScalarOutcome.MAXIMIZE),
    ]

    problem = Problem("uncertainties", decision_variables, objectives)
    solution_1 = Sample(a=0.1, b=2, c=1)._to_platypus_solution(problem)
    solution_2 = Sample(a=0.1, b=8, c="c")._to_platypus_solution(problem)

    variator = CombinedVariator()
    s1, s2 = variator.evolve([solution_1, solution_2])

    assert s1.variables[2] == "some value"
    assert s2.variables[2] == "some value"

    variator = CombinedVariator(crossover_prob=0.7)
    mocked_random.random.return_value = 0.6
    variator.evolve([solution_1, solution_2])

    assert s1.variables[2] == "some value"
    assert s2.variables[2] == "some value"


def test_optimize(mocker):
    """Tests for _optimize."""
    mocker.patch(
        "ema_workbench.em_framework.optimization.ArchiveStorageExtension",
        spec=ArchiveStorageExtension,
    )

    # mocking the algorithm
    algorithm = mocker.Mock(spec=AbstractGeneticAlgorithm)
    optimizer = mocker.Mock(spec=AbstractGeneticAlgorithm)
    algorithm.return_value = optimizer
    type(algorithm.return_value).archive = mocker.PropertyMock(return_value=[])
    type(algorithm.return_value).variator = mocker.PropertyMock(return_value=mocker.Mock(spec=Variator))
    type(algorithm.return_value).nfe = mocker.PropertyMock(return_value=100)

    # setup a test problem
    decision_variables = [
        RealParameter("a", 0, 1),
        IntegerParameter("b", 1, 10),
        CategoricalParameter("c", [1, 6, "c"]),
    ]
    objectives = [
        ScalarOutcome("c", kind=ScalarOutcome.MAXIMIZE),
    ]
    problem = Problem("uncertainties", decision_variables, objectives)
    nfe = 1000
    convergence_freq = 100
    logging_freq = 100
    evaluator = mocker.Mock(spec=SequentialEvaluator)

    # the actual call
    _optimize(problem, evaluator, algorithm, nfe, convergence_freq, logging_freq)

    assert isinstance(algorithm.call_args.kwargs["variator"], CombinedVariator)
    assert isinstance(algorithm.call_args.kwargs["generator"], platypus.RandomGenerator)
    assert optimizer.add_extension.call_count == 3
    assert optimizer.run.call_args.args == (nfe,)

    decision_variables = [
        RealParameter("a", 0, 1),
        RealParameter("b", 0, 1),
        RealParameter("c", 0, 1),
    ]
    objectives = [
        ScalarOutcome("c", kind=ScalarOutcome.MAXIMIZE),
    ]
    problem = Problem("uncertainties", decision_variables, objectives)
    _optimize(problem, evaluator, algorithm, nfe, convergence_freq, logging_freq)

    assert algorithm.call_args.kwargs["variator"] is None
    assert isinstance(algorithm.call_args.kwargs["generator"], platypus.RandomGenerator)

    with pytest.raises(ValueError):
        _optimize(
            problem,
            evaluator,
            algorithm,
            nfe,
            convergence_freq,
            logging_freq,
            epsilons=[
                0.1,
                0.1,
            ],
        )

    type(algorithm.return_value).archive = mocker.PropertyMock(
        return_value=[], side_effect=AttributeError()
    )
    type(algorithm.return_value).result = mocker.PropertyMock(return_value=[])
    _optimize(problem, evaluator, algorithm, nfe, convergence_freq, logging_freq)

    initial_population = [Sample(a=0.1, b=0.5, c=0.5) for _ in range(100)]
    _optimize(
        problem,
        evaluator,
        algorithm,
        nfe,
        convergence_freq,
        logging_freq,
        initial_population=initial_population,
    )
    assert isinstance(
        algorithm.call_args.kwargs["generator"], platypus.InjectedPopulation
    )
