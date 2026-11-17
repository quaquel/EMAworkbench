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
from ema_workbench.em_framework.optimization import (
    ArchiveStorageExtension,
    CombinedVariator,
    GenerationalBorg,
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
    assert parameters == list(problem.decision_variables)
    assert outcomes == problem.objectives
    assert constraints == problem.ema_constraints
    assert [c.name for c in constraints] == problem.constraint_names
    assert [o.name for o in outcomes] == problem.outcome_names
    assert [d.name for d in parameters] == problem.parameter_names

    searchover = "uncertainties"
    problem = Problem(searchover, parameters, outcomes, constraints)

    assert searchover == problem.searchover
    assert parameters == list(problem.decision_variables)
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
    assert parameters == list(problem.decision_variables)
    assert robustness_functions == problem.objectives
    assert constraints == problem.ema_constraints
    assert [c.name for c in constraints] == problem.constraint_names

    for scenario in [1, Sample(a=1)]:
        with pytest.raises(ValueError):
            Problem(searchover, parameters, robustness_functions, constraints, scenario)

    with pytest.raises(ValueError):
        robustness_functions[0].kind = 0  # set to  info
        Problem(searchover, parameters, robustness_functions, constraints, 10)


def test_optimize_uses_fixed_reference(tmp_path):
    """Regression test: all solutions are evaluated on the same scenario(s).

    Previously, without a reference, optimize sampled a new random scenario
    for every generation, and robust_optimize with a number of scenarios
    sampled new scenarios for every generation.
    """
    from ema_workbench import Model

    seen = []

    def some_model(x=0.0, u=0.5):
        seen.append(u)
        return {"y": x + u}

    model = Model("simple", function=some_model)
    model.levers = [RealParameter("x", 0, 1)]
    model.uncertainties = [RealParameter("u", 0, 100)]
    model.outcomes = [ScalarOutcome("y", kind=ScalarOutcome.MAXIMIZE)]

    # no reference: default values of the model
    with SequentialEvaluator(model) as evaluator:
        results, _ = evaluator.optimize(
            nfe=50,
            searchover="levers",
            epsilons=[0.1],
            population_size=10,
            filename="archive.tar.gz",
            directory=tmp_path,
            rng=42,
        )
    assert set(seen) == {0.5}
    np.testing.assert_allclose(results["y"], results["x"] + 0.5)

    # explicit reference
    seen.clear()
    with SequentialEvaluator(model) as evaluator:
        evaluator.optimize(
            nfe=50,
            searchover="levers",
            epsilons=[0.1],
            population_size=10,
            reference=Sample("ref", u=10),
            filename="archive_ref.tar.gz",
            directory=tmp_path,
            rng=42,
        )
    assert set(seen) == {10}

    # robust optimization with a number of scenarios
    seen.clear()
    robustness_functions = [
        ScalarOutcome(
            "mean_y", kind=ScalarOutcome.MAXIMIZE, variable_name="y", function=np.mean
        )
    ]
    with SequentialEvaluator(model) as evaluator:
        evaluator.robust_optimize(
            robustness_functions,
            4,
            nfe=50,
            population_size=10,
            epsilons=[0.1],
            filename="archive_robust.tar.gz",
            directory=tmp_path,
            rng=42,
        )
    assert len(set(seen)) == 4


def test_problem_robust_requires_scenarios():
    """Robust optimization without scenarios is not possible."""
    decision_variables = [RealParameter("a", 0, 1)]
    objectives = [ScalarOutcome("x", kind=ScalarOutcome.MAXIMIZE)]
    with pytest.raises(ValueError):
        Problem("robust", decision_variables, objectives)


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
    assert isinstance(scenarios, Sample)
    assert len(scenarios) == 0
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
    assert isinstance(policies, Sample)
    assert len(policies) == 0

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
            function=np.mean,
        ),
        ScalarOutcome(
            "y_robust",
            kind=ScalarOutcome.MAXIMIZE,
            variable_name="y",
            function=np.mean,
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


def test_robust_optimize_scores_each_policy_separately(tmp_path):
    """Regression test: robustness must be computed per policy.

    Previously, the robustness function was applied to the outcomes of all
    experiments in a generation, so every policy got the same score.
    """
    from ema_workbench import Model

    def some_model(x=0.0, u=0.0):
        return {"y": x + u}

    model = Model("simple", function=some_model)
    model.levers = [RealParameter("x", 0, 10)]
    model.uncertainties = [RealParameter("u", 0, 1)]
    model.outcomes = [ScalarOutcome("y")]

    u_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    scenarios = [Sample(name=f"{i}", u=u) for i, u in enumerate(u_values)]
    robustness_functions = [
        ScalarOutcome(
            "mean_y", kind=ScalarOutcome.MAXIMIZE, variable_name="y", function=np.mean
        )
    ]

    with SequentialEvaluator(model) as evaluator:
        results, _ = evaluator.robust_optimize(
            robustness_functions,
            scenarios,
            nfe=200,
            population_size=10,
            epsilons=[0.01],
            filename="archive.tar.gz",
            directory=tmp_path,
            rng=42,
        )

    # the score of each policy should be the mean over the scenarios for that
    # policy only, i.e., x + mean(u)
    expected = results["x"] + np.mean(u_values)
    np.testing.assert_allclose(results["mean_y"], expected)

    # and maximizing should push x toward its upper bound
    assert results["x"].max() > 8


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


@pytest.mark.parametrize("searchover", ["levers", "uncertainties"])
def test_optimize_with_constraint_on_parameter(tmp_path, searchover):
    """Regression test: a constraint on a decision variable.

    Previously, the constraint function received a pandas Series instead of
    a single value when searching over levers or uncertainties.
    """
    from ema_workbench import Model

    def some_model(x=0.0, z=0.0):
        return {"y": x + z}

    model = Model("simple", function=some_model)
    parameters = [RealParameter("x", 0, 1), RealParameter("z", 0, 1)]
    if searchover == "levers":
        model.levers = parameters
    else:
        model.uncertainties = parameters
    model.outcomes = [ScalarOutcome("y", kind=ScalarOutcome.MAXIMIZE)]

    received = []

    def constraint_function(x):
        received.append(x)
        return max(0, x - 0.5)

    constraints = [
        Constraint("constraint_on_x", parameter_names="x", function=constraint_function)
    ]

    with SequentialEvaluator(model) as evaluator:
        results, _ = evaluator.optimize(
            nfe=500,
            searchover=searchover,
            epsilons=[0.01],
            population_size=20,
            constraints=constraints,
            reference=Sample("reference"),
            filename="archive.tar.gz",
            directory=tmp_path,
            rng=42,
        )

    assert all(isinstance(value, float) for value in received)
    # all solutions meet the constraint, and the constraint is binding
    assert (results["x"] <= 0.5).all()
    assert results["y"].max() > 1.4


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
    # the compressed tarball is written to a temporary file that replaces the tarball
    mocker.patch("ema_workbench.em_framework.optimization.os.replace")

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


def test_final_archive_is_stored_once(tmp_path):
    """Regression test: the last archive is not stored twice.

    Previously, the archive and the convergence information were stored again
    after the optimization, also if they had just been stored because the number of function evaluations was a
    multiple of the convergence frequency.
    """
    from ema_workbench import Model
    from ema_workbench.em_framework.optimization import load_archives

    def some_model(x=0.0):
        return {"archive_test_y": x}

    model = Model("archivemodel", function=some_model)
    model.levers = [RealParameter("x", 0, 1)]
    model.outcomes = [ScalarOutcome("archive_test_y", kind=ScalarOutcome.MAXIMIZE)]

    for nfe, expected in [(100, [50, 100]), (70, [50, 70])]:
        filename = f"archive_{nfe}.tar.gz"
        with SequentialEvaluator(model) as evaluator:
            _, convergence = evaluator.optimize(
                nfe=nfe,
                searchover="levers",
                epsilons=[0.1],
                population_size=10,
                convergence_freq=50,
                reference=Sample("reference"),
                filename=filename,
                directory=tmp_path,
                rng=42,
            )

        archives = load_archives(tmp_path / filename)
        assert [entry[0] for entry in archives] == expected
        assert convergence["nfe"].tolist() == expected


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


@pytest.mark.parametrize(
    "filename, magic",
    [
        ("archives.tar.gz", b"\x1f\x8b"),
        ("archives.tgz", b"\x1f\x8b"),
        ("archives.tar.bz2", b"BZh"),
        ("archives.tar.xz", b"\xfd7zXZ"),
        ("archives.tar", None),
    ],
)
def test_archive_tarball_is_compressed(tmp_path, filename, magic):
    """Regression test: the tarball is compressed as its extension indicates.

    Previously, the tarball was never compressed, also not if the filename
    ended with .tar.gz.
    """
    from ema_workbench import Model
    from ema_workbench.em_framework.optimization import load_archives

    def some_model(x=0.0):
        return {"compression_test_y": x}

    model = Model("compressionmodel", function=some_model)
    model.levers = [RealParameter("x", 0, 1)]
    model.outcomes = [ScalarOutcome("compression_test_y", kind=ScalarOutcome.MAXIMIZE)]

    with SequentialEvaluator(model) as evaluator:
        results, _ = evaluator.optimize(
            nfe=95,
            searchover="levers",
            epsilons=[0.1],
            population_size=10,
            convergence_freq=30,
            reference=Sample("reference"),
            filename=filename,
            directory=tmp_path,
            rng=42,
        )

    # only the tarball is left in the directory
    assert [p.name for p in tmp_path.iterdir()] == [filename]

    with open(tmp_path / filename, "rb") as fh:
        start = fh.read(6)
    if magic is None:
        with tarfile.open(tmp_path / filename, "r:") as archive:
            assert len(archive.getnames()) == 4
    else:
        assert start.startswith(magic)

    # all intermediate archives and the final archive are stored, in order
    archives = load_archives(tmp_path / filename)
    assert [nfe for nfe, _ in archives] == [30, 60, 90, 100]
    for _, archive in archives:
        assert list(archive.columns) == ["x", "compression_test_y"]
    pd.testing.assert_frame_equal(archives[-1][1], results)


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


def test_combined_variator_default_mutation():
    """Regression test: by default, mutation changes only part of a solution.

    Previously, every decision variable was mutated in every child, and all
    bits of an integer were flipped, so children never kept the integer
    value of their parents.
    """
    import copy
    import random

    random.seed(42)

    decision_variables = [RealParameter("a", 0, 1), IntegerParameter("i", 0, 15)]
    objectives = [ScalarOutcome("y", kind=ScalarOutcome.MAXIMIZE)]
    problem = Problem("levers", decision_variables, objectives)

    parent = Sample(a=0.5, i=5)._to_platypus_solution(problem)
    variator = CombinedVariator()

    children = []
    for _ in range(200):
        for child in variator.evolve([parent, copy.deepcopy(parent)]):
            children.append(Sample._from_platypus_solution(child))

    share_same_integer = sum(c["i"] == 5 for c in children) / len(children)
    share_same_real = sum(c["a"] == 0.5 for c in children) / len(children)

    # with identical parents, crossover changes nothing, and on average a
    # single decision variable is mutated
    assert 0.3 < share_same_integer < 0.95
    assert 0.3 < share_same_real < 0.95
    assert len({int(c["i"]) for c in children}) > 2

    # explicit probability of 1 mutates all real decision variables
    variator = CombinedVariator(mutation_prob=1)
    children = [
        Sample._from_platypus_solution(child)
        for _ in range(20)
        for child in variator.evolve([parent, copy.deepcopy(parent)])
    ]
    assert all(c["a"] != 0.5 for c in children)


def test_optimize_mixed_real_and_integer_levers(tmp_path):
    """Regression test: optimization with mixed lever types converges."""
    from ema_workbench import Model

    def some_model(a=0.0, i=0, j=0):
        return {"y": -abs(i - 37) - abs(j - 11) - abs(a - 0.3)}

    model = Model("simple", function=some_model)
    model.levers = [
        RealParameter("a", 0, 1),
        IntegerParameter("i", 0, 63),
        IntegerParameter("j", 0, 63),
    ]
    model.outcomes = [ScalarOutcome("y", kind=ScalarOutcome.MAXIMIZE)]

    with SequentialEvaluator(model) as evaluator:
        results, _ = evaluator.optimize(
            nfe=2000,
            searchover="levers",
            epsilons=[0.05],
            population_size=50,
            reference=Sample("reference"),
            filename="archive.tar.gz",
            directory=tmp_path,
            rng=42,
        )

    # the optimum is 0, the old variator got stuck around -4 or worse
    assert results["y"].max() > -1


def test_combined_variator(mocker):
    """Tests for CombinedVariator."""
    # this test only ensures everything runs, not
    # that it's correct, but code has been taken from platypus
    mocked_random = mocker.patch("ema_workbench.em_framework.optimization.random")
    mocked_random.random.return_value = 0.1
    mocked_random.choice.return_value = 2

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

    assert s1.variables[2] == [2]
    assert s2.variables[2] == [2]

    variator = CombinedVariator(crossover_prob=0.7)
    mocked_random.random.return_value = 0.6
    variator.evolve([solution_1, solution_2])

    assert s1.variables[2] == [2]
    assert s2.variables[2] == [2]


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
    type(algorithm.return_value).variator = mocker.PropertyMock(
        return_value=mocker.Mock(spec=Variator)
    )
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

    mocker.patch.object(GenerationalBorg, "run")
    _optimize(
        problem,
        evaluator,
        GenerationalBorg,
        nfe,
        convergence_freq,
        logging_freq,
        epsilons=[
            0.1,
        ],
    )


def test_sample_platypus_round_trip_categorical():
    """Regression test: categorical and boolean values survive a round trip."""
    decision_variables = [
        RealParameter("a", 0, 1),
        CategoricalParameter("c", [1, 6, "c"]),
        BooleanParameter("d"),
    ]
    objectives = [ScalarOutcome("y", kind=ScalarOutcome.MAXIMIZE)]
    problem = Problem("levers", decision_variables, objectives)

    for c, d in [(1, False), (6, True), ("c", True)]:
        sample = Sample(a=0.5, c=c, d=d)
        solution = sample._to_platypus_solution(problem)
        # platypus Subset of size 1 over the category indices
        category_values = [cat.value for cat in decision_variables[1].categories]
        assert solution.variables[1] == [category_values.index(c)]
        decoded = Sample._from_platypus_solution(solution)
        assert decoded["a"] == 0.5
        assert decoded["c"] == c
        assert decoded["d"] == d


def test_mutate_categorical():
    """Regression test: mutation yields a different category index as a list."""
    decision_variables = [CategoricalParameter("c", ["a", "b", "c"])]
    objectives = [ScalarOutcome("y", kind=ScalarOutcome.MAXIMIZE)]
    problem = Problem("levers", decision_variables, objectives)

    variator = CombinedVariator()
    for value in ["a", "b", "c"]:
        solution = Sample(c=value)._to_platypus_solution(problem)
        current = solution.variables[0]
        mutated = variator.mutate_categorical(solution, 0, problem.types[0])
        new = mutated.variables[0]
        assert isinstance(new, list)
        assert len(new) == 1
        assert new != current
        assert new[0] in problem.types[0].elements


def test_optimize_categorical_and_boolean_levers(tmp_path):
    """Regression test for optimizing over categorical and boolean levers.

    Previously, this failed because platypus Subset values were not unwrapped
    and results contained category indices instead of values (issue #453).
    """
    from ema_workbench import Model

    seen = []

    def some_model(x=0.0, c="a", b=False, u=0.0):
        seen.append((c, b))
        return {"y": x + {"a": 0, "b": 5, "c": 10}[c] + (2 if b else 0) + u}

    model = Model("simple", function=some_model)
    model.uncertainties = [RealParameter("u", 0, 1)]
    model.levers = [
        RealParameter("x", 0, 1),
        CategoricalParameter("c", ["a", "b", "c"]),
        BooleanParameter("b"),
    ]
    model.outcomes = [ScalarOutcome("y", kind=ScalarOutcome.MAXIMIZE)]

    with SequentialEvaluator(model) as evaluator:
        results, _ = evaluator.optimize(
            nfe=300,
            searchover="levers",
            epsilons=[0.1],
            population_size=10,
            reference=Sample(u=0.5),
            filename="archive.tar.gz",
            directory=tmp_path,
            rng=42,
        )

    # the model only ever received actual category values
    assert {c for c, _ in seen} <= {"a", "b", "c"}
    assert {b for _, b in seen} <= {False, True}

    # results contain category values, not indices
    assert set(results["c"]) <= {"a", "b", "c"}
    assert set(results["b"]) <= {False, True}

    # the optimum is c="c" and b=True
    best = results.loc[results["y"].idxmax()]
    assert best["c"] == "c"
    assert bool(best["b"]) is True


def test_epsilon_nondominated_categorical():
    """Regression test: archives with categorical and boolean levers can be rebuilt."""
    decision_variables = [
        RealParameter("a", 0, 1),
        CategoricalParameter("c", ["x", "y", "z"]),
        BooleanParameter("b"),
    ]
    objectives = [ScalarOutcome("y", kind=ScalarOutcome.MAXIMIZE)]
    problem = Problem("levers", decision_variables, objectives)

    archive = pd.DataFrame(
        {"a": [0.1, 0.9], "c": ["x", "z"], "b": [True, False], "y": [1.0, 2.0]}
    )

    solutions = rebuild_platypus_population(archive, problem)
    assert solutions[0].variables[1] == [0]
    assert solutions[1].variables[1] == [2]
    assert solutions[0].variables[2] == [1]
    assert solutions[1].variables[2] == [0]

    merged = epsilon_nondominated([archive, archive], [0.1], problem)
    assert len(merged) == 1
    assert merged["c"].iloc[0] == "z"
    assert not merged["b"].iloc[0]
    assert merged["y"].iloc[0] == 2.0


def _shaped_lever_model(a=None, c=None, k=0):
    return {
        "y": float(np.sum(a) + sum(value == "hi" for value in c) + k),
        "z": float(-np.sum(a)),
    }


def test_optimize_with_shaped_levers(tmp_path):
    """Regression test: archives of an optimization with shaped levers.

    Previously, the archive had a column holding an array for each shaped
    lever. epsilon_nondominated and the convergence metrics could not use it,
    and the stored archives contained these arrays as strings.
    """
    from ema_workbench import Model
    from ema_workbench.em_framework.optimization import load_archives
    from ema_workbench.em_framework.optimization_convergence import (
        HypervolumeMetric,
    )
    from ema_workbench.em_framework.points import from_experiments

    model = Model("shaped", function=_shaped_lever_model)
    model.levers = [
        RealParameter("a", 0, 1, shape=(3,)),
        CategoricalParameter("c", ["lo", "hi"], shape=(2,)),
        IntegerParameter("k", 0, 3),
    ]
    model.outcomes = [
        ScalarOutcome("y", kind=ScalarOutcome.MAXIMIZE),
        ScalarOutcome("z", kind=ScalarOutcome.MAXIMIZE),
    ]
    epsilons = [0.05, 0.05]

    with SequentialEvaluator(model) as evaluator:
        results = evaluator.optimize(
            nfe=100,
            searchover="levers",
            epsilons=epsilons,
            population_size=10,
            reference=Sample("reference"),
            filename="archives.tar.gz",
            directory=str(tmp_path),
            rng=[1, 2],
        )
    archives = [archive for archive, _ in results]

    # one column for each element, as for the experiments
    expected_columns = ["a_0", "a_1", "a_2", "c_0", "c_1", "k", "y", "z"]
    assert all(list(archive.columns) == expected_columns for archive in archives)

    problem = Problem("levers", model.levers, list(model.outcomes))
    merged = epsilon_nondominated(archives, epsilons, problem)
    assert list(merged.columns) == expected_columns
    assert 0 < len(merged) <= sum(len(archive) for archive in archives)

    HypervolumeMetric(merged, problem).calculate(archives[0])

    # the stored archives can be used as well
    stored = [
        load_archives(tmp_path / f"{seed}_archives.tar.gz")[-1][1] for seed in (1, 2)
    ]
    for archive, returned in zip(stored, archives):
        assert list(archive.columns) == expected_columns
        assert archive["a_0"].dtype == float
        pd.testing.assert_frame_equal(
            archive.sort_values(["y", "z"]).reset_index(drop=True),
            returned.sort_values(["y", "z"]).reset_index(drop=True),
            check_dtype=False,
        )
    epsilon_nondominated(stored, epsilons, problem)

    # the rows can be turned back into policies
    policies = from_experiments(merged, parameters=model.levers)
    assert policies[0]["a"].shape == (3,)
    assert set(policies[0]["c"]) <= {"lo", "hi"}
    with SequentialEvaluator(model) as evaluator:
        _experiments, outcomes = evaluator.perform_experiments(
            scenarios=1, policies=policies
        )
    np.testing.assert_allclose(outcomes["y"], merged["y"])


def test_rebuild_platypus_population_with_shaped_decision_variables():
    decision_variables = [
        RealParameter("a", 0, 1, shape=(2,)),
        CategoricalParameter("c", ["lo", "hi"], shape=(2,)),
    ]
    objectives = [ScalarOutcome("y", kind=ScalarOutcome.MAXIMIZE)]
    problem = Problem("levers", decision_variables, objectives)
    assert problem.latent_parameter_names == ["a_0", "a_1", "c_0", "c_1"]

    archive = pd.DataFrame(
        {
            "a_0": [0.1, 0.5],
            "a_1": [0.2, 0.6],
            "c_0": ["lo", "hi"],
            "c_1": ["hi", "hi"],
            "y": [1.0, 2.0],
        }
    )
    solutions = rebuild_platypus_population(archive, problem)
    rebuilt = to_dataframe(solutions, problem.latent_parameter_names, ["y"])
    pd.testing.assert_frame_equal(
        rebuilt.sort_values("y").reset_index(drop=True), archive, check_dtype=False
    )

    # the arrays themselves are not accepted
    with pytest.raises(EMAError, match="does not match"):
        rebuild_platypus_population(
            pd.DataFrame(
                {
                    "a": [np.zeros(2)],
                    "c": [np.array(["lo", "lo"])],
                    "x": [0],
                    "y": [1.0],
                }
            ),
            problem,
        )
