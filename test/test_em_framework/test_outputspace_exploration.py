import pytest

from ema_workbench.em_framework import outputspace_exploration
from ema_workbench.em_framework.optimization import to_problem
from ema_workbench.em_framework import Model, RealParameter, ScalarOutcome
from ema_workbench.util import EMAError


def test_novelty(mocker):
    algorithm = mocker.patch(
        "ema_workbench.em_framework.outputspace_exploration.HitBox.get_novelty_score",
        return_value=1,
    )

    algorithm = mocker.Mock()
    algorithm.archive.get_novelty_score.side_effect = [1, 2, 2, 2, 2, 1]

    novelty = outputspace_exploration.Novelty(algorithm)
    assert novelty.compare("a", "b") == 1
    assert novelty.compare("a", "b") == 0
    assert novelty.compare("a", "b") == -1


def test_hitbox(mocker):
    grid_spec = [
        (0, 1, 0.1),
        (0, 1, 0.1),
    ]

    # test first more central and then farther away
    hitbox = outputspace_exploration.HitBox(grid_spec)

    solution = mocker.Mock()
    solution.objectives = [0.05, 0.05]
    hitbox.add(solution)

    key = (0, 0)
    assert hitbox.grid_counter[key] == 1
    assert hitbox.improvements == 1
    assert hitbox.archive[key] is solution
    assert hitbox.centroids[key] == [0.05, 0.05]
    assert hitbox.overall_novelty == 1

    solution2 = mocker.Mock()
    solution2.objectives = [0.06, 0.06]
    hitbox.add(solution2)

    key = (0, 0)
    assert hitbox.grid_counter[key] == 2
    assert hitbox.improvements == 1
    assert hitbox.archive[key] is solution
    assert hitbox.centroids[key] == [0.05, 0.05]
    assert hitbox.overall_novelty == 1.5

    # test first farther away and then more central
    hitbox = outputspace_exploration.HitBox(grid_spec)

    solution = mocker.Mock()
    solution.objectives = [0.06, 0.06]
    hitbox.add(solution)

    key = (0, 0)
    assert hitbox.grid_counter[key] == 1
    assert hitbox.improvements == 1
    assert hitbox.archive[key] is solution
    assert hitbox.centroids[key] == [0.05, 0.05]
    assert hitbox.overall_novelty == 1

    solution2 = mocker.Mock()
    solution2.objectives = [0.05, 0.05]
    hitbox.add(solution2)

    key = (0, 0)
    assert hitbox.grid_counter[key] == 2
    assert hitbox.improvements == 1
    assert hitbox.archive[key] is solution2
    assert hitbox.centroids[key] == [0.05, 0.05]
    assert hitbox.overall_novelty == 1.5

    # test get_novelty_score
    assert hitbox.get_novelty_score(solution2) == 1 / 2
    assert hitbox.get_novelty_score(solution2) == hitbox.get_novelty_score(solution)


def test_core_algorithm(mocker):
    function = mocker.Mock()
    model = Model("A", function)
    model.uncertainties = [
        RealParameter("a", 0, 1),
        RealParameter("b", 0, 1),
        RealParameter("c", 0, 1),
    ]
    model.outcomes = [
        ScalarOutcome("x", kind=ScalarOutcome.MAXIMIZE),
        ScalarOutcome("y", kind=ScalarOutcome.MAXIMIZE),
    ]

    def evaluate_all(jobs):
        [job.run() for job in jobs]
        return jobs

    def some_callable(vars):
        return min(vars), max(vars)

    problem = to_problem(model, searchover="uncertainties")
    problem.function = some_callable
    grid_spec = [
        (0, 1, 0.1),
        (0, 1, 0.1),
    ]
    evaluator = mocker.Mock()
    evaluator.evaluate_all.side_effect = evaluate_all
    population_size = 100

    # test only does minimum checking if code runs
    # but we are not testing inner workings of platypus
    algorithm = outputspace_exploration.OutputSpaceExplorationAlgorithm(
        problem,
        grid_spec=grid_spec,
        evaluator=evaluator,
        population_size=population_size,
    )
    algorithm.step()
    algorithm.step()

    assert algorithm.nfe <= 2 * population_size
    assert len(algorithm.population) == population_size

    with pytest.raises(EMAError):
        grid_spec = [
            (0, 1, 0.1),
            (0, 1, 0.1),
            (0, 1, 0.1),
        ]
        algorithm = outputspace_exploration.OutputSpaceExplorationAlgorithm(
            problem,
            grid_spec=grid_spec,
            evaluator=evaluator,
            population_size=population_size,
        )


def test_user_facing_algorithms(mocker):
    function = mocker.Mock()
    model = Model("A", function)
    model.uncertainties = [
        RealParameter("a", 0, 1),
        RealParameter("b", 0, 1),
        RealParameter("c", 0, 1),
    ]
    model.outcomes = [
        ScalarOutcome("x", kind=ScalarOutcome.MAXIMIZE),
        ScalarOutcome("y", kind=ScalarOutcome.MAXIMIZE),
    ]

    problem = to_problem(model, searchover="uncertainties")
    grid_spec = [
        (0, 1, 0.1),
        (0, 1, 0.1),
    ]

    outputspace_exploration.OutputSpaceExploration(problem, grid_spec=grid_spec)

    outputspace_exploration.AutoAdaptiveOutputSpaceExploration(
        problem, grid_spec=grid_spec
    )
