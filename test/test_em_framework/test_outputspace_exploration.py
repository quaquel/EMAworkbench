from ema_workbench.em_framework import outputspace_exploration

import pytest


def test_novelty(mocker):
    algorithm = mocker.patch('ema_workbench.em_framework.outputspace_exploration.HitBox.get_novelty_score',
                             return_value=1)

    algorithm = mocker.Mock()
    algorithm.archive.get_novelty_score.side_effect = [1, 2,
                                                       2, 2,
                                                       2, 1]

    novelty = outputspace_exploration.Novelty(algorithm)
    assert novelty.compare('a', 'b') == 1
    assert novelty.compare('a', 'b') == 0
    assert novelty.compare('a', 'b') == -1


def test_hitbox(mocker):

    grid_spec = [(0, 1, 0.1),
                 (0, 1, 0.1)]

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
    assert hitbox.overall_novelty == 1.5\


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
    assert hitbox.get_novelty_score(solution2) == 1/2
    assert hitbox.get_novelty_score(solution2) == hitbox.get_novelty_score(solution)


def test_algorithm():
    pass


def test_adaptive_algorithm():
    pass