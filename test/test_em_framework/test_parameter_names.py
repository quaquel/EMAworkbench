"""Tests for checking that parameter names are unique across a model."""

import numpy as np
import pytest

from ema_workbench import (
    Constant,
    Model,
    RealParameter,
    ReplicatorModel,
    Sample,
    ScalarOutcome,
    optimize,
    perform_experiments,
)
from ema_workbench.em_framework.evaluators import robust_optimize
from ema_workbench.util import EMAError


def model_function(a=None, a_0=0.0, b=0.0, c=0.0):
    """A simple model, only used to set up a model."""
    return {"y": float(np.sum(a)) if a is not None else 0.0}


def make_model(uncertainties=(), levers=(), constants=()):
    """Helper function for making a model."""
    model = Model("test_model", function=model_function)
    model.uncertainties = list(uncertainties)
    model.levers = list(levers)
    model.constants = list(constants)
    model.outcomes = [ScalarOutcome("y", kind=ScalarOutcome.MAXIMIZE)]
    return model


CLASHES = {
    "uncertainty and lever": (
        [RealParameter("b", 0, 1)],
        [RealParameter("b", 0, 1)],
        [],
        "'b' is used by uncertainty 'b' and lever 'b'",
    ),
    "uncertainty and constant": (
        [RealParameter("b", 0, 1)],
        [],
        [Constant("b", 0.5)],
        "'b' is used by uncertainty 'b' and constant 'b'",
    ),
    "lever and constant": (
        [],
        [RealParameter("b", 0, 1)],
        [Constant("b", 0.5)],
        "'b' is used by lever 'b' and constant 'b'",
    ),
    "element of shaped lever in same map": (
        [],
        [RealParameter("a", 0, 1, shape=(2,)), RealParameter("a_0", 0, 1)],
        [],
        "'a_0' is used by lever 'a' and lever 'a_0'",
    ),
    "element of shaped lever and uncertainty": (
        [RealParameter("a_1", 0, 1)],
        [RealParameter("a", 0, 1, shape=(2,))],
        [],
        "'a_1' is used by uncertainty 'a_1' and lever 'a'",
    ),
    "shaped uncertainty and scalar lever": (
        [RealParameter("a", 0, 1, shape=(2,))],
        [RealParameter("a", 0, 1)],
        [],
        "'a' is used by uncertainty 'a' and lever 'a'",
    ),
}


@pytest.mark.parametrize("clash", CLASHES.values(), ids=CLASHES.keys())
def test_clashing_names_are_rejected(clash):
    """Regression test: clashing names failed late with an unclear error."""
    uncertainties, levers, constants, message = clash
    model = make_model(uncertainties, levers, constants)
    with pytest.raises(EMAError, match=message):
        model._validate_parameter_names()


def test_order_of_specification_does_not_matter():
    """A name may move between uncertainties and levers in any order."""
    model = make_model(levers=[RealParameter("b", 0, 1)])
    model.uncertainties = [RealParameter("b", 0, 1)]  # temporarily used twice
    model.levers = [RealParameter("c", 0, 1)]
    model._validate_parameter_names()


def test_unique_names_are_accepted():
    model = make_model(
        uncertainties=[RealParameter("b", 0, 1)],
        levers=[RealParameter("a", 0, 1, shape=(2,))],
        constants=[Constant("c", 0.5)],
    )
    model._validate_parameter_names()


def test_perform_experiments_checks_names():
    model = make_model([RealParameter("b", 0, 1)], [RealParameter("b", 0, 1)])
    with pytest.raises(EMAError, match="must be unique"):
        perform_experiments(model, 2, 2)


def test_perform_experiments_checks_names_of_all_models():
    good = make_model([RealParameter("b", 0, 1)])
    bad = make_model([RealParameter("b", 0, 1)], constants=[Constant("b", 0.5)])
    bad.name = "bad_model"
    with pytest.raises(EMAError, match="bad_model"):
        perform_experiments([good, bad], 2)


def test_optimize_checks_names():
    model = make_model([RealParameter("b", 0, 1)], [RealParameter("b", 0, 1)])
    with pytest.raises(EMAError, match="must be unique"):
        optimize(model, nfe=10, epsilons=[0.1], reference=Sample("reference"))


def test_robust_optimize_checks_names():
    model = make_model([RealParameter("b", 0, 1)], [RealParameter("b", 0, 1)])
    robustness = [
        ScalarOutcome(
            "mean_y", kind=ScalarOutcome.MAXIMIZE, variable_name="y", function=np.mean
        )
    ]
    with pytest.raises(EMAError, match="must be unique"):
        robust_optimize(model, robustness, 2, nfe=10, epsilons=[0.1])


def make_replicator_model(replications, levers=(), constants=()):
    """Helper function for making a model with replications."""
    model = ReplicatorModel("replicator_model", function=model_function)
    model.uncertainties = [RealParameter("b", 0, 1)]
    model.levers = list(levers)
    model.constants = list(constants)
    model.outcomes = [ScalarOutcome("y")]
    model.replications = replications
    return model


def test_clash_with_replication_key_is_rejected():
    """Previously, this failed only at the first experiment, in combine."""
    model = make_replicator_model(
        [{"c": 1}, {"c": 2}], levers=[RealParameter("c", 0, 1)]
    )
    with pytest.raises(
        EMAError, match="'c' is used by lever 'c' and replication key 'c'"
    ):
        model._validate_parameter_names()


def test_clash_with_key_of_any_replication_is_rejected():
    model = make_replicator_model(
        [{"seed": 1}, {"seed": 2, "c": 0.5}], constants=[Constant("c", 0.5)]
    )
    with pytest.raises(EMAError, match="replication key 'c'"):
        model._validate_parameter_names()


@pytest.mark.parametrize(
    "replications", [3, [{"seed": 1}, {"seed": 2}]], ids=["int", "list of dicts"]
)
def test_replications_without_clash_are_accepted(replications):
    model = make_replicator_model(replications, levers=[RealParameter("c", 0, 1)])
    model._validate_parameter_names()
