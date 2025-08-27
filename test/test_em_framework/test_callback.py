"""Created on 22 Jan 2013

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
"""

import random

import numpy as np
import pandas as pd
import pytest

from ema_workbench.em_framework.callbacks import DefaultCallback, FileBasedCallback
from ema_workbench.em_framework.outcomes import (
    ArrayOutcome,
    ScalarOutcome,
    TimeSeriesOutcome,
)
from ema_workbench.em_framework.parameters import (
    BooleanParameter,
    CategoricalParameter,
    IntegerParameter,
    RealParameter,
)
from ema_workbench.em_framework.points import Experiment, Policy, Scenario
from ema_workbench.em_framework.util import NamedObject
from ema_workbench.util import EMAError


def test_store_results(mocker):
    nr_experiments = 3
    uncs = [RealParameter("a", 0, 1), RealParameter("b", 0, 1)]
    outcomes = [TimeSeriesOutcome("test")]
    model = NamedObject("test")

    experiment = Experiment(0, model, Policy("policy"), Scenario(a=1, b=0), 0)

    # case 1 scalar shape = (1)
    callback = DefaultCallback(uncs, [], outcomes, nr_experiments=nr_experiments)
    model_outcomes = {outcomes[0].name: 1.0}
    callback(experiment, model_outcomes)

    _, out = callback.get_results()

    assert outcomes[0].name in set(out.keys())
    assert out[outcomes[0].name].shape == (3,)

    # case 2 time series shape = (1, nr_time_steps)
    callback = DefaultCallback(uncs, [], outcomes, nr_experiments=nr_experiments)
    model_outcomes = {outcomes[0].name: np.random.rand(10)}
    callback(experiment, model_outcomes)

    _, out = callback.get_results()
    assert outcomes[0].name in out.keys()
    assert out[outcomes[0].name].shape == (3, 10)

    # case 3 maps etc. shape = (x,y)
    callback = DefaultCallback(uncs, [], outcomes, nr_experiments=nr_experiments)
    model_outcomes = {outcomes[0].name: np.random.rand(2, 2)}
    callback(experiment, model_outcomes)

    _, out = callback.get_results()
    assert outcomes[0].name in out.keys()
    assert out[outcomes[0].name].shape == (3, 2, 2)

    # case 4 assert raises EMAError
    callback = DefaultCallback(uncs, [], outcomes, nr_experiments=nr_experiments)
    model_outcomes = {outcomes[0].name: np.random.rand(2, 2, 2)}
    with pytest.raises(EMAError):
        callback(experiment, model_outcomes)

    # case 5 assert raises KeyError
    callback = DefaultCallback(uncs, [], outcomes, nr_experiments=nr_experiments)
    model_outcomes = {"some_other_name": np.random.rand(2, 2, 2)}
    mock = mocker.patch(
        "ema_workbench.em_framework.callbacks._logger.debug",
        autospec=True,
        side_effect=lambda *args, **kwargs: print(args, kwargs),
    )
    callback._store_outcomes(1, model_outcomes)
    assert mock.call_count == 1


def test_init():
    # let's add some uncertainties to this
    uncs = [RealParameter("a", 0, 1), RealParameter("b", 0, 1)]
    outcomes = [
        ScalarOutcome("scalar"),
        ArrayOutcome("array", shape=(10,), dtype=float),
        TimeSeriesOutcome("timeseries"),
    ]
    callback = DefaultCallback(uncs, [], outcomes, nr_experiments=100)

    assert callback.i == 0
    assert callback.nr_experiments == 100
    assert callback.cases.shape[0] == 100
    assert callback.reporting_interval == 100
    #         self.assertEqual(callback.outcomes, outcomes)

    names = [name for name, _ in callback.uncertainty_and_lever_labels]
    names = set(names)
    assert names == {"a", "b"}

    assert "scalar" not in callback.results
    assert "timeseries" not in callback.results
    assert "array" in callback.results
    assert np.ma.is_masked(callback.results["array"])

    # with levers
    levers = [RealParameter("c", 0, 10)]

    callback = DefaultCallback(
        uncs,
        levers,
        outcomes,
        nr_experiments=1000,
        reporting_interval=None,
        reporting_frequency=4,
    )

    assert callback.i == 0
    assert callback.nr_experiments == 1000
    assert callback.cases.shape[0] == 1000
    assert callback.reporting_interval == 250
    #         self.assertEqual(callback.outcomes, [o.name for o in outcomes])

    names = [name for name, _ in callback.uncertainty_and_lever_labels]
    names = set(names)
    assert names == {"a", "b", "c"}

    assert "scalar" not in callback.results
    assert "timeseries" not in callback.results
    assert "array" in callback.results
    assert np.ma.is_masked(callback.results["array"])


#         # KeyError
#         with mock.patch('ema_workbench.util.ema_logging.debug') as mocked_logging:
#             callback = DefaultCallback(uncs, [], outcomes,
#                                        nr_experiments=nr_experiments)
#             model_outcomes = {'incorrect': np.random.rand(2,)}
#             callback(experiment, model_outcomes)
#
#             for outcome in outcomes:
#                 mocked_logging.assert_called_with("%s not specified as outcome in msi" % outcome.name)


def test_store_cases():
    nr_experiments = 3
    uncs = [
        RealParameter("a", 0, 1),
        RealParameter("b", 0, 1),
        CategoricalParameter("c", [0, 1, 2]),
        IntegerParameter("d", 0, 1),
        BooleanParameter("e"),
    ]
    outcomes = [TimeSeriesOutcome("test")]
    case = {unc.name: random.random() for unc in uncs}
    case["c"] = int(round(case["c"] * 2))
    case["d"] = int(round(case["d"]))
    case["e"] = True

    model = NamedObject("test")
    policy = Policy("policy")
    scenario = Scenario(**case)
    experiment = Experiment(0, model.name, policy, scenario, 0)

    callback = DefaultCallback(
        uncs, [], outcomes, nr_experiments=nr_experiments, reporting_interval=1
    )
    model_outcomes = {outcomes[0].name: 1.0}
    callback(experiment, model_outcomes)

    experiments, _ = callback.get_results()
    # design = case
    case["policy"] = policy.name
    case["model"] = model.name
    case["scenario"] = scenario.name

    names = experiments.columns.values.tolist()
    for name in names:
        entry_a = experiments[name][0]
        entry_b = case[name]

        assert entry_a == entry_b, "failed for " + name

    # with levers
    nr_experiments = 3
    uncs = [RealParameter("a", 0, 1), RealParameter("b", 0, 1)]
    levers = [RealParameter("c", 0, 1), RealParameter("d", 0, 1)]
    outcomes = [TimeSeriesOutcome("test")]
    case = {unc.name: random.random() for unc in uncs}

    model = NamedObject("test")
    policy = Policy("policy", c=1, d=1)
    scenario = Scenario(**case)
    experiment = Experiment(0, model.name, policy, scenario, 0)

    callback = DefaultCallback(
        uncs, levers, outcomes, nr_experiments=nr_experiments, reporting_interval=1
    )
    model_outcomes = {outcomes[0].name: 1.0}
    callback(experiment, model_outcomes)

    experiments, _ = callback.get_results()
    design = case
    design["c"] = 1
    design["d"] = 1
    design["policy"] = policy.name
    design["model"] = model.name
    design["scenario"] = scenario.name

    names = experiments.columns.values.tolist()
    for name in names:
        assert experiments[name][0] == design.get(name), f"failed for name {name}"


def test_get_results(mocker):
    nr_experiments = 3
    uncs = [
        RealParameter("a", 0, 1),
        CategoricalParameter("b", ["0", "1", "2", "3"]),
        IntegerParameter("c", 0, 5),
        BooleanParameter("d"),
    ]
    outcomes = [ScalarOutcome("other_test")]
    outcomes[0].shape = (1,)

    callback = DefaultCallback(
        uncs, [], outcomes, nr_experiments=nr_experiments, reporting_interval=1
    )
    # test warning
    mock = mocker.patch("ema_workbench.em_framework.callbacks._logger.warning")
    callback.get_results()
    assert mock.call_count == 1

    # test without warning
    callback = DefaultCallback(
        uncs, [], outcomes, nr_experiments=nr_experiments, reporting_interval=1
    )

    cases = []
    for i in range(nr_experiments):
        model = NamedObject("test")
        policy = Policy("policy")
        case = {"a": i * 0.15, "b": f"{i}", "c": i, "d": True if i % 2 == 0 else False}
        scenario = Scenario(**case)
        experiment = Experiment(0, model.name, policy, scenario, i)
        model_outcomes = {outcomes[0].name: i * 1.25}
        callback(experiment, model_outcomes)
        cases.append(case)

    mock = mocker.patch("ema_workbench.em_framework.callbacks._logger.warning")
    callback.results = {k: v.data for k, v in callback.results.items()}
    experiments, results = callback.get_results()
    assert mock.call_count == 0

    # check if experiments dataframe contains the experiments correctly
    data = pd.DataFrame.from_dict(cases)
    assert np.all(data == experiments.loc[:, data.columns])

    # check data types of columns in experiments dataframe
    dtype_mapping = {
        RealParameter: float,
        CategoricalParameter: "category",
        IntegerParameter: int,
        BooleanParameter: bool,
    }

    for u in uncs:
        assert experiments.loc[:, u.name].dtype == dtype_mapping[u.__class__]


def test_filebasedcallback(mocker):
    # only most basic assertions are checked

    mock_os = mocker.patch("ema_workbench.em_framework.callbacks.os")
    mock_shutil = mocker.patch("ema_workbench.em_framework.callbacks.shutil")
    mock_open = mocker.patch("builtins.open")

    nr_experiments = 3
    uncs = [
        RealParameter("a", 0, 1),
        RealParameter("c", 0, 1),
    ]
    levers = [
        RealParameter("b", 0, 1),
    ]

    outcomes = [ScalarOutcome("other"), TimeSeriesOutcome("time")]

    model = NamedObject("test")
    scenario = Scenario(a=random.random())
    policy = Policy(name="policy", b=random.random())
    experiment = Experiment(0, model.name, policy, scenario, 0)

    callback = FileBasedCallback(
        uncs, levers, outcomes, nr_experiments=nr_experiments, reporting_interval=1
    )

    # we should have opened 3 files: experiments and 2 outcome files
    assert mock_open.call_count == 3
    assert "other" in callback.outcome_fhs.keys()
    assert "time" in callback.outcome_fhs.keys()

    callback(experiment, {"other": 1, "time": [1, 2, 3, 4]})

    callback.get_results()
