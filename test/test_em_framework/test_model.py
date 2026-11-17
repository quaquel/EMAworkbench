"""Unit tests for model classes."""

# Created on Jul 28, 2015
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

import unittest
import unittest.mock as mock

from ema_workbench.em_framework.model import FileModel, Model, ReplicatorModel
from ema_workbench.em_framework.outcomes import ArrayOutcome
from ema_workbench.em_framework.parameters import (
    CategoricalParameter,
    Category,
    RealParameter,
)
from ema_workbench.em_framework.points import Sample
from ema_workbench.util import EMAError


class FileModelTest(FileModel):
    pass


class TestFileModel(unittest.TestCase):
    def test_init(self):
        model_name = "modelname"
        model_file = "model_file"

        with self.assertRaises(ValueError):
            FileModelTest(model_name, ".", model_file)

        with mock.patch("ema_workbench.em_framework.model.os") as patch:
            patch.os.is_file.set_return_value(True)
            model = FileModelTest(model_name, ".", model_file)
            self.assertEqual(model.name, model_name, "FileModel name not equal")
            self.assertEqual(model.model_file, model_file)

    def test_run_model(self):
        model_name = "modelname"
        model_file = "model_file"

        with mock.patch("ema_workbench.em_framework.model.os") as patch:
            patch.os.is_file.set_return_value(True)
            model = FileModelTest(model_name, ".", model_file)
            model.run_model(Sample(a=1), Sample("test", b=2), Sample())
            self.assertEqual(model.policy.name, "test")


def test_assignment_replaces_parameters_and_outcomes():
    """Regression test: assigning to an attribute of a model replaces its content.

    Previously, assignment added to what was already there, so assigning
    levers twice resulted in the levers of both assignments.
    """
    from ema_workbench.em_framework.outcomes import ScalarOutcome
    from ema_workbench.em_framework.parameters import Constant

    model = Model("assignment", function=lambda **kwargs: {})

    for attribute, first, second in [
        ("uncertainties", [RealParameter("a", 0, 1)], [RealParameter("b", 0, 1)]),
        ("levers", [RealParameter("c", 0, 1)], [RealParameter("d", 0, 1)]),
        ("outcomes", [ScalarOutcome("assign_e")], [ScalarOutcome("assign_f")]),
        ("constants", [Constant("g", 1)], [Constant("h", 2)]),
    ]:
        setattr(model, attribute, first)
        assert list(getattr(model, attribute).keys()) == [first[0].name]

        setattr(model, attribute, second)
        assert list(getattr(model, attribute).keys()) == [second[0].name]

        # a single object can be assigned as well
        setattr(model, attribute, first[0])
        assert list(getattr(model, attribute).keys()) == [first[0].name]

    # augmented assignment and extend still add
    model.levers += [RealParameter("d", 0, 1)]
    assert list(model.levers.keys()) == ["c", "d"]
    model.levers.extend(RealParameter("i", 0, 1))
    assert list(model.levers.keys()) == ["c", "d", "i"]

    # assigning the mapping to itself keeps the content
    model.levers = model.levers
    assert list(model.levers.keys()) == ["c", "d", "i"]


class TestModel(unittest.TestCase):
    def test_init(self):
        model_name = "modelname"

        model = Model(model_name, lambda x: x)

        self.assertEqual(model.name, model_name)
        self.assertRaises(EMAError, Model, "", "model name")

    def test_model_init(self):
        model_name = "modelname"

        def initial_func(a=1):
            return a

        model = Model(model_name, initial_func)

        def policy_func(a=1):
            return a

        policy = Sample("test", function=policy_func, unknown="a")
        model.model_init(policy)

        self.assertEqual(policy, model.policy)
        self.assertEqual(model.function, policy_func)

        with self.assertRaises(AttributeError):
            _ = model.unknown

    def test_run_model(self):
        model_name = "modelname"

        function = mock.Mock()

        model = Model(model_name, function)
        model.uncertainties = [RealParameter("a", 0, 1)]
        model.run_model(Sample(a=0.1, b=1), Sample("test"), Sample())
        function.assert_called_once_with(a=0.1)

        # test complete translation of scenario

        model = Model(model_name, function)
        model.uncertainties = [RealParameter("a", 0, 1, variable_name=["a", "b"])]

        scenario = Sample(a=0.1)
        model.run_model(scenario, Sample("test"), Sample())

        self.assertIn("a", scenario.keys())
        self.assertIn("b", scenario.keys())

        model = Model(model_name, function)
        cats = [Category("some name", [1, 2]), Category("some other name", [3, 4])]
        model.uncertainties = [
            CategoricalParameter("a", cats, variable_name=["a", "b"], multivalue=True)
        ]

        scenario = Sample(a=cats[0].value)
        model.run_model(scenario, Sample("test"), Sample())

        self.assertIn("a", scenario.keys())
        self.assertIn("b", scenario.keys())
        self.assertEqual(scenario["a"], 1)
        self.assertEqual(scenario["b"], 2)

        scenario = Sample(a=cats[1].value)
        model.run_model(scenario, Sample("test"), Sample())

        self.assertIn("a", scenario.keys())
        self.assertIn("b", scenario.keys())
        self.assertEqual(scenario["a"], 3)
        self.assertEqual(scenario["b"], 4)

        model_name = "modelname"

        function = mock.Mock()

        model = Model(model_name, function)
        model.uncertainties = [RealParameter("a", 0, 1)]
        model.run_model(Sample(a=0.1, b=1), Sample("test"), Sample())
        function.assert_called_once_with(a=0.1)

        # test complete translation of scenario

        model = Model(model_name, function)
        model.uncertainties = [RealParameter("a", 0, 1, variable_name=["a", "b"])]

        scenario = Sample(a=0.1)
        model.run_model(scenario, Sample("test"), Sample())

        self.assertIn("a", scenario.keys())
        self.assertIn("b", scenario.keys())

        model = Model(model_name, function)
        cats = [Category("some name", [1, 2]), Category("some other name", [3, 4])]
        model.uncertainties = [
            CategoricalParameter("a", cats, variable_name=["a", "b"], multivalue=True)
        ]

        scenario = Sample(a=cats[0].value)
        model.run_model(scenario, Sample("test"), Sample())

        self.assertIn("a", scenario.keys())
        self.assertIn("b", scenario.keys())
        self.assertEqual(scenario["a"], 1)
        self.assertEqual(scenario["b"], 2)

        scenario = Sample(a=cats[1].value)
        model.run_model(scenario, Sample("test"), Sample())

        self.assertIn("a", scenario.keys())
        self.assertIn("b", scenario.keys())
        self.assertEqual(scenario["a"], 3)
        self.assertEqual(scenario["b"], 4)

    def test_cleanup(self):
        model_name = "modelname"

        model = Model(model_name, lambda x: x)
        model.cleanup()

    def test_model_uncertainties(self):
        model_name = "modelname"

        model = Model(model_name, lambda x: x)
        self.assertTrue(len(model.uncertainties.keys()) == 0)

        unc_a = RealParameter("a", 0, 1)
        model.uncertainties = unc_a
        self.assertTrue(len(model.uncertainties.keys()) == 1)
        self.assertTrue(unc_a.name in model.uncertainties)

    def test_as_dict(self):
        model_name = "modelname"

        model = Model(model_name, lambda x: x)
        model.uncertainties = [RealParameter("a", 0, 1)]

        expected_keys = [
            "class",
            "name",
            "uncertainties",
            "outcomes",
            "outcomes",
            "constants",
        ]

        dict_ = model.as_dict()

        for key in expected_keys:
            self.assertIn(key, dict_)


def test_reset_model_gives_new_outcomes_output():
    """Regression test: reset_model should reset the outcomes.

    Previously, a typo meant the same dict was reused and updated for every
    experiment, so the outcomes of an earlier experiment changed when a later
    experiment was run.
    """
    from ema_workbench.em_framework.experiment_runner import ExperimentRunner
    from ema_workbench.em_framework.outcomes import ScalarOutcome
    from ema_workbench.em_framework.points import Experiment

    def some_model(u=0.0):
        return {"reset_test_y": u}

    model = Model("resetmodel", function=some_model)
    model.uncertainties = [RealParameter("u", 0, 1)]
    model.outcomes = [ScalarOutcome("reset_test_y")]

    runner = ExperimentRunner([model])
    policy = Sample("no_policy")
    first = runner.run_experiment(
        Experiment("first", model.name, policy, Sample("s1", u=0.1), 0)
    )
    second = runner.run_experiment(
        Experiment("second", model.name, policy, Sample("s2", u=0.9), 1)
    )

    assert first is not second
    assert first == {"reset_test_y": 0.1}
    assert second == {"reset_test_y": 0.9}
    assert model.outcomes_output == {}


class TestReplicatorModel(unittest.TestCase):
    def test_run_model(self):
        model_name = "modelname"

        function = mock.Mock()
        function.return_value = {"outcome": 2}

        model = ReplicatorModel(model_name, function)
        model.uncertainties = [RealParameter("a", 0, 1)]
        model.outcomes = [ArrayOutcome("outcome")]
        model.replications = 2

        model.run_model(Sample(a=0.1, b=1), Sample("test"), Sample())
        self.assertEqual(function.call_count, 2)
        self.assertEqual({"outcome": [2, 2]}, model.outcomes_output)


#         self.assertEqual(scenario['b'], 4)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
