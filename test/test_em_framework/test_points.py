import numpy as np

import pytest

from ema_workbench.em_framework.parameters import RealParameter
from ema_workbench.em_framework.points import SampleCollection, Sample
from ema_workbench.em_framework import points
from ema_workbench.em_framework.util import NamedObject


def test_experiment_gemerator():
    scenarios = [NamedObject("scen_1"), NamedObject("scen_2")]
    model = [NamedObject("model")]
    policies = [NamedObject("1"), NamedObject("2"), NamedObject("3")]

    experiments = points.experiment_generator(
        scenarios, model, policies, combine="full_factorial"
    )
    experiments = list(experiments)

    assert len(experiments) == 6, "wrong number of experiments for factorial"

    experiments = points.experiment_generator(
        scenarios, model, policies, combine="cycle"
    )
    experiments = list(experiments)
    assert len(experiments) == 3, "wrong number of experiments for zipover"

    with pytest.raises(ValueError):
        experiments = points.experiment_generator(
            scenarios, model, policies, combine="adf"
        )
        _ = list(experiments)


def test_sample_collection():
    rng = np.random.default_rng(42)

    samples = rng.uniform(size=(10, 3))
    parameters = [RealParameter(entry, 0, 1) for entry in "abc"]
    iterator = SampleCollection(samples, parameters)

    # basic intit
    assert iterator.n == samples.shape[0]

    # iteration
    it = iter(iterator)
    for i, entry in enumerate(it):
        values = np.array([entry[k] for k in "abc"])
        assert np.all(values == samples[i])

    # combine
    ## full factorial
    samples1 = np.array([0, 1]).reshape(2, 1)
    samples2 = np.array([0, 1]).reshape(2, 1)

    it1 = SampleCollection(samples1, [RealParameter("a", 0, 1)])
    it2 = SampleCollection(samples2, [RealParameter("a", 0, 1)])

    it3 = it1.combine(it2, "full_factorial")
    samples = it3.samples

    assert np.all(samples == np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))

    ## equal length, so no need to cycle
    samples1 = np.array([0, 1]).reshape(2, 1)
    samples2 = np.array([0, 1]).reshape(2, 1)

    it1 = SampleCollection(samples1, [RealParameter("a", 0, 1)])
    it2 = SampleCollection(samples2, [RealParameter("a", 0, 1)])

    it3 = it1.combine(it2, "cycle")
    samples = it3.samples
    assert np.all(samples == np.array([[0, 0], [1, 1]]))

    ## cycle
    samples1 = np.array([0, 1, 2]).reshape(3, 1)
    samples2 = np.array([0, 1]).reshape(2, 1)

    it1 = SampleCollection(samples1, [RealParameter("a", 0, 1)])
    it2 = SampleCollection(samples2, [RealParameter("a", 0, 1)])

    it3 = it1.combine(it2, "cycle")
    samples = it3.samples
    assert np.all(samples == np.array([[0, 0], [1, 1], [2, 0]]))

    ## sample
    it3 = it1.combine(it2, "sample", rng=42)
    samples = it3.samples
    assert np.all(
        samples == np.array([[0, 0], [1, 1], [2, 1]])
    )  # return has been hard coded for rng 42

    with pytest.raises(ValueError):
        it1.combine(it2, "wrong value")


if __name__ == "__main__":
    unittest.main()
