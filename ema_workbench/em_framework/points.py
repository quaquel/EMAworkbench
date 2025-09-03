"""Classes for representing points in parameter space.

As well as associated helper functions

"""

import itertools
import random
from collections import ChainMap

from ema_workbench.em_framework.util import Counter, NamedDict, NamedObject, combine
from ema_workbench.util import get_module_logger

__all__ = [
    "Experiment",
    "ExperimentReplication",
    "Point",
    "Policy",
    "Scenario",
    "combine_cases_factorial",
    "combine_cases_sampling",
    "experiment_generator",
]
_logger = get_module_logger(__name__)


class Point(NamedDict):
    """A point in parameter space."""

    id_counter = Counter(1)
    name_counter = Counter(0)

    def __init__(self, name=None, unique_id=None, **kwargs): #noqa: D107
        if name is None:
            name = Point.name_counter()
        if unique_id is None:
            unique_id = Point.id_counter()

        super().__init__(name, **kwargs)
        self.unique_id = unique_id

    def __repr__(self): # noqa D105
        return f"Point({super().__repr__()})"

class Policy(Point):
    """Helper class representing a policy.

    Attributes:
    ----------
    name : str, int, or float
    id : int

    all keyword arguments are wrapped into a dict.

    """

    id_counter = Counter(1)

    def __init__(self, name=None, **kwargs): # noqa: D107
        super().__init__(name, unique_id=Policy.id_counter(), **kwargs)

    # def to_list(self, parameters):
    #     """get list like representation of policy where the
    #     parameters are in the order of levers"""
    #
    #     return [self[param.name] for param in parameters]

    def __repr__(self): # noqa D105
        return f"Policy({super().__repr__()})"


class Scenario(Point):
    """Helper class representing a scenario.

    Attributes:
    ----------
    name : str, int, or float
    id : int

    all keyword arguments are wrapped into a dict.

    """

    # we need to start from 1 so scenario id is known
    id_counter = Counter(1)

    def __init__(self, name=None, **kwargs):  # noqa: D107
        super().__init__(name, unique_id=Scenario.id_counter(), **kwargs)

    def __repr__(self): # noqa: D105
        return f"Scenario({super().__repr__()})"


class Experiment(NamedObject):
    """A convenience object that contains a specification of the model, policy, and scenario to run.

    Attributes:
    ----------
    name : str
    model_name : str
    policy : Policy instance
    scenario : Scenario instance
    experiment_id : int

    """

    def __init__(self, name, model_name, policy, scenario, experiment_id):  # noqa: D107
        super().__init__(name)
        self.experiment_id = experiment_id
        self.policy = policy
        self.model_name = model_name
        self.scenario = scenario

    def __repr__(self): # noqa: D105
        return (
            f"Experiment(name={self.name!r}, model_name={self.model_name!r}, "
            f"policy={self.policy!r}, scenario={self.scenario!r}, "
            f"experiment_id={self.experiment_id!r})"
        )

    def __str__(self):  # noqa: D105
        return f"Experiment {self.experiment_id} (model: {self.model_name}, policy: {self.policy.name}, scenario: {self.scenario.name})"


class ExperimentReplication(NamedDict):
    """Helper class that combines scenario, policy, any constants, and replication information.

    This class represent the complete specification of parameters to run for
    a given experiment.

    """

    def __init__(self, scenario, policy, constants, replication=None):  # noqa: D107
        scenario_id = scenario.unique_id
        policy_id = policy.unique_id

        if replication is None:
            replication_id = 1
        else:
            replication_id = replication.id
            constants = combine(constants, replication)

        # this is a unique identifier for an experiment
        # we might also create a better looking name
        self.id = scenario_id * policy_id * replication_id
        name = f"{scenario.name}_{policy.name}_{replication_id}"

        super().__init__(name, **combine(scenario, policy, constants))


def zip_cycle(*args):
    """Helper function for cycling over zips."""
    # zipover
    #     taken from jpn
    #     getting the max might by tricky
    #     policies and scenarios are generators themselves?
    # TODO to be replaced with sample based combining

    max_len = max(len(a) for a in args)
    return itertools.islice(zip(*(itertools.cycle(a) for a in args)), max_len)


def combine_cases_sampling(*point_collection):
    """Helper function for combining cases sampling.

    Combine collections of cases by iterating over the longest collection
    while sampling with replacement from the others.

    Parameters
    ----------
    point_collection : collection of collection of Point instances

    Yields:
    -------
    Point

    """

    # figure out the longest
    def exhaust_cases(cases):
        return list(cases)

    point_collection = [exhaust_cases(case) for case in point_collection]
    longest_cases = max(point_collection, key=len)
    other_cases = [case for case in point_collection if case is not longest_cases]

    for case in longest_cases:
        other = (random.choice(entry) for entry in other_cases)

        yield Point(**ChainMap(case, *other))


def combine_cases_factorial(*point_collections):
    """Combine collections of cases in a full factorial manner.

    Parameters
    ----------
    point_collections : collection of collections of Point instances

    Yields:
    -------
    Point

    """
    combined_cases = itertools.product(*point_collections)

    for entry in combined_cases:
        yield Point(**ChainMap(*entry))


# def combine_cases(method, *cases):
#     """
#
#     generator function which yields experiments
#
#     Parameters
#     ----------
#     combine = {'factorial, sample'}
#               controls how to combine scenarios, policies, and model_structures
#               into experiments.
#     cases
#
#     """
#
#     if method == 'sample':
#         combined_cases = zip_cycle(cases)
#     elif method == 'factorial':
#         # full factorial
#         combined_cases = itertools.product(*cases)
#     else:
#         ValueError(f"{combine} is unknown value for combine")
#
#     return combined_cases


def experiment_generator(scenarios, models, policies, combine="factorial"):
    """Generator function which yields experiments.

    Parameters
    ----------
    scenarios : iterable of dicts
    models : list
    policies : list
    combine = {'factorial, sample'}
              controls how to combine scenarios, policies, and models
              into experiments.

    Notes:
    -----
    if combine is 'factorial' then this generator is essentially three nested
    loops: for each model, for each policy, for each scenario,
    return the experiment. This means that designs should not be a generator
    because this will be exhausted after the running the first policy on the
    first model.
    if combine is 'zipover' then this generator cycles over scenarios, policies
    and models until the longest of the three collections is
    exhausted.

    """
    # TODO combine_ functions can be made more generic
    # basically combine any collection
    # wrap around to yield specific type of class (e.g. point

    if combine == "sample":
        jobs = zip_cycle(models, policies, scenarios)
    elif combine == "factorial":
        # full factorial
        jobs = itertools.product(models, policies, scenarios)
    else:
        raise ValueError(
            f"{combine} is unknown value for combine, use 'factorial' or 'sample'"
        )

    for i, job in enumerate(jobs):
        model, policy, scenario = job
        name = f"{model.name} {policy.name} {i}"
        experiment = Experiment(name, model.name, policy, scenario, i)
        yield experiment
