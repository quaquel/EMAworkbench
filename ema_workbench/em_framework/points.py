"""classes for representing points in parameter space, as well as associated
hellper functions

"""

from collections import ChainMap
import itertools
import random

from ema_workbench.em_framework.util import NamedDict, Counter, NamedObject
from ema_workbench.em_framework.util import combine
from ema_workbench.util import get_module_logger

__all__ = [
    "Point",
    "Policy",
    "Experiment",
    "ExperimentReplication",
    "combine_cases_sampling",
    "combine_cases_factorial",
    "experiment_generator",
    "Scenario",
]
_logger = get_module_logger(__name__)


class Point(NamedDict):
    id_counter = Counter(1)
    name_counter = Counter(0)

    def __init__(self, name=None, unique_id=None, **kwargs):
        if name is None:
            name = Point.name_counter()
        if unique_id is None:
            unique_id = Point.id_counter()

        super().__init__(name, **kwargs)
        self.unique_id = unique_id


class Policy(Point):
    """Helper class representing a policy

    Attributes
    ----------
    name : str, int, or float
    id : int

    all keyword arguments are wrapped into a dict.

    """

    id_counter = Counter(1)

    def __init__(self, name=None, **kwargs):
        super().__init__(name, Policy.id_counter(), **kwargs)

    # def to_list(self, parameters):
    #     """get list like representation of policy where the
    #     parameters are in the order of levers"""
    #
    #     return [self[param.name] for param in parameters]

    def __repr__(self):
        return f"Policy({super(Policy, self).__repr__()})"


class Scenario(Point):
    """Helper class representing a scenario

    Attributes
    ----------
    name : str, int, or float
    id : int

    all keyword arguments are wrapped into a dict.

    """

    # we need to start from 1 so scenario id is known
    id_counter = Counter(1)

    def __init__(self, name=None, **kwargs):
        super().__init__(name, Scenario.id_counter(), **kwargs)

    def __repr__(self):
        return f"Scenario({super(Scenario, self).__repr__()})"


class Experiment(NamedObject):
    """A convenience object that contains a specification
    of the model, policy, and scenario to run

    Attributes
    ----------
    name : str
    model_name : str
    policy : Policy instance
    scenario : Scenario instance
    experiment_id : int

    """

    def __init__(self, name, model_name, policy, scenario, experiment_id):
        super().__init__(name)
        self.experiment_id = experiment_id
        self.policy = policy
        self.model_name = model_name
        self.scenario = scenario


class ExperimentReplication(NamedDict):
    """helper class that combines scenario, policy, any constants, and
    replication information (seed etc) into a single dictionary.

    This class represent the complete specification of parameters to run for
    a given experiment.

    """

    def __init__(self, scenario, policy, constants, replication=None):
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
    # zipover
    #     taken from jpn
    #     getting the max might by tricky
    #     policies and scenarios are generators themselves?
    # TODO to be replaced with sample based combining

    max_len = max(len(a) for a in args)
    return itertools.islice(zip(*(itertools.cycle(a) for a in args)), max_len)


def combine_cases_sampling(*point_collection):
    """Combine collections of cases by iterating over the longest collection
    while sampling with replacement from the others

    Parameters
    ----------
    point_collection : collection of collection of Point instances

    Yields
    -------
    Point

    """

    # figure out the longest
    def exhaust_cases(cases):
        return [case for case in cases]

    point_collection = [exhaust_cases(case) for case in point_collection]
    longest_cases = max(point_collection, key=len)
    other_cases = [case for case in point_collection if case is not longest_cases]

    for case in longest_cases:
        other = (random.choice(entry) for entry in other_cases)

        yield Point(**ChainMap(case, *other))


def combine_cases_factorial(*point_collections):
    """Combine collections of cases in a full factorial manner

    Parameters
    ----------
    point_collections : collection of collections of Point instances

    Yields
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


def experiment_generator(scenarios, model_structures, policies, combine="factorial"):
    """

    generator function which yields experiments

    Parameters
    ----------
    scenarios : iterable of dicts
    model_structures : list
    policies : list
    combine = {'factorial, sample'}
              controls how to combine scenarios, policies, and model_structures
              into experiments.

    Notes
    -----
    if combine is 'factorial' then this generator is essentially three nested
    loops: for each model structure, for each policy, for each scenario,
    return the experiment. This means that designs should not be a generator
    because this will be exhausted after the running the first policy on the
    first model.
    if combine is 'zipover' then this generator cycles over scenarios, policies
    and model structures until the longest of the three collections is
    exhausted.

    """

    # TODO combine_ functions can be made more generic
    # basically combine any collection
    # wrap around to yield specifc type of class (e.g. point

    if combine == "sample":
        jobs = zip_cycle(model_structures, policies, scenarios)
    elif combine == "factorial":
        # full factorial
        jobs = itertools.product(model_structures, policies, scenarios)
    else:
        raise ValueError(f"{combine} is unknown value for combine")

    for i, job in enumerate(jobs):
        msi, policy, scenario = job
        name = f"{msi.name} {policy.name} {i}"
        experiment = Experiment(name, msi.name, policy, scenario, i)
        yield experiment
