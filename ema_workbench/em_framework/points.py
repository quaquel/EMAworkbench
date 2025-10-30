"""Classes for representing points in parameter space.

As well as associated helper functions

"""

import itertools
import math
from collections.abc import Generator, Iterable, Sequence
from typing import Literal, overload

import numpy as np
import pandas as pd
import platypus

from ..util import get_module_logger
from .parameters import (
    CategoricalParameter,
    IntegerParameter,
    Parameter,
)
from .util import Counter, NamedDict, NamedObject, combine

__all__ = [
    "Experiment",
    "ExperimentReplication",
    "Sample",
    "SampleCollection",
    "experiment_generator",
    "from_experiments",
]
_logger = get_module_logger(__name__)

SeedLike = int | np.integer | Sequence[int] | np.random.SeedSequence
RNGLike = np.random.Generator | np.random.BitGenerator


class Sample(NamedDict):
    """A point in parameter space."""

    id_counter = Counter(1)
    name_counter = Counter(0)

    def __init__(self, name=None, unique_id=None, **kwargs):
        if name is None:
            name = Sample.name_counter()
        if unique_id is None:
            unique_id = Sample.id_counter()

        super().__init__(name, **kwargs)
        self.unique_id = unique_id

    def __repr__(self):  # noqa D105
        return f"Sample({super().__repr__()})"

    def _to_platypus_solution(
        self, problem: "Problem"  # noqa: F821
    ) -> platypus.Solution:
        """Turn a Sample into a Platypus solution."""
        solution = platypus.Solution(problem)

        values = []
        for dtype, parameter in zip(problem.types, problem.decision_variables):
            value = self[parameter.name]
            converted_value = dtype.encode(value)
            values.append(converted_value)
        solution.variables[:] = values

        return solution

    @classmethod
    def _from_platypus_solution(cls, solution: platypus.Solution) -> "Sample":
        """Create a Sample from a Platypus solution."""
        problem = solution.problem
        converted_vars = {}
        for dtype, parameter, value in zip(
            problem.types, problem.decision_variables, solution.variables
        ):  # @ReservedAssignment
            converted_value = dtype.decode(value)
            converted_vars[parameter.name] = converted_value
        return Sample(**converted_vars)


class Experiment(NamedObject):
    """A convenience object that contains a specification of the model, policy, and scenario to run.

    Attributes
    ----------
    name : str
    model_name : str
    policy : Sample instance
    scenario : Sample instance
    experiment_id : int

    """

    def __init__(
        self,
        name: str,
        model_name: str,
        policy: Sample,
        scenario: Sample,
        experiment_id: int,
    ):
        super().__init__(name)
        self.experiment_id = experiment_id
        self.policy = policy
        self.model_name = model_name
        self.scenario = scenario

    def __repr__(self):  # noqa: D105
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


class SampleCollection(Iterable):
    """Collection of sample instances.

    A Sample is a point in a parameter space.

    """

    # the construction with a class and the generator ensures we can repeatedly iterate over the samples.

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the samples."""
        return self.samples.shape

    def __init__(
        self,
        samples: np.ndarray,
        parameters: list[Parameter],
    ):
        if samples.shape[1] != len(parameters):
            raise ValueError(
                "the number of columns in samples does not match the number of parameters"
            )
        self.samples = samples
        self.parameters = parameters
        self.n = self.samples.shape[0]

    def __iter__(self) -> Iterable[Sample]:
        """Return an iterator yielding Points instances."""
        return sample_generator(self.samples, self.parameters)

    def __str__(self):  # noqa: D105
        return f"ema_workbench.SampleCollection, {self.n} samples on {len(self.parameters)} parameters"

    def __len__(self):
        """Return the number of samples in the collection."""
        return self.samples.shape[0]

    @overload
    def __getitem__(self, key: int) -> Sample: ...

    @overload
    def __getitem__(self, key: slice) -> "SampleCollection": ...

    def __getitem__(self, key):
        """Return the samples for the index or slice."""
        if not isinstance(key, int | slice | np.integer):
            raise TypeError(
                f"SampleCollection indices must be integers or slices, not {type(key)}"
            )

        samples = self.samples[key]

        if np.issubdtype(type(key), np.integer):
            return next(
                sample_generator(
                    samples.reshape((1, samples.shape[0])), self.parameters
                )
            )

        return SampleCollection(samples, self.parameters[:])

    def combine(
        self,
        other: "SampleCollection",
        how: Literal["full_factorial", "sample", "cycle"],
        rng: SeedLike | RNGLike | None = None,
    ) -> "SampleCollection":
        """Combine two SampleCollections into a new SampleCollection.

        Use this if you have two sets of samples for different parameters that
        you want to combine into a bigger set of samples across the combined set
        of parameters.

        If you want to simple combine two sets of samples for the same parameters,
        use `concat` instead.


        Parameters
        ----------
        other : the SampleCollection to combine with this one
        how : how to combine the designs.
        rng : RNG or None, only relevant in case combine is "sample"

        Returns
        -------
        a new SampleCollection instance

        Raises
        ------
        ValueError if one or more parameters with the same name are present in both collections

        """
        combined_parameters = self.parameters + other.parameters
        own_names = {p.name for p in self.parameters}
        other_names = {p.name for p in other.parameters}
        overlap = own_names & other_names
        if overlap:
            raise ValueError(
                f"the parameters {overlap} exist in both SampleCollections"
            )

        combined_samples = None
        samples_1 = self.samples
        samples_2 = other.samples
        match how:
            case "full_factorial":
                samples_1_repeated = np.repeat(
                    samples_1, repeats=samples_2.shape[0], axis=0
                )
                samples_2_tiled = np.tile(samples_2, (samples_1.shape[0], 1))
                combined_samples = np.hstack((samples_1_repeated, samples_2_tiled))
            case "sample" | "cycle":
                if samples_1.shape[0] == samples_2.shape[0]:
                    combined_samples = np.hstack((samples_1, samples_2))
                else:
                    longest, shortest = (
                        (samples_1, samples_2)
                        if samples_1.shape[0] > samples_2.shape[0]
                        else (samples_2, samples_1)
                    )
                    if how == "sample":
                        rng = np.random.default_rng(rng)
                        indices = rng.integers(0, shortest.shape[0], longest.shape[0])
                        upsampled = shortest[indices]
                    else:
                        n = math.ceil(longest.shape[0] / shortest.shape[0])
                        upsampled = np.tile(shortest, (n, 1))[0 : longest.shape[0], :]
                    combined_samples = np.hstack((longest, upsampled))
            case _:
                raise ValueError(
                    f"unknown value for combine, got {how}, should be one of full_factorial, sample, or cycle"
                )

        return SampleCollection(combined_samples, combined_parameters)

    def concat(self, other: "SampleCollection") -> "SampleCollection":
        """Concatenate two SampleCollections.

        Parameters
        ----------
        other : the SampleCollection to combine with this one

        Returns
        -------
        a new SampleCollection instance

        Raises
        ------
        ValueError if one or more parameters are present in only one of the two SampleCollections.

        """
        own_names = {p.name for p in self.parameters}
        other_names = {p.name for p in other.parameters}
        missing = own_names - other_names
        if missing:
            raise ValueError(
                f"the parameters {missing} do not exist in both SampleCollections"
            )

        samples_1 = self.samples
        samples_2 = other.samples
        combined_samples = np.vstack((samples_1, samples_2))
        return SampleCollection(combined_samples, self.parameters[:])

    def __add__(self, other: "SampleCollection") -> "SampleCollection":
        """Add a SampleCollections to this one."""
        return self.concat(other)


def sample_generator(
    samples: np.ndarray, params: list[Parameter]
) -> Generator[Sample, None, None]:
    """Return a generator yielding points instances.

    This generator iterates over the samples, and turns each row into a Sample and ensures datatypes are correctly handled.

    Parameters
    ----------
    samples : The samples taken for the parameters
    params : the Parameter instances that have been sampled

    Yields
    ------
    Sample


    """
    for sample in samples:
        design_dict = {}
        for param, value in zip(params, sample):
            if isinstance(param, IntegerParameter):
                value = int(value)  # noqa: PLW2901
            if isinstance(param, CategoricalParameter):
                # categorical parameter is an integer parameter, so
                # conversion to int is already done
                # boolean is a subclass of categorical with False and True as categories, so this is handled
                # via this route as well
                value = param.cat_for_index(value).value  # noqa: PLW2901

            design_dict[param.name] = value

        yield Sample(**design_dict)


def experiment_generator(
    models: Iterable["AbstractModel"],  # noqa: F821
    scenarios: Iterable[Sample],
    policies: Iterable[Sample],
    combine: Literal["full_factorial", "sample", "cycle"] = "full_factorial",
    rng: SeedLike | RNGLike | None = None,
) -> Generator[Experiment, None, None]:
    """Generator function which yields experiments.

    Parameters
    ----------
    models : list
    scenarios : iterable of scenarios
    policies : iterable of policies
    combine = {'full_factorial, sample', "cycle"}
              controls how to combine scenarios, policies, and models
              into experiments.
    rng : a numpy random number generator, or a value to seed a generator.

    Notes
    -----
    if combine is full_factorial' then this generator is essentially three nested
    loops: for each model, for each policy, for each scenario,
    return the experiment. This means that scenarios should not be a generator
    because this will be exhausted after the running the first policy on the
    first model.
    if combine is 'cycle' then this generator cycles over scenarios, policies
    and models until the longest of the three collections is
    exhausted.

    """
    # TODO combine_ functions can be made more generic
    # basically combine any collection
    # wrap around to yield specific type of class (e.g. point)

    match combine:
        case "full_factorial":
            jobs = itertools.product(models, policies, scenarios)
        case "sample":
            jobs = sample(models, policies, scenarios, rng=rng)
        case "cycle":
            jobs = zip_cycle(models, policies, scenarios)
        case _:
            raise ValueError(
                f"{combine} is unknown value for combine, use 'full_factorial', 'cycle', or 'sample'"
            )

    for i, job in enumerate(jobs):
        model, policy, scenario = job
        name = f"{model.name} {policy.name} {i}"
        experiment = Experiment(name, model.name, policy, scenario, i)
        yield experiment


def zip_cycle(*args):
    """Helper function for cycling over zips."""
    max_len = max(len(a) for a in args)
    return itertools.islice(zip(*(itertools.cycle(a) for a in args)), max_len)


def sample(models, policies, scenarios, rng=None):
    rng = np.random.default_rng(rng)
    max_length = max(len(models), len(policies), len(scenarios))

    def upsample(collection, size):
        indices = rng.integers(0, len(collection), size)
        return [collection[i] for i in indices]

    if len(models) != max_length:
        models = upsample(models, max_length)
    if len(policies) != max_length:
        policies = upsample(policies, max_length)
    if len(scenarios) != max_length:
        scenarios = upsample(scenarios, max_length)

    return zip(models, policies, scenarios)


def from_experiments(
    experiments: pd.DataFrame, drop_defaults: bool = True
) -> list["Sample"]:
    """Generate scenarios from an existing experiments DataFrame.

    This function takes a pandas DataFrame and turns it into a list of Sample instances.
    There is no further processing done, so it is up to the user to ensure that the columsn in the
    dataframe map to parameters in the model.


    Parameters
    ----------
    experiments : DataFrame
    drop_defaults : bool
                    By default, an experiments dataframe as returned by the workbench after performing experiments contains
                    a 'model', 'scenario' and 'policy' column. If drop_defaults is True, these columns are automatically ignored.

    Returns
    -------
    list of Sample instances

    """
    if drop_defaults:
        experiments = experiments.drop(
            ["model", "scenario", "policy"], axis=1, inplace=False
        )

    samples = []
    for record in experiments.to_dict(orient="records"):
        samples.append(Sample(**record))

    return samples
