import abc
import warnings

import numpy as np
import pandas as pd

from platypus import (
    Hypervolume,
    GenerationalDistance,
    InvertedGenerationalDistance,
    EpsilonIndicator,
    Spacing,
)

from .optimization import rebuild_platypus_population
from .util import ProgressTrackingMixIn
from ..util import get_module_logger

__all__ = [
    "Convergence",
    "EpsilonIndicatorMetric",
    "EpsilonProgress",
    "GenerationalDistanceMetric",
    "HypervolumeMetric",
    "InvertedGenerationalDistanceMetric",
    "OperatorProbabilities",
    "SpacingMetric",
]

_logger = get_module_logger(__name__)

class AbstractConvergenceMetric(abc.ABC):
    """Base convergence metric class."""

    def __init__(self, name):
        """Init."""
        super().__init__()
        self.name = name
        self.results = []

    @abc.abstractmethod
    def __call__(self, optimizer):
        """Call the convergence metric."""

    def reset(self):
        self.results = []

    def get_results(self):
        return self.results


class EpsilonProgress(AbstractConvergenceMetric):
    """Epsilon progress convergence metric class."""

    def __init__(self):
        """Init."""
        super().__init__("epsilon_progress")

    def __call__(self, optimizer):  # noqa: D102
        self.results.append(optimizer.archive.improvements)


class MetricWrapper:
    """Wrapper class for wrapping platypus indicators.

    Parameters
    ----------
    reference_set : DataFrame
    problem : PlatypusProblem instance
    kwargs : dict
             any additional keyword arguments to be passed
             on to the wrapper platypus indicator class

    Notes
    -----
    this class relies on multi-inheritance and careful consideration
    of the MRO to conveniently wrap the convergence metrics provided
    by platypus.

    """

    def __init__(self, reference_set, problem, **kwargs):
        self.problem = problem
        reference_set = rebuild_platypus_population(reference_set, self.problem)
        super().__init__(reference_set=reference_set, **kwargs)

    def calculate(self, archive):
        solutions = rebuild_platypus_population(archive, self.problem)
        return super().calculate(solutions)


class HypervolumeMetric(MetricWrapper, Hypervolume):
    """Hypervolume metric.

    Parameters
    ----------
    reference_set : DataFrame
    problem : PlatypusProblem instance

    this is a thin wrapper around Hypervolume as provided
    by platypus to make it easier to use in conjunction with the
    workbench.

    """


class GenerationalDistanceMetric(MetricWrapper, GenerationalDistance):
    """GenerationalDistance metric.

    Parameters
    ----------
    reference_set : DataFrame
    problem : PlatypusProblem instance
    d : int, default=1
        the power in the intergenerational distance function


    This is a thin wrapper around GenerationalDistance as provided
    by platypus to make it easier to use in conjunction with the
    workbench.

    see https://link.springer.com/content/pdf/10.1007/978-3-319-15892-1_8.pdf
    for more information

    """


class InvertedGenerationalDistanceMetric(MetricWrapper, InvertedGenerationalDistance):
    """InvertedGenerationalDistance metric.

    Parameters
    ----------
    reference_set : DataFrame
    problem : PlatypusProblem instance
    d : int, default=1
        the power in the inverted intergenerational distance function


    This is a thin wrapper around InvertedGenerationalDistance as provided
    by platypus to make it easier to use in conjunction with the
    workbench.

    see https://link.springer.com/content/pdf/10.1007/978-3-319-15892-1_8.pdf
    for more information

    """


class EpsilonIndicatorMetric(MetricWrapper, EpsilonIndicator):
    """EpsilonIndicator metric.

    Parameters
    ----------
    reference_set : DataFrame
    problem : PlatypusProblem instance


    this is a thin wrapper around EpsilonIndicator as provided
    by platypus to make it easier to use in conjunction with the
    workbench.

    """


class SpacingMetric(MetricWrapper, Spacing):
    """Spacing metric.

    Parameters
    ----------
    problem : PlatypusProblem instance


    this is a thin wrapper around Spacing as provided
    by platypus to make it easier to use in conjunction with the
    workbench.

    """

    def __init__(self, problem):
        self.problem = problem


# class HyperVolume(AbstractConvergenceMetric):
#     """Hypervolume convergence metric class.
#
#     This metric is derived from a hyper-volume measure, which describes the
#     multi-dimensional volume of space contained within the pareto front. When
#     computed with minimum and maximums, it describes the ratio of dominated
#     outcomes to all possible outcomes in the extent of the space.  Getting this
#     number to be high or low is not necessarily important, as not all outcomes
#     within the min-max range will be feasible.  But, having the hypervolume remain
#     fairly stable over multiple generations of the evolutionary algorithm provides
#     an indicator of convergence.
#
#     Parameters
#     ----------
#     minimum : numpy array
#     maximum : numpy array
#
#
#     This class is deprecated and will be removed in version 3.0 of the EMAworkbench.
#     Use ArchiveLogger instead and calculate hypervolume in post using HypervolumeMetric
#     as also shown in the directed search tutorial.
#
#     """
#
#     def __init__(self, minimum, maximum):
#         super().__init__("hypervolume")
#         warnings.warn(
#             "HyperVolume is deprecated and will be removed in version 3.0 of the EMAworkbench."
#             "Use ArchiveLogger and HypervolumeMetric instead",
#             DeprecationWarning,
#             stacklevel=2,
#         )
#         self.hypervolume_func = Hypervolume(minimum=minimum, maximum=maximum)
#
#     def __call__(self, optimizer):
#         self.results.append(self.hypervolume_func.calculate(optimizer.archive))
#
#     @classmethod
#     def from_outcomes(cls, outcomes):
#         ranges = [o.expected_range for o in outcomes if o.kind != o.INFO]
#         minimum, maximum = np.asarray(list(zip(*ranges)))
#         return cls(minimum, maximum)


class OperatorProbabilities(AbstractConvergenceMetric):
    """OperatorProbabiliy convergence tracker for use with auto adaptive operator selection.

    Parameters
    ----------
    name : str
    index : int


    State of the art MOEAs like Borg (and GenerationalBorg provided by the workbench)
    use autoadaptive operator selection. The algorithm has multiple different evolutionary
    operators. Over the run, it tracks how well each operator is doing in producing fitter
    offspring. The probability of the algorithm using a given evolutionary operator is
    proportional to how well this operator has been doing in producing fitter offspring in
    recent generations. This class can be used to track these probabilities over the
    run of the algorithm.

    """

    def __init__(self, name, index):
        super().__init__(name)
        self.index = index

    def __call__(self, optimizer):  # noqa: D102
        try:
            props = optimizer.algorithm.variator.probabilities
            self.results.append(props[self.index])
        except AttributeError:
            pass


class Convergence(ProgressTrackingMixIn):
    """helper class for tracking convergence of optimization."""

    valid_metrics = {"hypervolume", "epsilon_progress", "archive_logger"}

    def __init__(
        self,
        metrics,
        max_nfe,
        convergence_freq=1000,
        logging_freq=5,
        log_progress=False,
    ):
        """Init."""
        super().__init__(
            max_nfe,
            logging_freq,
            _logger,
            log_progress=log_progress,
            log_func=lambda self: f"generation"
            f" {self.generation}, {self.i}/{self.max_nfe}",
        )

        self.max_nfe = max_nfe
        self.generation = -1
        self.index = []
        self.last_check = 0

        if metrics is None:
            metrics = []

        self.metrics = metrics
        self.convergence_freq = convergence_freq
        self.logging_freq = logging_freq

        # TODO what is the point of this code?
        for metric in metrics:
            assert isinstance(metric, AbstractConvergenceMetric)
            metric.reset()

    def __call__(self, optimizer, force=False):
        """Stores convergences information given specified convergence frequency.

        Parameters
        ----------
        optimizer : platypus optimizer instance
        force : boolean, optional
                if True, convergence information will always be stored
                if False, converge information will be stored if the
                the number of nfe since the last time of storing is equal to
                or higher then convergence_freq


        the primary use case for force is to force convergence frequency information
        to be stored once the stopping condition of the optimizer has been reached
        so that the final convergence information is kept.

        """
        nfe = optimizer.nfe
        super().__call__(nfe - self.i)

        self.generation += 1

        if (
            (nfe >= self.last_check + self.convergence_freq)
            or (self.last_check == 0)
            or force
        ):
            self.index.append(nfe)
            self.last_check = nfe

            for metric in self.metrics:
                metric(optimizer)

    def to_dataframe(self):  # noqa: D102
        progress = {
            metric.name: result
            for metric in self.metrics
            if (result := metric.get_results())
        }

        progress = pd.DataFrame.from_dict(progress)

        if not progress.empty:
            progress["nfe"] = self.index

        return progress
