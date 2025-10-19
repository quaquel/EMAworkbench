"""helper stuff for analyzing converngence of optimization results."""

from platypus import (
    EpsilonIndicator,
    GenerationalDistance,
    Hypervolume,
    InvertedGenerationalDistance,
    Spacing,
)

from ..util import get_module_logger
from .optimization import rebuild_platypus_population

__all__ = [
    "EpsilonIndicatorMetric",
    "GenerationalDistanceMetric",
    "HypervolumeMetric",
    "InvertedGenerationalDistanceMetric",
    "SpacingMetric",
]

_logger = get_module_logger(__name__)


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
        reference_set = rebuild_platypus_population(reference_set, problem)
        try:
            super().__init__(**kwargs)
        except TypeError:
            super().__init__(reference_set, **kwargs)
        self.problem = problem

        # super().__init__(reference_set=reference_set, **kwargs)

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class SpacingMetric(MetricWrapper, Spacing):
    """Spacing metric.

    Parameters
    ----------
    problem : PlatypusProblem instance


    this is a thin wrapper around Spacing as provided
    by platypus to make it easier to use in conjunction with the
    workbench.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
