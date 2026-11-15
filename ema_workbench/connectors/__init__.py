"""Connectors for various modeling packages.

Each connector is a module of this package, for example
``ema_workbench.connectors.vensim``, which is imported on first use. A
connector of which the dependencies are not installed raises an ImportError
when it is imported, so there are no warnings for connectors that are not
used. This also avoids that every worker process repeats such warnings when
running experiments in parallel.

"""

import importlib
import warnings
from contextlib import contextmanager

__all__ = [
    "excel",
    "netlogo",
    "pysd_connector",
    "simio_connector",
    "vadere",
    "vensim",
]


@contextmanager
def catch_and_ignore_import_warning():
    """Helper context manager to catch and ignore import warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ImportWarning)
        yield


def __getattr__(name):
    """Import a connector on first use, e.g. ``connectors.vensim``."""
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
