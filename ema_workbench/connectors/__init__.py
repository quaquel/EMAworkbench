"""Connectors for various modeling packages."""

import warnings
from contextlib import contextmanager

from ema_workbench import EMAError

warnings.simplefilter("once", ImportWarning)


@contextmanager
def catch_and_ignore_import_warning():
    """Helper context manager to catch and ignore import warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ImportWarning)
        yield


try:
    from . import vensim  # noqa F401
except ImportError:
    warnings.warn("vensim connector not available", ImportWarning, stacklevel=2)

try:
    from . import excel  # noqa F401
except ImportError:
    warnings.warn("excel connector not available", ImportWarning, stacklevel=2)

try:
    from . import netlogo  # noqa F401
except ImportError:
    warnings.warn("netlogo connector not available", ImportWarning, stacklevel=2)

try:
    from . import simio_connector  # noqa F401
except ImportError:
    warnings.warn("simio connector not available", ImportWarning, stacklevel=2)
except EMAError:
    warnings.warn("simio not found, connector not available", ImportWarning, stacklevel=2)


with catch_and_ignore_import_warning():
    try:
        from . import pysd_connector  # noqa F401
    except ImportError:
        warnings.warn("pysd connector not available", ImportWarning, stacklevel=2)
