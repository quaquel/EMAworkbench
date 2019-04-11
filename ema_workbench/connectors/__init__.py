import warnings
from contextlib import contextmanager

warnings.simplefilter("once", ImportWarning)


@contextmanager
def catch_and_ignore_import_warning():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=ImportWarning)
        yield


try:
    from . import vensim
except ImportError:
    warnings.warn("vensim connector not available", ImportWarning)

try:
    from . import excel
except ImportError:
    warnings.warn("excel connector not available", ImportWarning)

try:
    from . import netlogo
except ImportError:
    warnings.warn("netlogo connector not available", ImportWarning)

with catch_and_ignore_import_warning():
    try:
        from . import pysd_connector
    except ImportError:
        warnings.warn("pysd connector not available", ImportWarning)
