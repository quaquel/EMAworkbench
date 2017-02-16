import sys
try:
    from . import vensim
except ImportError:
    sys.stderr.write("vensim connector not available\n")

try:
    from . import excel
except ImportError:
    sys.stderr.write("excel connector not available\n")

try:
    from . import netlogo
except ImportError:
    sys.stderr.write("netlogo connector not available\n")

try:
    from . import pysd_connector
except ImportError:
    sys.stderr.write("pysd connector not available\n")

del sys