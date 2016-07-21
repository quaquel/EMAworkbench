import sys
try:
    import vensim
except ImportError:
    sys.stderr.write("vensim connector not available\n")

try:
    import excel
except ImportError:
    sys.stderr.write("excel connector not available\n")

try:
    import netlogo
except ImportError:
    sys.stderr.write("netlogo connector not available\n")

try:
    from pysd_connector import PySDConnector
except ImportError:
    sys.stderr.write("PySD library not installed\n")

del sys