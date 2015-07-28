import sys
try:
    import vensim
except ImportError:
    sys.stderr.write("vensim connector not available\n")

try:
    import excel
except ImportError:
    sys.stderr.write("excel connector not available\n")
del sys

try:
    import netlogo
except ImportError:
    sys.stderr.write("netlogo connector not available\n")
del sys