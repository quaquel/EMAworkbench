## Exploratory Modeling workbench
This is a stable version of the EMA workbench currently under 
development at Delft University of Technology.

[![Build Status](https://github.com/quaquel/EMAworkbench/actions/workflows/ci.yml/badge.svg?branch=2.1-inprogress)](https://github.com/quaquel/EMAworkbench/actions)
[![Coverage Status](https://coveralls.io/repos/github/quaquel/EMAworkbench/badge.svg?branch=2.1-inprogress)](https://coveralls.io/github/quaquel/EMAworkbench?branch=2.1-inprogress)
[![Documentation Status](https://readthedocs.org/projects/emaworkbench/badge/?version=latest)](http://emaworkbench.readthedocs.org/en/latest/?badge=master)
[![PyPi](https://img.shields.io/pypi/v/ema_workbench.svg)](https://pypi.python.org/pypi/ema_workbench)
[![PyPi](https://img.shields.io/pypi/dm/ema_workbench.svg)](https://pypi.python.org/pypi/ema_workbench)

If you are interested in using the most recent version of the workbench  and
would like to contribute to its further development, contact Jan Kwakkel at 
Delft University of Technology.  

The workbench is available from pip. Version 1.x is compatible with both
python 2 and 3, while the 2.x branch requires python 3.8 or newer.


# Releasing

Releases are published automatically when a tag is pushed to GitHub.

```bash

   # Set next version number
   export RELEASE=x.x.x

   # Create tags
   git commit --allow-empty -m "Release $RELEASE"
   git tag -a $RELEASE -m "Version $RELEASE"

   # Push
   git push upstream --tags # for a fork
   
   # use git push origin --tags if on origin
```