[![Build Status](https://github.com/quaquel/EMAworkbench/actions/workflows/ci.yml/badge.svg?master)](https://github.com/quaquel/EMAworkbench/actions)
[![Coverage Status](https://coveralls.io/repos/github/quaquel/EMAworkbench/badge.svg?branch=master)](https://coveralls.io/github/quaquel/EMAworkbench?branch=master)
[![Documentation Status](https://readthedocs.org/projects/emaworkbench/badge/?version=latest)](http://emaworkbench.readthedocs.org/en/latest/?badge=master)
[![PyPi](https://img.shields.io/pypi/v/ema_workbench.svg)](https://pypi.python.org/pypi/ema_workbench)
[![PyPi](https://img.shields.io/pypi/dm/ema_workbench.svg)](https://pypi.python.org/pypi/ema_workbench)

# Exploratory Modeling workbench

Exploratory Modeling and Analysis (EMA) is a research methodology that uses computational experiments to analyze complex and uncertain systems ([Bankes, 1993](http://www.jstor.org/stable/10.2307/171847)). That is, exploratory modeling aims at offering computational decision support for decision making under [deep uncertainty](http://inderscience.metapress.com/content/y77p3q512x475523/) and [robust decision making](http://en.wikipedia.org/wiki/Robust_decision_making).

The EMA workbench aims at providing support for performing exploratory modeling with models developed in various modelling packages and environments. Currently, the workbench offers connectors to [Vensim](https://vensim.com/), [Netlogo](https://ccl.northwestern.edu/netlogo/), [Simio](https://www.simio.com/), [Vadere](https://www.vadere.org/) and Excel.

The EMA workbench offers support for designing experiments, performing the experiments - including support for parallel processing on both a single machine as well as on clusters-, and analysing the results. To get started, take a look at the high level overview, the tutorial, or dive straight into the details of the API.

The EMA workbench currently under development at Delft University of Technology. If you would like to collaborate, open an issue/discussion or contact [Jan Kwakkel](https://www.tudelft.nl/en/tpm/our-faculty/departments/multi-actor-systems/people/professors/prof-drir-jh-jan-kwakkel).

## Documentation

Documentation for the workbench is availabe at [Read the Docs](https://emaworkbench.readthedocs.io/en/latest/index.html), including an introduction on Exploratory Modeling, tutorials and documentation on all the modules and functions.

There are also a lot of example models available at [ema_workbench/examples](ema_workbench/examples), both for pure Python models and some using the different connectors. A release notes for each new version are available at [CHANGELOG.md](CHANGELOG.md).

## Installation

The workbench is available from [PyPI](https://pypi.org/project/ema-workbench/), and currently requires Python 3.8 or newer. It can be installed with:
```
pip install -U ema_workbench
```
To also install some recommended packages for plotting, testing and Jupyter support, use the `recommended` extra:
```
pip install -U ema_workbench[recommended]
```
There are way more options installing the workbench, including installing connector packages, edible installs for development, installs of custom forks and branches and more. See [Installing the workbench](https://emaworkbench.readthedocs.io/en/latest/installation.html) in the docs for all options.

## Contributing

We greatly appreciate contributions to the EMA workbench! Reporting [Issues](https://github.com/quaquel/EMAworkbench/issues) such as bugs or unclairties in the documentation, opening a [Pull requests](https://github.com/quaquel/EMAworkbench/pulls) with code or documentation improvements or opening a [Discussion](https://github.com/quaquel/EMAworkbench/discussions) with a question, suggestions or comment helps us a lot.

Please check [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

## License

This repository is licensed under BSD 3-Clause License. See [LICENSE.md](LICENSE.md).
