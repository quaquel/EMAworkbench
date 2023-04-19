************************
Installing the workbench
************************

From version 2.5.0 the workbench requires Python 3.9 or higher. Version 2.0.0 to 2.4.x support Python 3.8+.

Regular installations
#####################

A stable version of the workbench can be installed via pip. ::

	pip install ema_workbench

This installs the EMAworkbench together with all the bare necessities to run Python models.

If you want to upgrade the workbench from a previous version, add ``-U`` (or ``--upgrade``) to the pip command. ::

	pip install -U ema_workbench

We have a few more install options available, which install optional dependencies not always necessary but either nice to have or for specific functions. These can be installed with so called "extras" using pip.

Therefore we recommended installing with::

	pip install -U ema_workbench[recommended]

Which currently includes everything needed to use the workbench in Jupyter notebooks, with interactive graphs and to successfully complete the tests with pytest.

However, the EMAworkbench can connect to many other simulation software, such as Netlogo, Simio, Vensim (pysd) and Vadere. For these there are also extras available::

	pip install -U ema_workbench[netlogo,simio,pysd]

Note that the Netlogo and Simio extras need Windows as OS.

These extras can be combined. If you're going to work with Netlogo for example, you can do::

	pip install -U ema_workbench[recommended,netlogo]

You can use ``all`` to install all dependencies, except the connectors. Prepare for a large install. ::

	pip install -U ema_workbench[all]

These are all the extras that are available:

- ``jupyter`` installs ``["jupyter", "jupyter_client", "ipython", "ipykernel"]``
- ``dev`` installs ``["pytest", "jupyter_client", "ipyparallel"]``
- ``cov`` installs ``["pytest-cov", "coverage", "coveralls"]``
- ``docs`` installs ``["sphinx", "nbsphinx", "myst", "pyscaffold"]``
- ``graph`` installs ``["altair", "pydot", "graphviz"]``
- ``parallel`` installs ``["ipyparallel", "traitlets"]``

- ``netlogo`` installs ``["jpype-1", "pynetlogo"]``
- ``pysd`` installs ``["pysd"]``
- ``simio`` installs ``["pythonnet"]``

Then ``recommended`` is currently equivalent to ``jupyter,dev,graph`` and ``all`` installs everything, except the connectors. These are defined in the ``pyproject.toml`` file.

Developer installations
#######################

As a developer you will want an edible install, in which you can modify the installation itself. This can be done by adding ``-e`` (for edible) to the pip command. ::

	pip install -e ema_workbench

The latest commit on the master branch can be installed with::

	pip install -e git+https://github.com/quaquel/EMAworkbench#egg=ema-workbench

Or any other (development) branch on this repo or your own fork::

	pip install -e git+https://github.com/YOUR_FORK/EMAworkbench@YOUR_BRANCH#egg=ema-workbench

The code is also available from `github <https://github.com/quaquel/EMAworkbench>`_.

Limitations
###########

Some connectors have specific limitations, listed below.

* Vensim only works on Windows. If you have 64-bit Vensim, you need 64-bit Python.
  If you have 32-bit Vensim, you will need 32-bit Python.
* Excel also only works on Windows.
