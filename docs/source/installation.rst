************************
Installing the workbench
************************


The 2.x version of the workbench requires Python 3.8 or higher.

A stable version of the workbench can be installed via pip. ::

	pip install ema_workbench
	
The latest commit on the master branch can be installed with::

	pip install -e git+https://github.com/projectmesa/EMAworkbench#egg=ema-workbench

Or any other (development) branch on this repo or your own fork::

	pip install -e git+https://github.com/YOUR_FORK/EMAworkbench@YOUR_BRANCH#egg=ema-workbench

The code is also available from `github <https://github.com/quaquel/EMAworkbench>`_.
The code comes with a `requirements.txt <https://github.com/quaquel/EMAworkbench/blob/master/requirements.txt>`_ file that indicates the key
dependencies. Basically, if you have a standard scientific computing
distribution for Python such as the Anaconda distribution, most of the
dependencies will already be met.


In addition to the libraries available in the default Anaconda distribution,
there are various optional dependencies. Please follow the installation
instructions for each of these libraries.

From conda or `conda-forge <https://conda-forge.org/docs/user/introduction.html>`_:

* `altair <https://altair-viz.github.io>`_ for interactive visualizations
* `ipyparallel <https://ipyparallel.readthedocs.io>`_ for support of interactive multiprocessing within
   the jupyter notebook. ::

	conda install altair ipyparallel

There are also some pip based dependencies:

* `SALib <https://salib.readthedocs.io/en/latest/>`_ , this is a necessary
  dependency for advanced senstivity analysis.
* `platypus-opt <https://github.com/Project-Platypus/Platypus>`_ , this is an
  optional dependency for many-objective optimization functionality.
* `pydot <https://pypi.python.org/pypi/pydot/>`_ and  Graphviz for some of the
  visualizations. ::

	pip install SALib platypus-opt pydot

The various connectors have their own specific requirements.

* Vensim only works on Windows. If you have 64 bit vensim, you need 64 bit Python.
  If you have 32 bit vensim, you will need 32 bit Python.
* Excel also only works on Windows.
* `jpype-1 <https://jpype.readthedocs.io/en/latest/>`_ and
  `pynetlogo <https://pynetlogo.readthedocs.io>`_ for NetLogo
* `pysd <https://pysd.readthedocs.io/en/master/>`_ optional for simple vensim models
* `pythonnet <https://pypi.org/project/pythonnet/>`_ for Simio

::

	conda install jpype1
	pip install pynetlogo pysd pythonnet