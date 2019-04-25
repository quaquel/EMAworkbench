************************
Installing the workbench
************************


The 2.x version of the workbench requires Python 3.6 or higher. 

A stable version of the workbench can be installed via pip. ::

	pip install ema_workbench

The code is also available from `github <https://github.com/quaquel/EMAworkbench>`_.
The code comes with a requirements.txt file that indicates the key 
dependencies. Basically, if you have a standard scientific computing 
distribution for Python such as the Anaconda distribution, most of the 
dependencies will already be met. 


In addition to the libraries available in the default Anaconda distribution,
there are various optional dependencies. From conda or conda forge
* `seaborn <https://web.stanford.edu/~mwaskom/software/seaborn/>`_ for enhanced matplotlib figures,  
* `pydot <https://pypi.python.org/pypi/pydot/>`_ and  Graphviz for some of the visualizations. 
* `altair <https://altair-viz.github.io>`_ for interactive visualizations
* `ipyparallel <http://ipyparallel.readthedocs.io/en/latest/>`_ for support of interactive multiprocessing within the jupyter notebook. 
* `jpype-1` <https://jpype.readthedocs.io/en/latest/>`_ for controlling NetLogo

There are also som pip based dependencies. These are
* `Salib <https://salib.readthedocs.io/en/latest/>`_, this is a necessary dependency for advanced senstivity analysis
* `platypus-opt <https://github.com/Project-Platypus/Platypus>`_, this is an optional dependency for many-objective optimization functionality
* `pynetlogo <https://pynetlogo.readthedocs.io>`_ optional for netlogo control
* `pysd <https://pysd.readthedocs.io/en/master/>`_ optional for simple vensim models

The various connectors have their own specific requirements. Vensim
only works on Windows and requires 32 bit Python. Excel also only works
on Windows.
