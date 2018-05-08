************************
Installing the workbench
************************

A stable version of the workbench can be installed via pip. ::

	pip install ema_workbench

The code is also available from `github <https://github.com/quaquel/EMAworkbench>`_.
The code comes with a requirements.txt file that indicates the key 
dependencies. Basically, if you have a standard scientific computing 
distribution for Python such as the Anaconda distribution, most of the 
dependencies will already be met. 

The workbench works with both Python 2.7, and Python 3.5 or higher. If 
you download the code from github, do not forget to add the directory 
where the code is located to Python's search path. ::  

   import sys
   sys.path.append(“./EMAworkbench/src/”) # or whatever your directory is

You can also use pip to install from GitHub directly ::

	pip install git+https://github.com/quaquel/EMAworkbench.git

In addition to the libraries available in Anaconda, you will need  
`seaborn <https://web.stanford.edu/~mwaskom/software/seaborn/>`_ for 
enhanced matplotlib figures,  and `pydot <https://pypi.python.org/pypi/pydot/>`_ 
and  Graphviz for some of the visualizations. Of these, seaborn is
essential for the analysis package, the others are optional and the 
code should  largely run fine for the without them. You will also need
`ipyparallel <http://ipyparallel.readthedocs.io/en/latest/>`_ for 
support of interactive multiprocessiging within the jupyter notebook. 

The various connectors have their own specific dependencies. Vensim
only works on Windows and requires 32 bit Python. Excel also only works
on Windows. Netlogo requires 
`jpype <http://jpype.readthedocs.org/en/latest/>`_  and
`pyNetlogo <https://pynetlogo.readthedocs.io/en/latest/>`_ for NetLogo 
support. Please follow the installation instructions provided there. 
PySD support is dependent on the 
`PySD <http://pysd.readthedocs.io/en/master/>`_ library. 
