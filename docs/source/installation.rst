************************
Installing the workbench
************************

The workbench is presently not yet available via any of the Python
package managers such as pip. The code is available from `github <https://github.com/quaquel/EMAworkbench>`_.
The code comes with a requirements.txt file that indicates the key 
dependencies. Basically, if you have a standard scientific computing 
distribution for Python such as the Anaconda distribution, most of the 
dependencies will already be met. 

Currently, the workbench only works with Python 2.7. Once you have downloaded 
the code, do not forget to add the directory where the code is located to 
Python's search path. ::  

   import sys
   sys.path.append(“./EMAworkbench/src/”) # or whatever your directory is

In addition to the libraries available in Anaconda, you will need 
`deap <https://pypi.python.org/pypi/deap/>`_, 
`jpype <http://jpype.readthedocs.org/en/latest/>`_ for NetLogo support, 
`mpld3 <http://mpld3.github.io/>`_ for interactive plots, 
`seaborn <https://web.stanford.edu/~mwaskom/software/seaborn/>`_ for enhanced
matplotlib figures,  and `pydot <https://pypi.python.org/pypi/pydot/>`_ 
and  Graphviz for some of the visualizations. Of these, deap and
seaborn are essential, the others are optional and the code should largely run 
fine for the without them.

The connectors to Excel and Vensim will only work on a Windows machine. The
Excel connector depends on pywin32 and requires Excel. Vensim support is 
available only for the DSS version and requires that the vensim DLL is 
installed. Vensim support also is limited to 32 bit Python only.  