************************
Installing the workbench
************************

A development version of the workbench is now available via pip (pip install 
ema_workbench). The code is also available from `github <https://github.com/quaquel/EMAworkbench>`_.
The code comes with a requirements.txt file that indicates the key 
dependencies. Basically, if you have a standard scientific computing 
distribution for Python such as the Anaconda distribution, most of the 
dependencies will already be met. 

The workbench works with both Python 2.7, and Python 3.5. If you download the
code from github, do not forget to add the directory where the code is located to 
Python's search path. ::  

   import sys
   sys.path.append(“./EMAworkbench/src/”) # or whatever your directory is

In addition to the libraries available in Anaconda, you will need 
`jpype <http://jpype.readthedocs.org/en/latest/>`_ for NetLogo support, 
`mpld3 <http://mpld3.github.io/>`_ for interactive plots, 
`seaborn <https://web.stanford.edu/~mwaskom/software/seaborn/>`_ for enhanced
matplotlib figures,  and `pydot <https://pypi.python.org/pypi/pydot/>`_ 
and  Graphviz for some of the visualizations. Of these, seaborn and mpld3 are
essential for the analysis package, the others are optional and the code should 
largely run fine for the without them.

The connectors to Excel and Vensim will only work on a Windows machine. The
Excel connector depends on pywin32 and requires Excel. Vensim support is 
available only for the DSS version and requires that the vensim DLL is 
installed. Vensim support also is limited to 32 bit Python only.  