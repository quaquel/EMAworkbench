************************
Installing the workbench
************************

The workbench is presently not yet available via any of the python
package managers such as pip. The code is available from `github <https://github.com/quaquel/EMAworkbench.>'_
The code comes with a requirements.txt file that indicates the key 
dependencies. Basically, if you have a standard scientific computing 
distribution for python such as the Anaconda distribution, most of the 
dependencies will already be met. 

In addition to the libraries available in Anaconda, you will need `deap`,
`jpype` for netlogo support, `mpld3` for interactive plots, and pydot and 
Graphviz for some of the visualizations. Of these, deap is essential, the
others are optional and the code should run for the large part fine without it.