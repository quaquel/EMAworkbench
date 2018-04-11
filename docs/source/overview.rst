.. meta::
   :description: A python library for exploratory modeling and analysis for 
                 supporting model based decision making under deep uncertainty
   :keywords: exploratory modeling, deep uncertainty, robust decision making,
              vensim, python

*********************
A High Level Overview
*********************

   * :ref:`emframework`
   * :ref:`connectors`
   * :ref:`analysis`


.. _emframework:

==============================
Exploratory modeling framework
==============================

The core package contains the core functionality for setting up, designing,
and performing series of computational experiments on one or more models 
simultaneously. 

* Model (:mod:`ema_workbench.em_framework.model`): an abstract base class for 
  specifying the interface to the model on which you want to perform 
  exploratory modeling.
* Samplers (:mod:`ema_workbench.em_framework.samplers`): the various sampling 
  techniques that are readily available in the workbench.
* Uncertainties (:mod:`ema_workbench.em_framework.uncertainties`): various 
  types of uncertainty classes that can be used to specify the uncertainties in 
  a model interface.
* Evaluators (:mod:`ema_workbench.em_framework.evaluators`): various evaluators
  for running experiments in sequence or in parallel.

.. _connectors:

==========
Connectors
==========

The connectors package contains connectors to some existing simulation modeling
environments. For each of these, a standard ModelStructureInterface class is
provided that users can use as a starting point for specifying the interface
to their own model. 

* Vensim connector (:mod:`vensim`): This enables controlling (e.g. setting 
  parameters, simulation setup, run, get output, etc.) a simulation model that 
  is built in Vensim software, and conducting an EMA study based on this model.
* Pysd connector (:mod:`pysd_connector`)
* Excel connector (:mod:`excel`): This enables controlling models build in 
  Excel.
* NetLogo connector (:mod:`netlogo.py`): This enables controlling
  (e.g. setting parameters, simulation setup, run, get output, etc.) a 
  simulation model that is built in NetLogo software, and conducting an EMA 
  study based on this model.


.. _analysis:

========
Analysis
========

The ananlysis package contains a variety of analysis and visualization 
techniques for analysing the results from the exploratory modeling. The 
analysis scripts are tailored for use in combination with the workbench, but 
they can also be used on their own with data generated in some other manner.

* Patient Rule Induction Method (:mod:`prim`) 
* Classification Trees (:mod:`cart`)
* Feature Scoring (:mod:`feature_scoring`)
* Regional Sensitivity Analysis (:mod:`regional_sa`)
* Dimensional Stacking (:mod:`dimensional_stacking`)
* Behaviour clustering (:mod:`clusterer`): This analysis feature automatically 
  allocates output behaviours that are similar in characteristics to groups 
  (i.e. clusters). 'Similarity' between dynamic behaviours is defined using 
  distance functions, and the feature can operate using different distance 
  functions that measure the (dis)similarity very differently. Currently 
  available distances are as follows;
  
   * Behaviour Mode Distance (:func:`distance_gonenc`): A distance that 
     focuses purely on qualitative pattern features. For example, two S-shaped 
     curves that are very different in initial level, take-off point, final 
     value, etc. are evaluated as identical according to BM distance since both 
     have identical qualitaive characteristics of an S-shaped behaviour 
     (i.e. a constant early phase, then growth with increasing rate, then 
     growth with decreasing rate and terminate with a constant late phase)
     on their differences in these three features.
   * Sum of squared error (:func:`distance_sse`): See any statistics text.
   * Mean square error (:func:`distance_mse`): See any statistics text.

* various plotting functions (:mod:`plotting`)
* pair wise plots (:mod:`pairs_plotting`)
* support for converting figures to black and white (:mod:`b_an_w_plotting`) 

