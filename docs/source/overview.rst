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
* Uncertainties (:mod:`ema_workbench.em_framework.parameters`): various 
  types of parameter classes that can be used to specify the uncertainties
  and/or levers on the model
* Outcomes (:mod:`ema_workbench.em_framework.outcomes`): various types
  of outcome classes
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

* Vensim connector (:mod:`ema_workbench.connectors.vensim`): This enables
  controlling (e.g. setting parameters, simulation setup, run, get output, etc
  .) a simulation model that is built in Vensim software, and conducting an
  EMA study based on this model.
* Excel connector (:mod:`ema_workbench.connectors.excel`): This enables
  controlling models build in Excel.
* NetLogo connector (:mod:`ema_workbench.connectors.netlogo`): This enables
  controlling (e.g. setting parameters, simulation setup, run, get output, etc
  .) a simulation model that is built in NetLogo software, and conducting an
  EMA study based on this model.
* Simio connector (:mod:`ema_workbench.connectors.simio_connector`): This
  enables controlling models built in Simio
* Pysd connector (:mod:`ema_workbench.connectors.pysd_connector`)


.. _analysis:

========
Analysis
========

The analysis package contains a variety of analysis and visualization 
techniques for analyzing the results from the exploratory modeling. The 
analysis scripts are tailored for use in combination with the workbench, but 
they can also be used on their own with data generated in some other manner.

* Patient Rule Induction Method (:mod:`ema_workbench.analysis.prim`) 
* Classification Trees (:mod:`ema_workbench.analysis.cart`)
* Logistic Regression (:mod:`ema_workbench.analysis.logistic_regression`)
* Dimensional Stacking (:mod:`ema_workbench.analysis.dimensional_stacking`)
* Feature Scoring (:mod:`ema_workbench.analysis.feature_scoring`)
* Regional Sensitivity Analysis (:mod:`ema_workbench.analysis.regional_sa`)
* various plotting functions for time series data (:mod:`ema_workbench.analysis.plotting`)
* pair wise plots (:mod:`ema_workbench.analysis.pairs_plotting`)
* parallel coordinate plots (:mod:`ema_workbench.analysis.parcoords`)
* support for converting figures to black and white (:mod:`ema_workbench.analysis.b_an_w_plotting`) 

