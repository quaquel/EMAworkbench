.. meta::
   :description: A python library for exploratory modeling and analysis for 
                 supporting model based decision making under deep uncertainty
   :keywords: exploratory modeling, deep uncertainty, robust decision making,
              vensim, python

*********************
A high level overview
*********************

   * :ref:`simulation-control`
   * :ref:`connectors`
   * :ref:`analysis`
   * :ref:`visualization`

.. _simulation-control:

====
Core
====

* Model ensemble (:mod:`expWorkbench.model_ensemble`: the class responsbile 
  for setting up and performing experiments
* Model (:mod:`expWorkbench.model`: an abstract base class for specifying
  the interface to the model on which you want to perform exploratory modeling.
  Default implementiations are provided for Vensim, Netlogo, and Excel. 
* Samplers (:mod:`expWorkbench.samplers`: the various sampling techniques
  that are readily available in the workbench.


.. _connectors:

==========
Connectors
==========
* Vensim connector (:mod:`vensim`): This enables controlling (e.g. setting 
  parameters, simulation setup, run, get output, etc.) a simulation model that 
  is built in Vensim software, and conducting an EMA study based on this model.
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
* Patient Rule Induction Method (:mod:`prim`) 
* Classification Trees (:mod:`cart`)
* Feature Scoring (:mod:`feature_scoring`)
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

.. _visualization:

=============
Visualization
=============
* lines, envelopes, multiplot graphs (:mod:`plotting`)
* pair wise plots (:mod:`pairs_plotting`)
* support for converting figures to black and white (:mod:`b_an_w_plotting`) 

