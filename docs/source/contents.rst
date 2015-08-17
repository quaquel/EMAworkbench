.. EMA workbench documentation master file, created by
   sphinx-quickstart on Wed Sep 07 13:56:32 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. meta::
   :description: A python library for exploratory modeling and analysis for 
                 supporting model based decision making under deep uncertainty
   :keywords: exploratory modeling, deep uncertainty, robust decision making,
              vensim, python

.. _contents:

.. _exploratory-modelling-analysis-workbench:

**************************************************
Exploratory Modelling and Analysis (EMA) Workbench
**************************************************

Exploratory Modeling and Analysis (EMA) is a research methodology that uses 
computational experiments to analyze complex and uncertain systems 
(`Bankes, 1993 <http://www.jstor.org/stable/10.2307/171847>`_).That is, 
exploratory modeling aims at offering computational decision support for 
decision making under `deep uncertainty <http://inderscience.metapress.com/content/y77p3q512x475523/>`_ 
and `Robust decision making <http://en.wikipedia.org/wiki/Robust_decision_making>`_.  

The EMA workbench is aimed at providing support for doing EMA on models 
developed in various modelling packages and environments. Currently, we focus 
on offering support for doing EMA on models developed in 
`Vensim <http://www.vensim.com/>`_, Excel, and Python. Future plans include
support for `Netlogo <http://ccl.northwestern.edu/netlogo/>`_ 
and `Repast <http://repast.sourceforge.net/>`_. The EMA workbench offers support 
for designing experiments, performing the experiments - including support for 
parallel processing-, and analysing the results. A key design principle is that 
people should be able to perform EMA on normal computers, instead of having 
to take recourse to a HPC.

The Exploratory Modeling and Analysis (EMA) Workbench is an evolving set of 
tools and methods. It evolved out of code written by Jan Kwakkel for his PhD 
research. The EMA workbench is implemented in Python and relies on 
`Numpy and Scipy <http://numpy.scipy.org/>`_.   

 * :ref:`exploratory-modelling-analysis-workbench`

    * :ref:`workbench-content`
 
      * :ref:`simulation-control`
      * :ref:`analysis`
      * :ref:`visualization`

 * :ref:`exploratory-modeling-and-analysis`

.. _workbench-content:

=================
Workbench Content
=================

.. _simulation-control:

^^^^^^^^^^^
controllers
^^^^^^^^^^^

* Vensim controller (:mod:`vensim`): This enables controlling (e.g. setting 
  parameters, simulation setup, run, get output, etc.) a simulation model that 
  is built in Vensim software, and conducting an EMA study based on this model.
* Excel controller (:mod:`excel`): This enables controlling models build in 
  Excel.
* NetLogo controller (:mod:`netlogo.py`): This enables controlling
  (e.g. setting parameters, simulation setup, run, get output, etc.) a 
  simulation model that is built in NetLogo software, and conducting an EMA 
  study based on this model.


.. _analysis:

^^^^^^^^
Analysis
^^^^^^^^
* Patient Rule Induction Method (:mod:`prim`) 
* Feature Scoring (:mod:`feature_scoring`)
* Classification Trees (:mod:`cart`)
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

^^^^^^^^^^^^^
Visualization
^^^^^^^^^^^^^
* lines, envelopes, multiplot graphs (:mod:`plotting`)
* pair wise plots (:mod:`pairs_plotting`)
* support for converting figures to black and white (:mod:`b_an_w_plotting`) 

.. _exploratory-modeling-and-analysis:

***************************************
Exploratory Modeling and Analysis (EMA)
***************************************
Exploratory Modeling and Analysis (EMA) is a research methodology that uses 
computational experiments to analyze complex and uncertain systems 
(Bankes, 1993, 1994). EMA can be understood as searching or sampling over an 
ensemble of models that are plausible, given a priori knowledge or are 
otherwise of interest. This ensemble may often be large or infinite in size. 
Consequently, the central challenge of exploratory modeling is the design of 
search or sampling strategies that support valid conclusions or reliable 
insights based on a limited number of computational experiments.

EMA can be contrasted with the use of models to predict system behavior, 
where models are built by consolidating known facts into a single package 
(Hodges, 1991). When experimentally validated, this single model can be used 
for analysis as a surrogate for the actual system. Examples of this approach 
include the engineering models that are used in computer-aided design systems. 
Where applicable, this *consolidative* methodology is a powerful technique for 
understanding the behavior of complex systems. Unfortunately, for many systems 
of interest, the construction of models that may be validly used as surrogates 
is simply not a possibility. This may be due to a variety of factors, including 
the infeasibility of critical experiments, impossibility of accurate 
measurements or observations, immaturity of theory, openness of the system to 
unpredictable outside perturbations, or nonlinearity of system behavior, but is 
fundamentally a matter of not knowing enough to make predictions 
(Campbell et al., 1985; Hodges and Dewar, 1992). For such systems, a 
methodology based on consolidating all known information into a single model 
and using it to make best estimate predictions can be highly misleading.

EMA can be useful when relevant information exists that can be exploited by 
building models, but where this information is insufficient to specify a single 
model that accurately describes system behavior. In this circumstance, models 
can be constructed that are consistent with the available information, but such 
models are not unique. Rather than specifying a single model and falsely 
treating it as a reliable image of the target system, the available information 
is consistent with a set of models, whose implications for potential decisions 
may be quite diverse. A single model run drawn from this potentially infinite 
set of plausible models is not a *prediction*; rather, it provides a 
computational experiment that reveals how the world would behave if the 
various guesses any particular model makes about the various unresolvable 
uncertainties were correct. EMA is the explicit representation of the set of 
plausible models, the process of exploiting the information contained in such 
a set through a large number of computational experiments, and the analysis of 
the results of these experiments.

A set, universe, or ensemble of models that are plausible or interesting in the 
context of the research or analysis being conducted is generated by the 
uncertainties associated with the problem of interest, and is constrained by 
available data and knowledge. ExploratoryModelingAndAnalysis can be 
viewed as a means for inference from the constraint information that specifies 
this set or ensemble. Selecting a particular model out of an ensemble of 
plausible ones requires making suppositions about factors that are uncertain or 
unknown. One such computational experiment is typically not that informative 
(beyond suggesting the plausibility of its outcomes). Instead, EMA supports 
reasoning about general conclusions through the examination of the results of 
numerous such experiments. Thus, EMA can be understood as search or sampling 
over the ensemble of models that are plausible given a priori knowledge.
   



