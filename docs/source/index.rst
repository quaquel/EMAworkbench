.. EMA workbench documentation master file, created by
   sphinx-quickstart on Wed Sep 07 13:56:32 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. meta::
   :description: A python library for exploratory modeling and analysis for 
                 supporting model based decision making under deep uncertainty
   :keywords: exploratory modeling, deep uncertainty, robust decision making,
              vensim, python

***************************
EMA Workbench documentation
***************************

.. htmlonly::

    :Release: |version|
    :Date: |today|

.. _contents:

.. _exploratory-modelling-analysis-workbench:

**************************************************
Exploratory Modelling and Analysis (EMA) Workbench
**************************************************

Exploratory Modeling and Analysis (EMA) is a research methodology that uses 
computational experiments to analyze complex and uncertain systems 
(`Bankes, 1993 <http://www.jstor.org/stable/10.2307/171847>`_). That is, 
exploratory modeling aims at offering computational decision support for 
decision making under `deep uncertainty <http://inderscience.metapress.com/content/y77p3q512x475523/>`_ 
and `Robust decision making <http://en.wikipedia.org/wiki/Robust_decision_making>`_.  

The EMA workbench aims at providing support for performing exploratory
modeling with models developed in various modelling packages and environments. 
Currently, the workbench offers connectors to 
`Vensim <http://www.vensim.com/>`_, `Netlogo <http://ccl.northwestern.edu/netlogo/>`_, 
and Excel. 

The EMA workbench offers support for designing experiments, performing the 
experiments - including support for parallel processing on both a single 
machine as well as on clusters-, and analysing the results. To get started, 
take a look at the high level overview, the tutorial, or dive straight into 
the details of the API. 

.. toctree::
   :maxdepth: 2
   
   ./overview.rst
   ./installation.rst
   ./indepth_tutorial/general-introduction.ipynb
   ./indepth_tutorial/open-exploration.ipynb
   ./indepth_tutorial/directed-search.ipynb
   ./basic_tutorial.rst
   ./vensim-tips-and-tricks.rst
   ./api_index.rst
   ./ema_documentation/glossary.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
* :ref:`glossary`

