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

.. only:: html

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
and `Robust Decision Making <http://en.wikipedia.org/wiki/Robust_decision_making>`_.

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

So how does the workbench differ from other open source tools available for
exploratory modeling? For Python, the main alternative tool is `rhodium <https://github.com/Project-Platypus/Rhodium>`_,
which is part of `Project Platypus <https://github.com/Project-Platypus>`_. Project Platypus is a collection of
libraries for doing many objective optimization (`platypus-opt <https://platypus.readthedocs.io/en/latest/>`_), setting
up and performing simulation experiments (`rhodium <https://github.com/Project-Platypus/Rhodium>`_), and
scenario discovery using the Patient Rule Induction Method (`prim <https://github.com/Project-Platypus/PRIM>`_). The
relationship between the workbench and the tools that form project platypus is a
bit messy. For example, the workbench too relies on `platypus-opt <https://platypus.readthedocs.io/en/latest/>`_ for many
objective optimization, the `PRIM <https://github.com/Project-Platypus/PRIM>`_ package is a, by now very dated, fork of the
prim code in the workbench, and both `rhodium <https://github.com/Project-Platypus/Rhodium>`_ and the workbench rely on
`SALib <https://salib.readthedocs.io>`_ for global sensitivity analysis. Moreover, the API
of `rhodium <https://github.com/Project-Platypus/Rhodium>`_ was partly inspired by an older version of the
workbench, while new ideas from the rhodium API have in turned resulting in profound changes in the API of the
workbench.

Currently, the workbench is still actively being developed. It is also not just used
in teaching but also for research, and in practice by various organization globally.
Moreover, the workbench is quite a bit more developed when it comes to providing off
the shelf connectors for some popular modeling and simulation tools. Basically,
everything that can be done with project Platypus can be done with the workbench
and then the workbench offers additional functionality, a more up to date code
base, and active support.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   ./overview
   ./installation
   ./changelog

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   ./basic_tutorial.rst
   ./indepth_tutorial/general-introduction.ipynb
   ./indepth_tutorial/open-exploration.ipynb
   ./indepth_tutorial/directed-search.ipynb
   ./best_practices.rst
   ./vensim-tips-and-tricks.rst
   ./ema_documentation/glossary.rst

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   ./api_index.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
* :ref:`glossary`